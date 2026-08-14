#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace {

using Clock = std::chrono::steady_clock;
using Image = sensor_msgs::msg::Image;

std::string av_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

int64_t steady_nanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

AVPixelFormat pixel_format(const std::string &encoding) {
  if (encoding == "rgb8") {
    return AV_PIX_FMT_RGB24;
  }
  if (encoding == "bgr8") {
    return AV_PIX_FMT_BGR24;
  }
  if (encoding == "rgba8") {
    return AV_PIX_FMT_RGBA;
  }
  if (encoding == "bgra8") {
    return AV_PIX_FMT_BGRA;
  }
  if (encoding == "mono8" || encoding == "8UC1") {
    return AV_PIX_FMT_GRAY8;
  }
  return AV_PIX_FMT_NONE;
}

int bytes_per_pixel(AVPixelFormat format) {
  switch (format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      return 3;
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_BGRA:
      return 4;
    case AV_PIX_FMT_GRAY8:
      return 1;
    default:
      return 0;
  }
}

bool tcp_port_ready(const std::string &host, int port, int timeout_ms) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *addresses = nullptr;
  const std::string service = std::to_string(port);
  if (getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses) != 0) {
    return false;
  }
  bool ready = false;
  for (addrinfo *address = addresses; address != nullptr && !ready;
       address = address->ai_next) {
    const int socket_fd =
        socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (socket_fd < 0) {
      continue;
    }
    const int original_flags = fcntl(socket_fd, F_GETFL, 0);
    if (original_flags >= 0) {
      fcntl(socket_fd, F_SETFL, original_flags | O_NONBLOCK);
    }
    int result = connect(socket_fd, address->ai_addr, address->ai_addrlen);
    if (result == 0) {
      ready = true;
    } else if (errno == EINPROGRESS) {
      pollfd descriptor{socket_fd, POLLOUT, 0};
      result = poll(&descriptor, 1, timeout_ms);
      if (result > 0 && (descriptor.revents & POLLOUT) != 0) {
        int error = 0;
        socklen_t error_size = sizeof(error);
        ready = getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, &error,
                           &error_size) == 0 &&
                error == 0;
      }
    }
    close(socket_fd);
  }
  freeaddrinfo(addresses);
  return ready;
}

struct EncoderConfig {
  std::string url;
  std::string transport;
  std::string encoder;
  int width = 424;
  int height = 240;
  int fps = 60;
  int bitrate = 3000000;
  int gop_size = 15;
  double network_timeout_s = 2.0;
};

class RtspEncoder {
 public:
  explicit RtspEncoder(EncoderConfig config) : config_(std::move(config)) {}

  RtspEncoder(const RtspEncoder &) = delete;
  RtspEncoder &operator=(const RtspEncoder &) = delete;

  ~RtspEncoder() { close(); }

  bool is_open() const { return format_context_ != nullptr && header_written_; }

  void request_interrupt() { interrupt_requested_.store(true); }

  void open() {
    close();
    interrupt_requested_.store(false);
    const int format_result = avformat_alloc_output_context2(
        &format_context_, nullptr, "rtsp", config_.url.c_str());
    if (format_result < 0 || format_context_ == nullptr) {
      throw std::runtime_error("cannot allocate RTSP output: " +
                               av_error(format_result));
    }
    format_context_->interrupt_callback.callback = &RtspEncoder::interrupt;
    format_context_->interrupt_callback.opaque = this;

    const AVCodec *codec = avcodec_find_encoder_by_name(config_.encoder.c_str());
    if (codec == nullptr) {
      throw std::runtime_error("FFmpeg encoder is unavailable: " +
                               config_.encoder);
    }
    codec_context_ = avcodec_alloc_context3(codec);
    if (codec_context_ == nullptr) {
      throw std::runtime_error("cannot allocate FFmpeg codec context");
    }
    codec_context_->codec_id = codec->id;
    codec_context_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_context_->width = config_.width;
    codec_context_->height = config_.height;
    codec_context_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_context_->time_base = AVRational{1, config_.fps};
    codec_context_->framerate = AVRational{config_.fps, 1};
    codec_context_->bit_rate = config_.bitrate;
    codec_context_->gop_size = config_.gop_size;
    codec_context_->max_b_frames = 0;
    codec_context_->thread_count = 1;
    if ((format_context_->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
      codec_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    AVDictionary *codec_options = nullptr;
    av_dict_set(&codec_options, "preset", "ultrafast", 0);
    av_dict_set(&codec_options, "tune", "zerolatency", 0);
    const int codec_result =
        avcodec_open2(codec_context_, codec, &codec_options);
    av_dict_free(&codec_options);
    if (codec_result < 0) {
      throw std::runtime_error("cannot open FFmpeg encoder '" +
                               config_.encoder + "': " +
                               av_error(codec_result));
    }

    stream_ = avformat_new_stream(format_context_, nullptr);
    if (stream_ == nullptr) {
      throw std::runtime_error("cannot allocate RTSP video stream");
    }
    stream_->time_base = codec_context_->time_base;
    const int parameters_result =
        avcodec_parameters_from_context(stream_->codecpar, codec_context_);
    if (parameters_result < 0) {
      throw std::runtime_error("cannot copy encoder parameters: " +
                               av_error(parameters_result));
    }

    if ((format_context_->oformat->flags & AVFMT_NOFILE) == 0) {
      arm_deadline();
      const int io_result = avio_open2(&format_context_->pb, config_.url.c_str(),
                                       AVIO_FLAG_WRITE, nullptr, nullptr);
      if (io_result < 0) {
        throw std::runtime_error("cannot open RTSP output: " +
                                 av_error(io_result));
      }
    }

    AVDictionary *muxer_options = nullptr;
    av_dict_set(&muxer_options, "rtsp_transport", config_.transport.c_str(), 0);
    av_dict_set(&muxer_options, "muxdelay", "0", 0);
    const std::string timeout_us =
        std::to_string(static_cast<int64_t>(config_.network_timeout_s * 1.0e6));
    av_dict_set(&muxer_options, "rw_timeout", timeout_us.c_str(), 0);
    arm_deadline();
    const int header_result =
        avformat_write_header(format_context_, &muxer_options);
    av_dict_free(&muxer_options);
    if (header_result < 0) {
      throw std::runtime_error("cannot publish RTSP header: " +
                               av_error(header_result));
    }
    header_written_ = true;

    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (frame_ == nullptr || packet_ == nullptr) {
      throw std::runtime_error("cannot allocate FFmpeg frame or packet");
    }
    frame_->format = codec_context_->pix_fmt;
    frame_->width = codec_context_->width;
    frame_->height = codec_context_->height;
    const int buffer_result = av_frame_get_buffer(frame_, 32);
    if (buffer_result < 0) {
      throw std::runtime_error("cannot allocate output image buffer: " +
                               av_error(buffer_result));
    }
    first_frame_time_.reset();
    last_pts_ = -1;
  }

  void write(const Image &image, Clock::time_point received) {
    if (!is_open()) {
      throw std::runtime_error("RTSP encoder is not open");
    }
    const AVPixelFormat input_format = pixel_format(image.encoding);
    const int pixel_size = bytes_per_pixel(input_format);
    if (input_format == AV_PIX_FMT_NONE) {
      throw std::invalid_argument("unsupported ROS image encoding: " +
                                  image.encoding);
    }
    if (image.width == 0 || image.height == 0 ||
        image.step < image.width * static_cast<uint32_t>(pixel_size) ||
        image.data.size() < static_cast<size_t>(image.step) * image.height) {
      throw std::invalid_argument("ROS image dimensions, step, or data are invalid");
    }

    scaler_ = sws_getCachedContext(
        scaler_, static_cast<int>(image.width), static_cast<int>(image.height),
        input_format, config_.width, config_.height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (scaler_ == nullptr) {
      throw std::runtime_error("cannot create FFmpeg image scaler");
    }
    const int writable_result = av_frame_make_writable(frame_);
    if (writable_result < 0) {
      throw std::runtime_error("output frame is not writable: " +
                               av_error(writable_result));
    }
    const uint8_t *source_data[4] = {image.data.data(), nullptr, nullptr,
                                     nullptr};
    const int source_linesize[4] = {static_cast<int>(image.step), 0, 0, 0};
    const int scaled_rows = sws_scale(
        scaler_, source_data, source_linesize, 0, static_cast<int>(image.height),
        frame_->data, frame_->linesize);
    if (scaled_rows != config_.height) {
      throw std::runtime_error("FFmpeg image conversion returned an incomplete frame");
    }

    if (!first_frame_time_.has_value()) {
      first_frame_time_ = received;
    }
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                received - *first_frame_time_)
                                .count();
    const int64_t clock_pts = av_rescale_q(
        elapsed_ns, AVRational{1, 1000000000}, codec_context_->time_base);
    frame_->pts = std::max(last_pts_ + 1, clock_pts);
    last_pts_ = frame_->pts;

    const int send_result = avcodec_send_frame(codec_context_, frame_);
    if (send_result < 0) {
      throw std::runtime_error("H.264 encoder rejected a frame: " +
                               av_error(send_result));
    }
    drain_packets();
  }

  void close() noexcept {
    interrupt_requested_.store(true);
    if (packet_ != nullptr) {
      av_packet_free(&packet_);
    }
    if (frame_ != nullptr) {
      av_frame_free(&frame_);
    }
    if (scaler_ != nullptr) {
      sws_freeContext(scaler_);
      scaler_ = nullptr;
    }
    if (codec_context_ != nullptr) {
      avcodec_free_context(&codec_context_);
    }
    if (format_context_ != nullptr) {
      if ((format_context_->oformat->flags & AVFMT_NOFILE) == 0 &&
          format_context_->pb != nullptr) {
        avio_closep(&format_context_->pb);
      }
      avformat_free_context(format_context_);
      format_context_ = nullptr;
    }
    stream_ = nullptr;
    header_written_ = false;
    first_frame_time_.reset();
    last_pts_ = -1;
  }

 private:
  static int interrupt(void *opaque) {
    const auto *self = static_cast<RtspEncoder *>(opaque);
    return self->interrupt_requested_.load() ||
                   steady_nanoseconds() > self->deadline_ns_.load()
               ? 1
               : 0;
  }

  void arm_deadline() {
    deadline_ns_.store(steady_nanoseconds() + static_cast<int64_t>(
                                                  config_.network_timeout_s *
                                                  1.0e9));
  }

  void drain_packets() {
    while (true) {
      const int receive_result =
          avcodec_receive_packet(codec_context_, packet_);
      if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
        return;
      }
      if (receive_result < 0) {
        throw std::runtime_error("cannot receive H.264 packet: " +
                                 av_error(receive_result));
      }
      av_packet_rescale_ts(packet_, codec_context_->time_base,
                           stream_->time_base);
      packet_->stream_index = stream_->index;
      arm_deadline();
      const int write_result =
          av_interleaved_write_frame(format_context_, packet_);
      av_packet_unref(packet_);
      if (write_result < 0) {
        throw std::runtime_error("cannot write RTSP packet: " +
                                 av_error(write_result));
      }
    }
  }

  EncoderConfig config_;
  AVFormatContext *format_context_ = nullptr;
  AVCodecContext *codec_context_ = nullptr;
  AVStream *stream_ = nullptr;
  AVFrame *frame_ = nullptr;
  AVPacket *packet_ = nullptr;
  SwsContext *scaler_ = nullptr;
  bool header_written_ = false;
  std::optional<Clock::time_point> first_frame_time_;
  int64_t last_pts_ = -1;
  std::atomic<bool> interrupt_requested_{false};
  std::atomic<int64_t> deadline_ns_{0};
};

enum class Source { kNone, kSimulation, kHardware };

const char *source_name(Source source) {
  switch (source) {
    case Source::kSimulation:
      return "simulation";
    case Source::kHardware:
      return "hardware";
    default:
      return "none";
  }
}

struct FrameSlot {
  Image::ConstSharedPtr image;
  Clock::time_point received{};
  uint64_t sequence = 0;
  int consecutive = 0;
};

struct SelectedFrame {
  Source source = Source::kNone;
  Image::ConstSharedPtr image;
  Clock::time_point received{};
  uint64_t sequence = 0;
};

class HeadCameraRtspNode : public rclcpp::Node {
 public:
  HeadCameraRtspNode() : Node("head_camera_rtsp") {
    simulation_topic_ = declare_parameter<std::string>(
        "simulation_topic", "/simulation/head_depth_camera/color/image_raw");
    hardware_topic_ = declare_parameter<std::string>(
        "hardware_topic", "/hardware/head_depth_camera/color/image_raw");
    source_mode_ = declare_parameter<std::string>("source_mode", "auto");
    source_timeout_s_ = declare_parameter<double>("source_timeout_s", 0.5);
    hardware_promote_frames_ =
        declare_parameter<int>("hardware_promote_frames", 3);
    readiness_host_ =
        declare_parameter<std::string>("readiness_host", "127.0.0.1");
    readiness_port_ = declare_parameter<int>("readiness_port", 2212);
    statistics_interval_s_ =
        declare_parameter<double>("statistics_interval_s", 5.0);
    config_.url = declare_parameter<std::string>(
        "rtsp_url", "rtsp://127.0.0.1:2212/video");
    config_.transport =
        declare_parameter<std::string>("rtsp_transport", "udp");
    config_.encoder = declare_parameter<std::string>("encoder", "libx264");
    config_.width = declare_parameter<int>("output_width", 424);
    config_.height = declare_parameter<int>("output_height", 240);
    config_.fps = declare_parameter<int>("output_fps", 60);
    config_.bitrate = declare_parameter<int>("bitrate", 3000000);
    config_.gop_size = declare_parameter<int>("gop_size", 15);
    config_.network_timeout_s =
        declare_parameter<double>("network_timeout_s", 2.0);
    validate_parameters();

    rclcpp::SensorDataQoS qos;
    qos.keep_last(1);
    simulation_subscription_ = create_subscription<Image>(
        simulation_topic_, qos,
        [this](Image::ConstSharedPtr image) {
          receive(Source::kSimulation, std::move(image));
        });
    hardware_subscription_ = create_subscription<Image>(
        hardware_topic_, qos,
        [this](Image::ConstSharedPtr image) {
          receive(Source::kHardware, std::move(image));
        });
    encoder_ = std::make_unique<RtspEncoder>(config_);
    worker_ = std::thread([this] { worker_loop(); });
    RCLCPP_INFO(get_logger(),
                "head camera RTSP node ready: mode=%s, simulation=%s, "
                "hardware=%s, output=%s",
                source_mode_.c_str(), simulation_topic_.c_str(),
                hardware_topic_.c_str(), config_.url.c_str());
  }

  ~HeadCameraRtspNode() override {
    stop_.store(true);
    condition_.notify_all();
    if (encoder_ != nullptr) {
      encoder_->request_interrupt();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  void validate_parameters() {
    if (source_mode_ != "auto" && source_mode_ != "simulation" &&
        source_mode_ != "hardware") {
      throw std::invalid_argument(
          "source_mode must be auto, simulation, or hardware");
    }
    if (simulation_topic_.empty() || hardware_topic_.empty() ||
        config_.url.empty() || readiness_host_.empty()) {
      throw std::invalid_argument("topics, RTSP URL, and readiness host are required");
    }
    if (source_timeout_s_ <= 0.0 || statistics_interval_s_ <= 0.0 ||
        config_.network_timeout_s <= 0.0 || readiness_port_ <= 0 ||
        readiness_port_ > 65535 || hardware_promote_frames_ < 1 ||
        config_.width < 2 || config_.height < 2 || config_.width % 2 != 0 ||
        config_.height % 2 != 0 || config_.fps < 1 || config_.bitrate < 1 ||
        config_.gop_size < 1) {
      throw std::invalid_argument("invalid RTSP timing, port, or encoder parameters");
    }
    if (config_.transport != "udp" && config_.transport != "tcp") {
      throw std::invalid_argument("rtsp_transport must be udp or tcp");
    }
  }

  void receive(Source source, Image::ConstSharedPtr image) {
    const Clock::time_point now = Clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      FrameSlot &slot =
          source == Source::kHardware ? hardware_slot_ : simulation_slot_;
      if (slot.image == nullptr ||
          std::chrono::duration<double>(now - slot.received).count() >
              source_timeout_s_) {
        slot.consecutive = 1;
      } else {
        ++slot.consecutive;
      }
      slot.image = std::move(image);
      slot.received = now;
      ++slot.sequence;
    }
    if (source == Source::kHardware) {
      ++hardware_received_;
    } else {
      ++simulation_received_;
    }
    condition_.notify_one();
  }

  bool fresh(const FrameSlot &slot, Clock::time_point now) const {
    return slot.image != nullptr &&
           std::chrono::duration<double>(now - slot.received).count() <=
               source_timeout_s_;
  }

  SelectedFrame select_frame(Clock::time_point now) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool simulation_fresh = fresh(simulation_slot_, now);
    const bool hardware_fresh = fresh(hardware_slot_, now);
    Source selected = Source::kNone;
    if (source_mode_ == "simulation") {
      selected = simulation_fresh ? Source::kSimulation : Source::kNone;
    } else if (source_mode_ == "hardware") {
      selected = hardware_fresh ? Source::kHardware : Source::kNone;
    } else if (hardware_fresh &&
               hardware_slot_.consecutive >= hardware_promote_frames_) {
      selected = Source::kHardware;
    } else if (simulation_fresh) {
      selected = Source::kSimulation;
    } else if (hardware_fresh) {
      selected = Source::kHardware;
    }
    const FrameSlot *slot = nullptr;
    if (selected == Source::kHardware) {
      slot = &hardware_slot_;
    } else if (selected == Source::kSimulation) {
      slot = &simulation_slot_;
    }
    if (slot == nullptr) {
      return {};
    }
    return SelectedFrame{selected, slot->image, slot->received, slot->sequence};
  }

  bool is_new(const SelectedFrame &frame) const {
    if (frame.source == Source::kHardware) {
      return frame.sequence != consumed_hardware_;
    }
    if (frame.source == Source::kSimulation) {
      return frame.sequence != consumed_simulation_;
    }
    return false;
  }

  void mark_consumed(const SelectedFrame &frame) {
    uint64_t *consumed = frame.source == Source::kHardware
                             ? &consumed_hardware_
                             : &consumed_simulation_;
    if (*consumed != 0 && frame.sequence > *consumed + 1) {
      dropped_ += frame.sequence - *consumed - 1;
    }
    *consumed = frame.sequence;
  }

  void worker_loop() {
    Clock::time_point retry_after{};
    Clock::time_point statistics_started = Clock::now();
    Source active_source = Source::kNone;
    while (!stop_.load()) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(100));
      }
      if (stop_.load()) {
        break;
      }
      const Clock::time_point now = Clock::now();
      const SelectedFrame frame = select_frame(now);
      if (frame.source == Source::kNone) {
        if (active_source != Source::kNone) {
          RCLCPP_WARN(get_logger(),
                      "head camera source is stale; closing RTSP publisher");
          active_source = Source::kNone;
          encoder_->close();
        }
        report_statistics(now, statistics_started, active_source);
        continue;
      }
      if (frame.source != active_source) {
        active_source = frame.source;
        RCLCPP_INFO(get_logger(), "head camera RTSP source: %s",
                    source_name(active_source));
      }
      if (!is_new(frame)) {
        report_statistics(now, statistics_started, active_source);
        continue;
      }
      if (now < retry_after) {
        continue;
      }
      if (!encoder_->is_open()) {
        if (!tcp_port_ready(readiness_host_, readiness_port_, 200)) {
          retry_after = now + std::chrono::seconds(1);
          if (now >= next_server_warning_) {
            RCLCPP_WARN(get_logger(),
                        "waiting for MediaMTX at %s:%d before publishing",
                        readiness_host_.c_str(), readiness_port_);
            next_server_warning_ = now + std::chrono::seconds(5);
          }
          continue;
        }
        try {
          encoder_->open();
          RCLCPP_INFO(get_logger(),
                      "publishing H.264 %dx%d to %s with encoder=%s transport=%s",
                      config_.width, config_.height, config_.url.c_str(),
                      config_.encoder.c_str(), config_.transport.c_str());
        } catch (const std::exception &error) {
          ++stream_errors_;
          encoder_->close();
          retry_after = now + std::chrono::seconds(1);
          RCLCPP_WARN(get_logger(), "cannot open RTSP publisher: %s",
                      error.what());
          continue;
        }
      }
      try {
        encoder_->write(*frame.image, frame.received);
        mark_consumed(frame);
        ++encoded_;
      } catch (const std::invalid_argument &error) {
        mark_consumed(frame);
        ++invalid_frames_;
        if (now >= next_frame_warning_) {
          RCLCPP_WARN(get_logger(), "head camera frame rejected: %s", error.what());
          next_frame_warning_ = now + std::chrono::seconds(5);
        }
      } catch (const std::exception &error) {
        ++stream_errors_;
        encoder_->close();
        retry_after = now + std::chrono::seconds(1);
        RCLCPP_WARN(get_logger(), "RTSP stream interrupted, reconnecting: %s",
                    error.what());
      }
      report_statistics(now, statistics_started, active_source);
    }
    encoder_->request_interrupt();
    encoder_->close();
  }

  void report_statistics(Clock::time_point now, Clock::time_point &started,
                         Source source) {
    const double elapsed = std::chrono::duration<double>(now - started).count();
    if (elapsed < statistics_interval_s_) {
      return;
    }
    const uint64_t simulation = simulation_received_.exchange(0);
    const uint64_t hardware = hardware_received_.exchange(0);
    const uint64_t encoded = encoded_.exchange(0);
    const uint64_t dropped = dropped_.exchange(0);
    const uint64_t invalid = invalid_frames_.exchange(0);
    const uint64_t errors = stream_errors_.exchange(0);
    RCLCPP_INFO(get_logger(),
                "head camera RTSP perf: source=%s simulation=%.1fHz "
                "hardware=%.1fHz encoded=%.1fHz dropped=%lu invalid=%lu errors=%lu",
                source_name(source), simulation / elapsed, hardware / elapsed,
                encoded / elapsed, static_cast<unsigned long>(dropped),
                static_cast<unsigned long>(invalid),
                static_cast<unsigned long>(errors));
    started = now;
  }

  EncoderConfig config_;
  std::string simulation_topic_;
  std::string hardware_topic_;
  std::string source_mode_;
  std::string readiness_host_;
  int readiness_port_ = 2212;
  double source_timeout_s_ = 0.5;
  int hardware_promote_frames_ = 3;
  double statistics_interval_s_ = 5.0;

  rclcpp::Subscription<Image>::SharedPtr simulation_subscription_;
  rclcpp::Subscription<Image>::SharedPtr hardware_subscription_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  FrameSlot simulation_slot_;
  FrameSlot hardware_slot_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
  std::unique_ptr<RtspEncoder> encoder_;
  uint64_t consumed_simulation_ = 0;
  uint64_t consumed_hardware_ = 0;
  std::atomic<uint64_t> simulation_received_{0};
  std::atomic<uint64_t> hardware_received_{0};
  std::atomic<uint64_t> encoded_{0};
  std::atomic<uint64_t> dropped_{0};
  std::atomic<uint64_t> invalid_frames_{0};
  std::atomic<uint64_t> stream_errors_{0};
  Clock::time_point next_server_warning_{};
  Clock::time_point next_frame_warning_{};
};

}  // namespace

int main(int argc, char **argv) {
  av_log_set_level(AV_LOG_WARNING);
  avformat_network_init();
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    auto node = std::make_shared<HeadCameraRtspNode>();
    rclcpp::spin(node);
    node.reset();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "head_camera_rtsp_node: %s\n", error.what());
    result = 78;
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  avformat_network_deinit();
  return result;
}
