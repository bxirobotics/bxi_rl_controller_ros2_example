#include "bxi_depth_camera/camera_worker.hpp"

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <cstring>
#include <iterator>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace bxi_depth_camera
{
namespace
{

struct RealSenseDiscoveryContext {
    // Keep generation alive until after the SDK context has shut down its
    // callback thread (members are destroyed in reverse declaration order).
    std::atomic<std::uint64_t> generation{ 1 };
    rs2::context context;

    RealSenseDiscoveryContext()
    {
        context.set_devices_changed_callback(
            [this](rs2::event_information &) {
                generation.fetch_add(1, std::memory_order_relaxed);
            });
    }
};

RealSenseDiscoveryContext &realsense_discovery_context()
{
    static RealSenseDiscoveryContext instance;
    return instance;
}

Calibration calibration_from(const rs2_intrinsics &intrinsics)
{
    Calibration calibration;
    calibration.fx = intrinsics.fx;
    calibration.fy = intrinsics.fy;
    calibration.cx = intrinsics.ppx;
    calibration.cy = intrinsics.ppy;
    calibration.distortion.assign(std::begin(intrinsics.coeffs),
                                  std::end(intrinsics.coeffs));
    calibration.distortion_model =
        intrinsics.model == RS2_DISTORTION_KANNALA_BRANDT4 ||
                intrinsics.model == RS2_DISTORTION_FTHETA ?
            "equidistant" :
            "plumb_bob";
    return calibration;
}

std::string stream_name(rs2_stream stream)
{
    return rs2_stream_to_string(stream);
}

std::string format_name(rs2_format format)
{
    return rs2_format_to_string(format);
}

std::string sensor_name(const rs2::sensor &sensor)
{
    return sensor.supports(RS2_CAMERA_INFO_NAME) ?
               sensor.get_info(RS2_CAMERA_INFO_NAME) :
               "unknown RealSense sensor";
}

std::string video_profile_name(rs2_stream stream, int index, rs2_format format,
                               const StreamProfile &profile)
{
    std::ostringstream out;
    out << stream_name(stream);
    if (index >= 0) {
        out << ":" << index;
    }
    out << " " << format_name(format);
    if (profile.automatic()) {
        out << " automatic";
    } else {
        out << " " << profile.width << "x" << profile.height << "x"
            << profile.fps;
    }
    return out.str();
}

std::optional<rs2::stream_profile> choose_video_profile(
    const std::vector<rs2::stream_profile> &profiles, rs2_stream stream,
    int index, rs2_format format, const StreamProfile &wanted)
{
    std::optional<rs2::stream_profile> fallback;
    for (const auto &profile : profiles) {
        if (profile.stream_type() != stream || profile.format() != format) {
            continue;
        }
        if (index >= 0 && profile.stream_index() != index) {
            continue;
        }
        const auto video = profile.as<rs2::video_stream_profile>();
        if (!video) {
            continue;
        }
        if (!wanted.automatic()) {
            if (video.width() == wanted.width &&
                video.height() == wanted.height &&
                profile.fps() == wanted.fps) {
                return profile;
            }
            continue;
        }
        if (profile.is_default()) {
            return profile;
        }
        if (!fallback) {
            fallback = profile;
        }
    }
    return fallback;
}

std::optional<rs2::stream_profile>
choose_motion_profile(const std::vector<rs2::stream_profile> &profiles,
                      rs2_stream stream)
{
    std::optional<rs2::stream_profile> fallback;
    for (const auto &profile : profiles) {
        if (profile.stream_type() != stream ||
            profile.format() != RS2_FORMAT_MOTION_XYZ32F) {
            continue;
        }
        if (profile.is_default()) {
            return profile;
        }
        if (!fallback) {
            fallback = profile;
        }
    }
    return fallback;
}

class RealSenseCamera final : public CameraWorker {
public:
    RealSenseCamera(rclcpp::Node &node, DeviceDescriptor descriptor,
                    std::string logical_name, CameraConfig config)
        : CameraWorker(node, std::move(descriptor), std::move(logical_name),
                       std::move(config))
        , align_to_color_(RS2_STREAM_COLOR)
        , hole_filter_(config_.hole_filling_mode)
        , second_hole_filter_(config_.second_hole_filling_mode)
    {
        try {
            start_sensors();
            video_thread_ =
                std::thread(&RealSenseCamera::run_video_worker, this);
        } catch (...) {
            stop_color_worker();
            throw;
        }
    }

    ~RealSenseCamera() override
    {
        stop();
    }

    void stop() noexcept override
    {
        if (local_stopped_.exchange(true)) {
            return;
        }
        stop_sensors();
        stop_color_worker();
        {
            std::lock_guard<std::mutex> lock(video_mutex_);
            video_stopping_ = true;
            latest_frameset_.reset();
        }
        video_cv_.notify_all();
        if (video_thread_.joinable()) {
            video_thread_.join();
        }
        CameraWorker::stop();
    }

private:
    struct PendingFrameset {
        rs2::frameset frameset;
        rclcpp::Time stamp;
    };

    struct PendingColorFrame {
        rs2::video_frame frame;
        rclcpp::Time stamp;
    };

    struct ActiveSensor {
        rs2::sensor sensor;
        std::string name;
        bool started{ false };
        bool opened{ false };
    };

    struct MotionSample {
        rclcpp::Time stamp;
        float x{ 0.0F };
        float y{ 0.0F };
        float z{ 0.0F };
    };

    rclcpp::Time frame_stamp(const rs2::frame &frame)
    {
        const double timestamp_ms = frame.get_timestamp();
        if (frame.get_frame_timestamp_domain() !=
            RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK) {
            return rclcpp::Time(
                static_cast<std::int64_t>(std::llround(timestamp_ms * 1.0e6)));
        }

        std::lock_guard<std::mutex> lock(time_mutex_);
        if (!time_base_initialized_ || previous_camera_time_ms_ > timestamp_ms) {
            ros_time_base_ns_ = node_.now().nanoseconds();
            camera_time_base_ms_ = timestamp_ms;
            time_base_initialized_ = true;
        }
        previous_camera_time_ms_ = timestamp_ms;
        const auto elapsed_ns = static_cast<std::int64_t>(
            std::llround((timestamp_ms - camera_time_base_ms_) * 1.0e6));
        return rclcpp::Time(ros_time_base_ns_ + elapsed_ns,
                            node_.get_clock()->get_clock_type());
    }

    rs2::device find_device()
    {
        auto devices = context_.query_devices();
        for (auto device : devices) {
            if (!device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
                continue;
            }
            if (descriptor_.serial ==
                device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
                return device;
            }
        }
        throw std::runtime_error("RealSense serial " + descriptor_.serial +
                                 " is no longer present");
    }

    bool depth_filters_enabled() const noexcept
    {
        return config_.decimation_enabled || config_.threshold_enabled ||
               config_.spatial_enabled ||
               config_.temporal_enabled || config_.hole_filling_enabled ||
               config_.second_hole_filling_enabled;
    }

    bool frameset_worker_requested() const
    {
        return aligned_depth_requested() || rgbd_requested() ||
               (config_.enable_sync && video_consumers_requested()) ||
               (pub_pointcloud_ &&
                pub_pointcloud_->get_subscription_count() > 0);
    }

    void require_profile(bool condition, const std::string &description)
    {
        if (!condition) {
            throw std::runtime_error("RealSense " + descriptor_.serial +
                                     " does not provide requested " +
                                     description);
        }
    }

    std::vector<rs2::stream_profile>
    requested_profiles_for_sensor(const rs2::sensor &sensor,
                                  bool &found_depth, bool &found_color,
                                  bool &found_infra1, bool &found_infra2,
                                  bool &found_gyro, bool &found_accel)
    {
        const auto available = sensor.get_stream_profiles();
        std::vector<rs2::stream_profile> wanted;

        if (config_.enable_depth) {
            if (auto profile = choose_video_profile(
                    available, RS2_STREAM_DEPTH, -1, RS2_FORMAT_Z16,
                    config_.depth_profile)) {
                wanted.push_back(*profile);
                found_depth = true;
            }
        }
        if (config_.enable_color) {
            if (auto profile = choose_video_profile(
                    available, RS2_STREAM_COLOR, -1, RS2_FORMAT_BGR8,
                    config_.color_profile)) {
                wanted.push_back(*profile);
                found_color = true;
            }
        }
        if (config_.enable_infra1) {
            if (auto profile = choose_video_profile(
                    available, RS2_STREAM_INFRARED, 1, RS2_FORMAT_Y8,
                    config_.depth_profile)) {
                wanted.push_back(*profile);
                found_infra1 = true;
            }
        }
        if (config_.enable_infra2) {
            if (auto profile = choose_video_profile(
                    available, RS2_STREAM_INFRARED, 2, RS2_FORMAT_Y8,
                    config_.depth_profile)) {
                wanted.push_back(*profile);
                found_infra2 = true;
            }
        }
        if (config_.enable_gyro) {
            if (auto profile = choose_motion_profile(available,
                                                     RS2_STREAM_GYRO)) {
                wanted.push_back(*profile);
                found_gyro = true;
            }
        }
        if (config_.enable_accel) {
            if (auto profile = choose_motion_profile(available,
                                                     RS2_STREAM_ACCEL)) {
                wanted.push_back(*profile);
                found_accel = true;
            }
        }
        return wanted;
    }

    void start_sensors()
    {
        device_ = find_device();
        configure_device(device_);
        configure_filters();

        bool found_depth = !config_.enable_depth;
        bool found_color = !config_.enable_color;
        bool found_infra1 = !config_.enable_infra1;
        bool found_infra2 = !config_.enable_infra2;
        bool found_gyro = !config_.enable_gyro;
        bool found_accel = !config_.enable_accel;

        try {
            for (auto sensor : device_.query_sensors()) {
                auto wanted = requested_profiles_for_sensor(
                    sensor, found_depth, found_color, found_infra1,
                    found_infra2, found_gyro, found_accel);
                if (wanted.empty()) {
                    continue;
                }

                ActiveSensor active;
                active.sensor = sensor;
                active.name = sensor_name(sensor);
                active.sensor.open(wanted);
                active.opened = true;
                active_sensors_.push_back(std::move(active));
                auto &stored = active_sensors_.back();
                stored.sensor.start([this](rs2::frame frame) {
                    on_frame(std::move(frame));
                });
                stored.started = true;
            }

            require_profile(found_depth,
                            video_profile_name(RS2_STREAM_DEPTH, -1,
                                               RS2_FORMAT_Z16,
                                               config_.depth_profile));
            require_profile(found_color,
                            video_profile_name(RS2_STREAM_COLOR, -1,
                                               RS2_FORMAT_BGR8,
                                               config_.color_profile));
            require_profile(found_infra1,
                            video_profile_name(RS2_STREAM_INFRARED, 1,
                                               RS2_FORMAT_Y8,
                                               config_.depth_profile));
            require_profile(found_infra2,
                            video_profile_name(RS2_STREAM_INFRARED, 2,
                                               RS2_FORMAT_Y8,
                                               config_.depth_profile));
            require_profile(found_gyro,
                            std::string(stream_name(RS2_STREAM_GYRO)) +
                                " " + format_name(RS2_FORMAT_MOTION_XYZ32F));
            require_profile(found_accel,
                            std::string(stream_name(RS2_STREAM_ACCEL)) +
                                " " + format_name(RS2_FORMAT_MOTION_XYZ32F));
        } catch (...) {
            stop_sensors();
            throw;
        }

        RCLCPP_INFO(node_.get_logger(), "started RealSense %s serial=%s",
                    descriptor_.name.c_str(), descriptor_.serial.c_str());
    }

    void stop_sensors() noexcept
    {
        for (auto it = active_sensors_.rbegin(); it != active_sensors_.rend();
             ++it) {
            if (it->started) {
                try {
                    it->sensor.stop();
                } catch (const std::exception &error) {
                    warn_throttled("realsense-stop-" + it->name,
                                   "cannot stop " + it->name + ": " +
                                       error.what());
                }
                it->started = false;
            }
            if (it->opened) {
                try {
                    it->sensor.close();
                } catch (const std::exception &error) {
                    warn_throttled("realsense-close-" + it->name,
                                   "cannot close " + it->name + ": " +
                                       error.what());
                }
                it->opened = false;
            }
        }
        active_sensors_.clear();
    }

    bool sensor_has_stream(const rs2::sensor &sensor, rs2_stream stream) const
    {
        for (const auto &profile : sensor.get_stream_profiles()) {
            if (profile.stream_type() == stream) {
                return true;
            }
        }
        return false;
    }

    void set_sensor_option(rs2::sensor &sensor, rs2_option option, float value,
                           const std::string &scope)
    {
        if (!sensor.supports(option)) {
            warn_throttled(
                "realsense-option-unsupported-" + scope + "-" +
                    std::to_string(static_cast<int>(option)),
                "RealSense " + descriptor_.serial + " " + scope +
                    " does not support option " + rs2_option_to_string(option));
            return;
        }
        const auto range = sensor.get_option_range(option);
        const float clamped = std::clamp(value, range.min, range.max);
        sensor.set_option(option, clamped);
        if (std::abs(clamped - value) > 1e-6F) {
            warn_throttled(
                "realsense-option-clamped-" + scope + "-" +
                    std::to_string(static_cast<int>(option)),
                "RealSense " + descriptor_.serial + " " + scope + " option " +
                    rs2_option_to_string(option) + " clamped to supported range");
        }
    }

    void apply_exposure_options(rs2::sensor &sensor, const std::string &scope,
                                bool auto_exposure, double exposure,
                                double gain)
    {
        if (!auto_exposure || exposure >= 0.0 || gain >= 0.0) {
            set_sensor_option(sensor, RS2_OPTION_ENABLE_AUTO_EXPOSURE,
                              auto_exposure ? 1.0F : 0.0F, scope);
        }
        if (exposure >= 0.0) {
            set_sensor_option(sensor, RS2_OPTION_EXPOSURE,
                              static_cast<float>(exposure), scope);
        }
        if (gain >= 0.0) {
            set_sensor_option(sensor, RS2_OPTION_GAIN,
                              static_cast<float>(gain), scope);
        }
    }

    bool roi_configured() const noexcept
    {
        return config_.auto_exposure_roi_left >= 0;
    }

    void apply_roi(rs2::sensor &sensor, const std::string &scope)
    {
        if (!roi_configured()) {
            return;
        }
        rs2::roi_sensor roi_sensor(sensor);
        if (!roi_sensor) {
            warn_throttled("realsense-roi-unsupported-" + scope,
                           "RealSense " + descriptor_.serial + " " + scope +
                               " does not support auto-exposure ROI");
            return;
        }
        roi_sensor.set_region_of_interest(rs2::region_of_interest{
            config_.auto_exposure_roi_left, config_.auto_exposure_roi_top,
            config_.auto_exposure_roi_right, config_.auto_exposure_roi_bottom });
    }

    void log_supported_options(rs2::sensor sensor, const std::string &scope) const
    {
        std::ostringstream out;
        bool first = true;
        for (const auto option : sensor.get_supported_options()) {
            if (!first) {
                out << ", ";
            }
            first = false;
            out << rs2_option_to_string(option);
        }
        RCLCPP_DEBUG(node_.get_logger(),
                     "RealSense %s %s supported SDK options: %s",
                     descriptor_.serial.c_str(), scope.c_str(),
                     out.str().c_str());
    }

    void configure_device(const rs2::device &device)
    {
        for (auto sensor : device.query_sensors()) {
            try {
                const bool is_color = sensor_has_stream(sensor, RS2_STREAM_COLOR);
                const bool is_depth = sensor_has_stream(sensor, RS2_STREAM_DEPTH);
                const std::string scope =
                    is_color ? "color" : (is_depth ? "depth" : "sensor");
                log_supported_options(sensor, scope);
                if (sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
                    sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.0F);
                }
                if (sensor.supports(RS2_OPTION_LASER_POWER)) {
                    const auto range =
                        sensor.get_option_range(RS2_OPTION_LASER_POWER);
                    sensor.set_option(RS2_OPTION_LASER_POWER, range.max);
                }
                if (sensor.supports(RS2_OPTION_VISUAL_PRESET)) {
                    sensor.set_option(
                        RS2_OPTION_VISUAL_PRESET,
                        static_cast<float>(
                            RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY));
                }
                if (is_color) {
                    apply_exposure_options(sensor, "color",
                                           config_.color_enable_auto_exposure,
                                           config_.color_exposure,
                                           config_.color_gain);
                    apply_roi(sensor, "color");
                } else if (is_depth) {
                    apply_exposure_options(sensor, "depth",
                                           config_.depth_enable_auto_exposure,
                                           config_.depth_exposure,
                                           config_.depth_gain);
                    apply_roi(sensor, "depth");
                }
            } catch (const std::exception &error) {
                warn_throttled("realsense-options",
                               "cannot configure RealSense " +
                                   descriptor_.serial + ": " + error.what());
            }
        }
    }

    void configure_filters()
    {
        if (config_.threshold_enabled) {
            threshold_filter_.set_option(
                RS2_OPTION_MIN_DISTANCE,
                static_cast<float>(config_.threshold_min_distance));
            threshold_filter_.set_option(
                RS2_OPTION_MAX_DISTANCE,
                static_cast<float>(config_.threshold_max_distance));
        }
        decimation_filter_.set_option(
            RS2_OPTION_FILTER_MAGNITUDE,
            static_cast<float>(config_.decimation_magnitude));
        spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA,
                                   static_cast<float>(config_.spatial_alpha));
        spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA,
                                   static_cast<float>(config_.spatial_delta));
        spatial_filter_.set_option(
            RS2_OPTION_HOLES_FILL,
            static_cast<float>(config_.spatial_holes_fill));
        temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA,
                                    static_cast<float>(config_.temporal_alpha));
        temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA,
                                    static_cast<float>(config_.temporal_delta));
        temporal_filter_.set_option(
            RS2_OPTION_HOLES_FILL,
            static_cast<float>(config_.temporal_holes_fill));
    }

    void on_frame(rs2::frame frame)
    {
        try {
            if (local_stopped_.load(std::memory_order_relaxed)) {
                return;
            }
            mark_frame();
            if (auto frameset = frame.as<rs2::frameset>()) {
                enqueue_frameset(std::move(frameset));
                return;
            }
            if (auto video = frame.as<rs2::video_frame>()) {
                on_video_frame(std::move(video));
                if (frameset_worker_requested()) {
                    syncer_(frame);
                    drain_synced_framesets();
                }
                return;
            }
            auto motion = frame.as<rs2::motion_frame>();
            if (!motion) {
                return;
            }
            const auto stream = motion.get_profile().stream_type();
            const auto stamp = frame_stamp(motion);
            if (stream == RS2_STREAM_GYRO) {
                const auto value = motion.get_motion_data();
                publish_imu(pub_gyro_, gyro_frame_id(), stamp, true, value.x,
                            value.y, value.z);
                handle_gyro(MotionSample{ stamp, value.x, value.y, value.z });
            } else if (stream == RS2_STREAM_ACCEL) {
                const auto value = motion.get_motion_data();
                publish_imu(pub_accel_, accel_frame_id(), stamp, false,
                            value.x, value.y, value.z);
                handle_accel(MotionSample{ stamp, value.x, value.y, value.z });
            }
        } catch (const std::exception &error) {
            error_throttled("realsense-callback",
                            "RealSense " + descriptor_.serial +
                                " callback failed: " + error.what());
        }
    }

    void handle_gyro(const MotionSample &gyro)
    {
        if (!pub_imu_) {
            return;
        }
        std::lock_guard<std::mutex> lock(imu_mutex_);
        latest_gyro_ = gyro;
        if (config_.imu_sync_method == "COPY") {
            if (latest_accel_) {
                publish_combined_imu(gyro.stamp, gyro.x, gyro.y, gyro.z,
                                     latest_accel_->x, latest_accel_->y,
                                     latest_accel_->z);
            }
            return;
        }
        if (config_.imu_sync_method == "LINEAR_INTERPOLATION") {
            if (try_publish_interpolated_imu(gyro)) {
                return;
            }
            pending_gyros_.push_back(gyro);
            while (pending_gyros_.size() > 128) {
                pending_gyros_.pop_front();
            }
        }
    }

    void handle_accel(const MotionSample &accel)
    {
        if (!pub_imu_) {
            return;
        }
        std::lock_guard<std::mutex> lock(imu_mutex_);
        previous_accel_ = latest_accel_;
        latest_accel_ = accel;
        if (config_.imu_sync_method == "COPY") {
            return;
        }
        for (auto it = pending_gyros_.begin(); it != pending_gyros_.end();) {
            if (try_publish_interpolated_imu(*it)) {
                it = pending_gyros_.erase(it);
            } else if (latest_accel_ &&
                       it->stamp.nanoseconds() <
                           latest_accel_->stamp.nanoseconds() -
                               200000000LL) {
                if (previous_accel_) {
                    publish_combined_imu(it->stamp, it->x, it->y, it->z,
                                         previous_accel_->x,
                                         previous_accel_->y,
                                         previous_accel_->z);
                }
                it = pending_gyros_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool try_publish_interpolated_imu(const MotionSample &gyro)
    {
        if (!previous_accel_ || !latest_accel_) {
            return false;
        }
        const auto t0 = previous_accel_->stamp.nanoseconds();
        const auto t1 = latest_accel_->stamp.nanoseconds();
        const auto tg = gyro.stamp.nanoseconds();
        if (t1 <= t0 || tg < t0 || tg > t1) {
            return false;
        }
        const float ratio = static_cast<float>(
            static_cast<double>(tg - t0) / static_cast<double>(t1 - t0));
        const auto lerp = [ratio](float a, float b) {
            return a + (b - a) * ratio;
        };
        publish_combined_imu(gyro.stamp, gyro.x, gyro.y, gyro.z,
                             lerp(previous_accel_->x, latest_accel_->x),
                             lerp(previous_accel_->y, latest_accel_->y),
                             lerp(previous_accel_->z, latest_accel_->z));
        return true;
    }

    void on_video_frame(const rs2::video_frame &frame)
    {
        const auto stamp = frame_stamp(frame);
        const auto profile = frame.get_profile();
        switch (profile.stream_type()) {
        case RS2_STREAM_COLOR:
            if (color_requested() && !frameset_worker_requested()) {
                enqueue_color(frame, stamp);
            }
            break;
        case RS2_STREAM_DEPTH:
            if (depth_requested() && !frameset_worker_requested()) {
                publish_depth(process_depth_frame(frame.as<rs2::depth_frame>()),
                              stamp, false);
            }
            break;
        case RS2_STREAM_INFRARED:
            if (profile.stream_index() == 1 && infra1_requested()) {
                publish_video(frame, stamp, pub_infra1_, pub_infra1_info_,
                              config_.rectify_infra1, "mono8",
                              infra1_frame_id(), CV_8UC1);
            } else if (profile.stream_index() == 2 && infra2_requested()) {
                publish_video(frame, stamp, pub_infra2_, pub_infra2_info_,
                              config_.rectify_infra2, "mono8",
                              infra2_frame_id(), CV_8UC1);
            }
            break;
        default:
            break;
        }
    }

    void drain_synced_framesets()
    {
        rs2::frameset frameset;
        while (syncer_.poll_for_frames(&frameset)) {
            enqueue_frameset(std::move(frameset));
        }
    }

    void enqueue_frameset(rs2::frameset frameset)
    {
        const auto color = frameset.get_color_frame();
        const auto depth = frameset.get_depth_frame();
        const auto stamp_frame = color ? rs2::frame(color) : rs2::frame(depth);
        const auto stamp = stamp_frame ? frame_stamp(stamp_frame) : node_.now();
        {
            std::lock_guard<std::mutex> lock(video_mutex_);
            latest_frameset_ = PendingFrameset{ std::move(frameset), stamp };
        }
        video_cv_.notify_one();
    }

    void enqueue_color(const rs2::video_frame &frame,
                       const rclcpp::Time &stamp)
    {
        {
            std::lock_guard<std::mutex> lock(color_mutex_);
            if (color_stopping_) {
                return;
            }
            if (color_queue_.size() >= kMaxPendingColorFrames) {
                color_queue_.pop_front();
                ++dropped_color_frames_;
            }
            color_queue_.push_back(PendingColorFrame{ frame, stamp });
        }
        color_cv_.notify_one();
    }

    void run_color_worker()
    {
        while (true) {
            std::optional<PendingColorFrame> pending;
            {
                std::unique_lock<std::mutex> lock(color_mutex_);
                color_cv_.wait(lock, [this] {
                    return color_stopping_ || !color_queue_.empty();
                });
                if (color_stopping_ && color_queue_.empty()) {
                    return;
                }
                pending.emplace(std::move(color_queue_.front()));
                color_queue_.pop_front();
            }
            try {
                publish_video(pending->frame, pending->stamp, pub_color_,
                              pub_color_info_, config_.rectify_color, "bgr8",
                              color_frame_id(), CV_8UC3);
            } catch (const std::exception &error) {
                error_throttled("realsense-color",
                                "RealSense " + descriptor_.serial +
                                    " color frame failed: " + error.what());
            }
        }
    }

    void stop_color_worker() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(color_mutex_);
            color_stopping_ = true;
            color_queue_.clear();
        }
        color_cv_.notify_all();
        if (color_thread_.joinable()) {
            color_thread_.join();
        }
    }

    void run_video_worker()
    {
        while (true) {
            std::optional<PendingFrameset> pending;
            {
                std::unique_lock<std::mutex> lock(video_mutex_);
                video_cv_.wait(lock, [this] {
                    return video_stopping_ || latest_frameset_.has_value();
                });
                if (video_stopping_) {
                    return;
                }
                pending = std::move(latest_frameset_);
                latest_frameset_.reset();
            }
            try {
                publish_frameset(pending->frameset, pending->stamp);
            } catch (const std::exception &error) {
                error_throttled("realsense-frameset",
                                "RealSense " + descriptor_.serial +
                                    " frame failed: " + error.what());
            }
        }
    }

    rs2::frameset process_depth(rs2::frameset frameset)
    {
        rs2::frame processed = frameset;
        if (config_.decimation_enabled) {
            processed = decimation_filter_.process(processed);
        }
        if (config_.threshold_enabled) {
            processed = threshold_filter_.process(processed);
        }
        if (config_.spatial_enabled) {
            processed = spatial_filter_.process(processed);
        }
        if (config_.temporal_enabled) {
            processed = temporal_filter_.process(processed);
        }
        if (config_.hole_filling_enabled) {
            processed = hole_filter_.process(processed);
        }
        if (config_.second_hole_filling_enabled) {
            processed = second_hole_filter_.process(processed);
        }
        auto result = processed.as<rs2::frameset>();
        if (!result) {
            throw std::runtime_error(
                "RealSense depth filters did not return a frameset");
        }
        return result;
    }

    rs2::depth_frame process_depth_frame(rs2::depth_frame frame)
    {
        if (!depth_filters_enabled()) {
            return frame;
        }
        rs2::frame processed = frame;
        if (config_.decimation_enabled) {
            processed = decimation_filter_.process(processed);
        }
        if (config_.threshold_enabled) {
            processed = threshold_filter_.process(processed);
        }
        if (config_.spatial_enabled) {
            processed = spatial_filter_.process(processed);
        }
        if (config_.temporal_enabled) {
            processed = temporal_filter_.process(processed);
        }
        if (config_.hole_filling_enabled) {
            processed = hole_filter_.process(processed);
        }
        if (config_.second_hole_filling_enabled) {
            processed = second_hole_filter_.process(processed);
        }
        auto result = processed.as<rs2::depth_frame>();
        if (!result) {
            throw std::runtime_error(
                "RealSense depth filters did not return a depth frame");
        }
        return result;
    }

    void publish_frameset(const rs2::frameset &frameset,
                          const rclcpp::Time &stamp)
    {
        const bool need_depth = depth_requested();
        const bool need_color = color_requested();
        const bool need_aligned = aligned_depth_requested();
        const bool need_rgbd = rgbd_requested();
        const bool need_pointcloud = pointcloud_requested();
        if (!(need_depth || need_color || need_aligned || need_rgbd ||
              need_pointcloud)) {
            return;
        }
        rs2::frameset processed = frameset;
        if (config_.enable_depth &&
            (need_depth || need_aligned || need_rgbd || need_pointcloud)) {
            processed = process_depth(frameset);
        }
        auto color = frameset.get_color_frame();
        if (need_color && color) {
            publish_video(color, stamp, pub_color_, pub_color_info_,
                          config_.rectify_color, "bgr8", color_frame_id(),
                          CV_8UC3);
        }
        if (need_depth) {
            publish_depth(processed.get_depth_frame(), stamp, false);
        }
        std::optional<rs2::frameset> aligned;
        if (need_aligned || need_rgbd) {
            aligned = align_to_color_.process(processed).as<rs2::frameset>();
        }
        if (need_aligned && aligned) {
            auto depth = aligned->get_depth_frame();
            if (depth && color) {
                const auto intrinsics = color.get_profile()
                                            .as<rs2::video_stream_profile>()
                                            .get_intrinsics();
                publish_calibrated_image(
                    depth.get_data(), depth.get_width(), depth.get_height(),
                    static_cast<std::size_t>(depth.get_stride_in_bytes()),
                    CV_16UC1, "16UC1", color_frame_id(), stamp,
                    pub_aligned_depth_, pub_aligned_depth_info_,
                    calibration_from(intrinsics), config_.rectify_color, true);
            }
        }
        if (need_rgbd && aligned && color) {
#if BXI_DEPTH_CAMERA_HAS_RGBD_MSG
            auto depth = aligned->get_depth_frame();
            if (depth) {
                const auto intrinsics = color.get_profile()
                                            .as<rs2::video_stream_profile>()
                                            .get_intrinsics();
                const auto calibration = calibration_from(intrinsics);
                realsense2_camera_msgs::msg::RGBD message;
                message.header.stamp = stamp;
                message.header.frame_id = color_frame_id();
                message.rgb = make_image(
                    color.get_data(), color.get_width(), color.get_height(),
                    static_cast<std::size_t>(color.get_stride_in_bytes()),
                    CV_8UC3, "bgr8", color_frame_id(), stamp);
                message.depth = make_image(
                    depth.get_data(), depth.get_width(), depth.get_height(),
                    static_cast<std::size_t>(depth.get_stride_in_bytes()),
                    CV_16UC1, "16UC1", color_frame_id(), stamp);
                message.rgb_camera_info = make_camera_info(
                    color.get_width(), color.get_height(), color_frame_id(),
                    stamp, calibration, false);
                message.depth_camera_info = make_camera_info(
                    depth.get_width(), depth.get_height(), color_frame_id(),
                    stamp, calibration, false);
                pub_rgbd_->publish(std::move(message));
            }
#endif
        }
        if (need_pointcloud) {
            enqueue_pointcloud([this, processed, frameset, stamp] {
                create_pointcloud(processed, frameset, stamp);
            });
        }
    }

    void publish_depth(const rs2::depth_frame &frame, const rclcpp::Time &stamp,
                       bool aligned)
    {
        if (!frame) {
            return;
        }
        const auto profile =
            frame.get_profile().as<rs2::video_stream_profile>();
        const auto image_publisher =
            aligned ? pub_aligned_depth_ : pub_depth_;
        const auto info_publisher =
            aligned ? pub_aligned_depth_info_ : pub_depth_info_;
        const bool rectify = aligned ? config_.rectify_color :
                                      config_.rectify_depth;
        Calibration calibration;
        if (rectify ||
            (info_publisher && info_publisher->get_subscription_count() > 0)) {
            calibration = calibration_from(profile.get_intrinsics());
        }
        publish_calibrated_image(
            frame.get_data(), frame.get_width(), frame.get_height(),
            static_cast<std::size_t>(frame.get_stride_in_bytes()), CV_16UC1,
            "16UC1", aligned ? color_frame_id() : depth_frame_id(), stamp,
            image_publisher, info_publisher, calibration, rectify, true);
    }

    void publish_video(const rs2::video_frame &frame, const rclcpp::Time &stamp,
                       const ImagePublisher::SharedPtr &image,
                       const InfoPublisher::SharedPtr &info, bool rectify,
                       const std::string &encoding, const std::string &frame_id,
                       int cv_type)
    {
        if (!frame) {
            return;
        }
        const auto profile =
            frame.get_profile().as<rs2::video_stream_profile>();
        Calibration calibration;
        if (rectify || (info && info->get_subscription_count() > 0)) {
            calibration = calibration_from(profile.get_intrinsics());
        }
        publish_calibrated_image(
            frame.get_data(), frame.get_width(), frame.get_height(),
            static_cast<std::size_t>(frame.get_stride_in_bytes()), cv_type,
            encoding, frame_id, stamp, image, info, calibration, rectify,
            false);
    }

    void create_pointcloud(const rs2::frameset &filtered,
                           const rs2::frameset &original,
                           const rclcpp::Time &stamp)
    {
        auto depth = filtered.get_depth_frame();
        if (!depth) {
            return;
        }
        auto color = original.get_color_frame();
        if (color) {
            pointcloud_.map_to(color);
        }
        auto points_frame = pointcloud_.calculate(depth);
        const auto *vertices = points_frame.get_vertices();
        const auto *texture = color ? points_frame.get_texture_coordinates() :
                                      nullptr;
        const auto count = points_frame.size();
        std::vector<PointXYZRGB> points(count);
        const auto *color_data =
            color ? static_cast<const std::uint8_t *>(color.get_data()) :
                    nullptr;
        const int color_width = color ? color.get_width() : 0;
        const int color_height = color ? color.get_height() : 0;
        const int color_stride = color ? color.get_stride_in_bytes() : 0;
        for (std::size_t index = 0; index < count; ++index) {
            auto &output = points[index];
            output.x = vertices[index].x;
            output.y = vertices[index].y;
            output.z = vertices[index].z;
            if (!texture || !color_data) {
                continue;
            }
            const float u = texture[index].u;
            const float v = texture[index].v;
            output.has_color = true;
            output.texture_valid = std::isfinite(u) && std::isfinite(v) &&
                                   u >= 0.0F && u < 1.0F && v >= 0.0F &&
                                   v < 1.0F;
            if (output.texture_valid) {
                const int x =
                    std::min(color_width - 1,
                             static_cast<int>(std::floor(u * color_width)));
                const int y =
                    std::min(color_height - 1,
                             static_cast<int>(std::floor(v * color_height)));
                const auto *pixel = color_data + y * color_stride + x * 3;
                output.r = pixel[2];
                output.g = pixel[1];
                output.b = pixel[0];
            }
        }
        publish_pointcloud(points,
                           static_cast<std::uint32_t>(depth.get_width()),
                           static_cast<std::uint32_t>(depth.get_height()),
                           stamp);
    }

    rs2::context context_;
    rs2::device device_;
    std::vector<ActiveSensor> active_sensors_;
    rs2::syncer syncer_{ 1 };
    rs2::align align_to_color_;
    rs2::decimation_filter decimation_filter_;
    rs2::threshold_filter threshold_filter_;
    rs2::spatial_filter spatial_filter_;
    rs2::temporal_filter temporal_filter_;
    rs2::hole_filling_filter hole_filter_;
    rs2::hole_filling_filter second_hole_filter_;
    rs2::pointcloud pointcloud_;

    std::mutex time_mutex_;
    bool time_base_initialized_{ false };
    double camera_time_base_ms_{ 0.0 };
    double previous_camera_time_ms_{ 0.0 };
    std::int64_t ros_time_base_ns_{ 0 };

    std::mutex imu_mutex_;
    std::optional<MotionSample> latest_gyro_;
    std::optional<MotionSample> previous_accel_;
    std::optional<MotionSample> latest_accel_;
    std::deque<MotionSample> pending_gyros_;

    static constexpr std::size_t kMaxPendingColorFrames = 8;
    std::mutex color_mutex_;
    std::condition_variable color_cv_;
    std::deque<PendingColorFrame> color_queue_;
    bool color_stopping_{ false };
    std::thread color_thread_{ &RealSenseCamera::run_color_worker, this };
    std::uint64_t dropped_color_frames_{ 0 };

    std::mutex video_mutex_;
    std::condition_variable video_cv_;
    std::optional<PendingFrameset> latest_frameset_;
    bool video_stopping_{ false };
    std::thread video_thread_;
    std::atomic<bool> local_stopped_{ false };
};

} // namespace

std::vector<DeviceDescriptor> discover_realsense()
{
    std::vector<DeviceDescriptor> result;
    auto &discovery = realsense_discovery_context();
    for (auto device : discovery.context.query_devices()) {
        try {
            const std::string serial =
                device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            if (serial.empty()) {
                continue;
            }
            result.push_back(DeviceDescriptor{
                "realsense", serial, device.get_info(RS2_CAMERA_INFO_NAME),
                device.supports(RS2_CAMERA_INFO_PHYSICAL_PORT) ?
                    device.get_info(RS2_CAMERA_INFO_PHYSICAL_PORT) :
                    "" });
        } catch (const std::exception &) {
        }
    }
    return result;
}

std::uint64_t realsense_device_generation()
{
    return realsense_discovery_context().generation.load(
        std::memory_order_relaxed);
}

std::unique_ptr<CameraDevice> make_realsense_camera(
    rclcpp::Node &node, const DeviceDescriptor &descriptor,
    const std::string &logical_name, const CameraConfig &config)
{
    return std::make_unique<RealSenseCamera>(node, descriptor, logical_name,
                                             config);
}

} // namespace bxi_depth_camera
