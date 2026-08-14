#include <libobsensor/ObSensor.hpp>
#include <librealsense2/rs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#ifndef BXI_DEPTH_CAMERA_TOOL_MODE
#define BXI_DEPTH_CAMERA_TOOL_MODE 0
#endif

namespace
{

constexpr int kOrbbecVendorId = 0x2BC5;
constexpr int kGemini335ProductId = 0x0800;
constexpr double kPi = 3.14159265358979323846;
constexpr int kToolMode = BXI_DEPTH_CAMERA_TOOL_MODE;
constexpr bool kValidationProgram = kToolMode != 0;
constexpr bool kFpsTestProgram = kToolMode == 2;
constexpr const char *kProgramName =
    kFpsTestProgram ?
        "cameras-fps-test" :
        (kValidationProgram ? "cameras-validate" : "cameras-inspect");
std::atomic<bool> running{ true };

struct Options {
    bool watch{ false };
    std::string serial;
    double interval{ 1.0 };
    unsigned int frame_timeout_ms{ 3000 };
    double measure_seconds{ 3.0 };
    unsigned int warmup_frames{ 3 };
    double fps_tolerance_percent{ 5.0 };
};

struct Identity {
    std::string backend;
    std::string serial;
    std::string name;
    std::string uid;

    std::string key() const
    {
        return backend + ":" + serial;
    }
};

struct StreamInfo {
    std::string sensor;
    std::string type;
    int width{ 0 };
    int height{ 0 };
    int fps{ 0 };
    std::string format;
    bool is_default{ false };
    int stream_index{ -1 };
    int unique_id{ -1 };
    double fx{ 0.0 };
    double fy{ 0.0 };
    double cx{ 0.0 };
    double cy{ 0.0 };
    bool has_intrinsics{ false };
    std::string distortion_model;
    std::vector<std::pair<std::string, double>> distortion;
    std::string calibration_error;
    bool verification_attempted{ false };
    bool verified{ false };
    int actual_width{ 0 };
    int actual_height{ 0 };
    int actual_fps{ 0 };
    std::string actual_format;
    std::uint64_t frame_number{ 0 };
    double frame_timestamp_ms{ 0.0 };
    std::string timestamp_domain;
    double verification_ms{ 0.0 };
    std::string verification_error;
    std::size_t measured_frames{ 0 };
    double host_measured_fps{ 0.0 };
    double device_measured_fps{ 0.0 };
    double p95_interval_ms{ 0.0 };
    double max_interval_ms{ 0.0 };
    std::uint64_t dropped_frames{ 0 };
};

struct Report {
    Identity identity;
    std::vector<std::pair<std::string, std::string>> device_details;
    std::vector<StreamInfo> supported_profiles;
    std::optional<double> depth_scale_m;
    std::vector<std::string> warnings;
};

using InventoryCallback = std::function<void(const Report &)>;
using ProgressCallback =
    std::function<void(const Report &, std::size_t completed_index)>;

struct PendingProfile {
    StreamInfo stream;
    std::function<void(StreamInfo &, std::optional<double> &)> verify;
};

void signal_handler(int)
{
    running.store(false);
}

double parse_positive_double(const std::string &text, const std::string &name)
{
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(name + " must be a positive number");
    }
    return value;
}

unsigned int parse_positive_unsigned(const std::string &text,
                                     const std::string &name)
{
    std::size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument(name + " must be a positive integer");
    }
    return static_cast<unsigned int>(value);
}

void append_error(std::string &destination, const std::string &message)
{
    if (!destination.empty()) {
        destination += "; ";
    }
    destination += message;
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--watch") {
            options.watch = true;
        } else if (argument == "--serial") {
            if (++index >= argc) {
                throw std::invalid_argument("--serial requires a value");
            }
            options.serial = argv[index];
        } else if (argument == "--interval") {
            if (++index >= argc) {
                throw std::invalid_argument("--interval requires a value");
            }
            options.interval = parse_positive_double(argv[index], "--interval");
        } else if (argument == "--frame-timeout-ms") {
            if (!kValidationProgram) {
                throw std::invalid_argument(
                    "--frame-timeout-ms is only supported by "
                    "cameras-validate");
            }
            if (++index >= argc) {
                throw std::invalid_argument(
                    "--frame-timeout-ms requires a value");
            }
            options.frame_timeout_ms =
                parse_positive_unsigned(argv[index], "--frame-timeout-ms");
        } else if (argument == "--measure-seconds") {
            if (!kFpsTestProgram) {
                throw std::invalid_argument(
                    "--measure-seconds is only supported by "
                    "cameras-fps-test");
            }
            if (++index >= argc) {
                throw std::invalid_argument(
                    "--measure-seconds requires a value");
            }
            options.measure_seconds =
                parse_positive_double(argv[index], "--measure-seconds");
        } else if (argument == "--warmup-frames") {
            if (!kFpsTestProgram) {
                throw std::invalid_argument(
                    "--warmup-frames is only supported by cameras-fps-test");
            }
            if (++index >= argc) {
                throw std::invalid_argument("--warmup-frames requires a value");
            }
            options.warmup_frames =
                parse_positive_unsigned(argv[index], "--warmup-frames");
        } else if (argument == "--fps-tolerance-percent") {
            if (!kFpsTestProgram) {
                throw std::invalid_argument(
                    "--fps-tolerance-percent is only supported by "
                    "cameras-fps-test");
            }
            if (++index >= argc) {
                throw std::invalid_argument(
                    "--fps-tolerance-percent requires a value");
            }
            options.fps_tolerance_percent =
                parse_positive_double(argv[index], "--fps-tolerance-percent");
            if (options.fps_tolerance_percent >= 100.0) {
                throw std::invalid_argument(
                    "--fps-tolerance-percent must be less than 100");
            }
        } else if (argument == "-h" || argument == "--help") {
            std::cout << "Usage: " << kProgramName
                      << " [--watch] [--serial SERIAL] [--interval SECONDS]";
            if (kValidationProgram) {
                std::cout << " [--frame-timeout-ms MS]";
            }
            if (kFpsTestProgram) {
                std::cout << " [--measure-seconds SECONDS]"
                             " [--warmup-frames COUNT]"
                             " [--fps-tolerance-percent PERCENT]";
            }
            std::cout << "\n\n";
            if (kFpsTestProgram) {
                std::cout
                    << "Measures sustained host/device FPS, frame intervals "
                       "and frame-number drops for every COLOR/DEPTH profile.\n"
                       "A profile fails when measured FPS is below the "
                       "configured tolerance or any frame is dropped.\n";
            } else if (kValidationProgram) {
                std::cout
                    << "Enumerates the complete COLOR/DEPTH inventory, then "
                       "opens every profile individually and requires one "
                       "matching frame.\n"
                       "Interactive terminals use a live TUI; redirected "
                       "output uses plain progress lines.\n";
            } else {
                std::cout
                    << "Quickly enumerates and organizes all advertised "
                       "COLOR/DEPTH profiles and calibration records without "
                       "starting any stream.\n";
            }
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return options;
}

std::string hexadecimal_id(int value)
{
    std::ostringstream stream;
    stream << "0x" << std::uppercase << std::hex << std::setw(4)
           << std::setfill('0') << value;
    return stream.str();
}

std::string orbbec_distortion_name(OBCameraDistortionModel model)
{
    switch (model) {
    case OB_DISTORTION_NONE:
        return "NONE";
    case OB_DISTORTION_MODIFIED_BROWN_CONRADY:
        return "MODIFIED_BROWN_CONRADY";
    case OB_DISTORTION_INVERSE_BROWN_CONRADY:
        return "INVERSE_BROWN_CONRADY";
    case OB_DISTORTION_BROWN_CONRADY:
        return "BROWN_CONRADY";
    case OB_DISTORTION_BROWN_CONRADY_K6:
        return "BROWN_CONRADY_K6";
    case OB_DISTORTION_KANNALA_BRANDT4:
        return "KANNALA_BRANDT4";
    }
    return "UNKNOWN(" + std::to_string(static_cast<int>(model)) + ")";
}

bool same_orbbec_profile(const std::shared_ptr<ob::VideoStreamProfile> &left,
                         const std::shared_ptr<ob::VideoStreamProfile> &right)
{
    return left && right && left->getType() == right->getType() &&
           left->getFormat() == right->getFormat() &&
           left->getWidth() == right->getWidth() &&
           left->getHeight() == right->getHeight() &&
           left->getFps() == right->getFps();
}

bool profile_less(const StreamInfo &left, const StreamInfo &right)
{
    if (left.type != right.type) {
        return left.type < right.type;
    }
    if (left.sensor != right.sensor) {
        return left.sensor < right.sensor;
    }
    if (left.width != right.width) {
        return left.width > right.width;
    }
    if (left.height != right.height) {
        return left.height > right.height;
    }
    if (left.fps != right.fps) {
        return left.fps > right.fps;
    }
    if (left.format != right.format) {
        return left.format < right.format;
    }
    return left.stream_index < right.stream_index;
}

void prepare_and_verify_profiles(Report &report,
                                 std::vector<PendingProfile> &pending,
                                 bool validation_enabled,
                                 const InventoryCallback &inventory_callback,
                                 const ProgressCallback &progress_callback)
{
    std::stable_sort(pending.begin(), pending.end(),
                     [](const PendingProfile &left,
                        const PendingProfile &right) {
                         return profile_less(left.stream, right.stream);
                     });
    report.supported_profiles.reserve(pending.size());
    for (auto &item : pending) {
        report.supported_profiles.push_back(std::move(item.stream));
    }

    if (inventory_callback) {
        inventory_callback(report);
    }
    if (!validation_enabled) {
        return;
    }
    for (std::size_t index = 0; index < pending.size() && running.load();
         ++index) {
        pending[index].verify(report.supported_profiles[index],
                              report.depth_scale_m);
        if (progress_callback) {
            progress_callback(report, index);
        }
    }
}

std::optional<std::pair<double, double>> field_of_view(const StreamInfo &profile)
{
    if (!profile.has_intrinsics || !std::isfinite(profile.fx) ||
        !std::isfinite(profile.fy) || profile.fx <= 0.0 || profile.fy <= 0.0 ||
        profile.width <= 0 || profile.height <= 0) {
        return std::nullopt;
    }
    const double left = std::atan(profile.cx / profile.fx);
    const double right = std::atan(
        (static_cast<double>(profile.width) - profile.cx) / profile.fx);
    const double top = std::atan(profile.cy / profile.fy);
    const double bottom = std::atan(
        (static_cast<double>(profile.height) - profile.cy) / profile.fy);
    return std::make_pair((left + right) * 180.0 / kPi,
                          (top + bottom) * 180.0 / kPi);
}

std::vector<Identity> discover_realsense()
{
    std::vector<Identity> result;
    rs2::context context;
    for (auto device : context.query_devices()) {
        try {
            const std::string serial =
                device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            if (serial.empty()) {
                continue;
            }
            result.push_back(Identity{
                "realsense", serial, device.get_info(RS2_CAMERA_INFO_NAME),
                device.supports(RS2_CAMERA_INFO_PHYSICAL_PORT) ?
                    device.get_info(RS2_CAMERA_INFO_PHYSICAL_PORT) :
                    "" });
        } catch (const std::exception &) {
        }
    }
    return result;
}

std::vector<Identity> discover_orbbec()
{
    std::vector<Identity> result;
    ob::Context context;
    auto devices = context.queryDeviceList();
    for (std::uint32_t index = 0; index < devices->getCount(); ++index) {
        try {
            auto device = devices->getDevice(index);
            auto info = device->getDeviceInfo();
            if (info->getVid() != kOrbbecVendorId ||
                info->getPid() != kGemini335ProductId) {
                continue;
            }
            const std::string serial = info->getSerialNumber();
            if (serial.empty()) {
                continue;
            }
            result.push_back(
                Identity{ "orbbec", serial, info->getName(), info->getUid() });
        } catch (const std::exception &) {
        }
    }
    return result;
}

std::map<std::string, Identity> scan(const std::string &serial)
{
    std::map<std::string, Identity> result;
    try {
        for (auto &identity : discover_realsense()) {
            if (serial.empty() || identity.serial == serial) {
                result.emplace(identity.key(), std::move(identity));
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "RealSense enumeration failed: " << error.what() << '\n';
    }
    try {
        for (auto &identity : discover_orbbec()) {
            if (serial.empty() || identity.serial == serial) {
                result.emplace(identity.key(), std::move(identity));
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "Orbbec enumeration failed: " << error.what() << '\n';
    }
    return result;
}

void verify_realsense_profile(const rs2::sensor &sensor,
                              const rs2::stream_profile &profile,
                              unsigned int timeout_ms, StreamInfo &stream)
{
    const auto started_at = std::chrono::steady_clock::now();
    bool opened = false;
    bool streaming = false;
    stream.verification_attempted = true;

    try {
        rs2::frame_queue queue(1);
        sensor.open(profile);
        opened = true;
        sensor.start(queue);
        streaming = true;

        const auto frame = queue.wait_for_frame(timeout_ms);
        const auto video = frame.as<rs2::video_frame>();
        if (!video) {
            throw std::runtime_error("received frame is not a video frame");
        }
        const auto actual_profile = frame.get_profile();
        stream.actual_width = video.get_width();
        stream.actual_height = video.get_height();
        stream.actual_fps = actual_profile.fps();
        stream.actual_format = rs2_format_to_string(actual_profile.format());
        stream.frame_number = frame.get_frame_number();
        stream.frame_timestamp_ms = frame.get_timestamp();
        stream.timestamp_domain =
            rs2_timestamp_domain_to_string(frame.get_frame_timestamp_domain());

        if (actual_profile.stream_type() != profile.stream_type() ||
            stream.actual_width != stream.width ||
            stream.actual_height != stream.height ||
            stream.actual_fps != stream.fps ||
            actual_profile.format() != profile.format()) {
            throw std::runtime_error(
                "received frame does not match requested profile");
        }
        stream.verified = true;
    } catch (const std::exception &error) {
        append_error(stream.verification_error, error.what());
    }

    if (streaming) {
        try {
            sensor.stop();
        } catch (const std::exception &error) {
            stream.verified = false;
            append_error(stream.verification_error,
                         std::string("stop failed: ") + error.what());
        }
    }
    if (opened) {
        try {
            sensor.close();
        } catch (const std::exception &error) {
            stream.verified = false;
            append_error(stream.verification_error,
                         std::string("close failed: ") + error.what());
        }
    }
    stream.verification_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started_at)
                                 .count();
}

void verify_orbbec_profile(
    const std::shared_ptr<ob::Device> &device,
    const std::shared_ptr<ob::VideoStreamProfile> &profile,
    unsigned int timeout_ms, StreamInfo &stream,
    std::optional<double> &depth_scale_m)
{
    const auto started_at = std::chrono::steady_clock::now();
    bool streaming = false;
    stream.verification_attempted = true;
    std::shared_ptr<ob::Pipeline> pipeline;

    try {
        pipeline = std::make_shared<ob::Pipeline>(device);
        auto config = std::make_shared<ob::Config>();
        config->enableStream(profile);
        pipeline->start(config);
        streaming = true;

        const auto frameset = pipeline->waitForFrames(timeout_ms);
        if (!frameset) {
            throw std::runtime_error("no frame arrived within " +
                                     std::to_string(timeout_ms) + " ms");
        }

        std::shared_ptr<ob::VideoFrame> frame;
        if (profile->getType() == OB_STREAM_COLOR) {
            frame = frameset->getColorFrame();
        } else if (profile->getType() == OB_STREAM_DEPTH) {
            const auto depth = frameset->getDepthFrame();
            frame = depth;
            if (depth) {
                const double scale_m = depth->getValueScale() * 0.001;
                if (std::isfinite(scale_m) && scale_m > 0.0) {
                    depth_scale_m = scale_m;
                }
            }
        }
        if (!frame) {
            throw std::runtime_error(
                "frameset does not contain the requested stream");
        }

        const auto actual_profile =
            frame->getStreamProfile()->as<ob::VideoStreamProfile>();
        stream.actual_width = static_cast<int>(frame->getWidth());
        stream.actual_height = static_cast<int>(frame->getHeight());
        stream.actual_fps = static_cast<int>(actual_profile->getFps());
        stream.actual_format =
            ob::TypeHelper::convertOBFormatTypeToString(frame->getFormat());
        stream.frame_number = frame->getIndex();
        stream.frame_timestamp_ms =
            static_cast<double>(frame->getTimeStampUs()) / 1000.0;
        stream.timestamp_domain = "device hardware clock";

        if (actual_profile->getType() != profile->getType() ||
            stream.actual_width != stream.width ||
            stream.actual_height != stream.height ||
            stream.actual_fps != stream.fps ||
            frame->getFormat() != profile->getFormat()) {
            throw std::runtime_error(
                "received frame does not match requested profile");
        }
        stream.verified = true;
    } catch (const std::exception &error) {
        append_error(stream.verification_error, error.what());
    }

    if (pipeline && streaming) {
        try {
            pipeline->stop();
        } catch (const std::exception &error) {
            stream.verified = false;
            append_error(stream.verification_error,
                         std::string("stop failed: ") + error.what());
        }
    }
    stream.verification_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started_at)
                                 .count();
}

struct FpsSamples {
    std::vector<std::chrono::steady_clock::time_point> host_times;
    std::vector<double> device_times_ms;
    std::vector<std::uint64_t> frame_numbers;

    void add(std::uint64_t frame_number, double device_time_ms)
    {
        host_times.push_back(std::chrono::steady_clock::now());
        device_times_ms.push_back(device_time_ms);
        frame_numbers.push_back(frame_number);
    }

    double elapsed_seconds() const
    {
        if (host_times.size() < 2) {
            return 0.0;
        }
        return std::chrono::duration<double>(host_times.back() -
                                             host_times.front())
            .count();
    }
};

void finish_fps_measurement(FpsSamples &samples, double tolerance_percent,
                            StreamInfo &stream)
{
    if (samples.host_times.size() < 2) {
        throw std::runtime_error("fewer than two measurement frames arrived");
    }

    stream.measured_frames = samples.host_times.size();
    const double host_seconds = samples.elapsed_seconds();
    stream.host_measured_fps =
        static_cast<double>(samples.host_times.size() - 1) / host_seconds;

    const double device_duration_ms =
        samples.device_times_ms.back() - samples.device_times_ms.front();
    if (std::isfinite(device_duration_ms) && device_duration_ms > 0.0) {
        stream.device_measured_fps =
            static_cast<double>(samples.device_times_ms.size() - 1) * 1000.0 /
            device_duration_ms;
    } else {
        append_error(stream.verification_error,
                     "device timestamps are not monotonic");
    }

    std::vector<double> intervals_ms;
    intervals_ms.reserve(samples.host_times.size() - 1);
    for (std::size_t index = 1; index < samples.host_times.size(); ++index) {
        const double interval =
            std::chrono::duration<double, std::milli>(
                samples.host_times[index] - samples.host_times[index - 1])
                .count();
        intervals_ms.push_back(interval);
        stream.max_interval_ms = std::max(stream.max_interval_ms, interval);

        const auto previous = samples.frame_numbers[index - 1];
        const auto current = samples.frame_numbers[index];
        if (current > previous + 1) {
            stream.dropped_frames += current - previous - 1;
        }
    }
    std::sort(intervals_ms.begin(), intervals_ms.end());
    const std::size_t p95_index =
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(intervals_ms.size()))) -
        1;
    stream.p95_interval_ms = intervals_ms[p95_index];

    const double minimum_fps =
        static_cast<double>(stream.fps) * (1.0 - tolerance_percent / 100.0);
    if (stream.host_measured_fps < minimum_fps) {
        append_error(stream.verification_error,
                     "host FPS " + std::to_string(stream.host_measured_fps) +
                         " is below minimum " + std::to_string(minimum_fps));
    }
    if (stream.device_measured_fps > 0.0 &&
        stream.device_measured_fps < minimum_fps) {
        append_error(stream.verification_error,
                     "device FPS " +
                         std::to_string(stream.device_measured_fps) +
                         " is below minimum " + std::to_string(minimum_fps));
    }
    if (stream.dropped_frames > 0) {
        append_error(stream.verification_error,
                     "detected " + std::to_string(stream.dropped_frames) +
                         " dropped frame(s)");
    }
    stream.verified = stream.verification_error.empty();
}

void measure_realsense_fps(const rs2::sensor &sensor,
                           const rs2::stream_profile &profile,
                           const Options &options, StreamInfo &stream)
{
    const auto started_at = std::chrono::steady_clock::now();
    bool opened = false;
    bool streaming = false;
    stream.verification_attempted = true;
    try {
        rs2::frame_queue queue(1);
        sensor.open(profile);
        opened = true;
        sensor.start(queue);
        streaming = true;
        for (unsigned int index = 0; index < options.warmup_frames; ++index) {
            queue.wait_for_frame(options.frame_timeout_ms);
        }

        FpsSamples samples;
        do {
            const auto frame = queue.wait_for_frame(options.frame_timeout_ms);
            const auto video = frame.as<rs2::video_frame>();
            if (!video) {
                throw std::runtime_error("received frame is not a video frame");
            }
            const auto actual_profile = frame.get_profile();
            stream.actual_width = video.get_width();
            stream.actual_height = video.get_height();
            stream.actual_fps = actual_profile.fps();
            stream.actual_format =
                rs2_format_to_string(actual_profile.format());
            stream.frame_number = frame.get_frame_number();
            stream.frame_timestamp_ms = frame.get_timestamp();
            stream.timestamp_domain = rs2_timestamp_domain_to_string(
                frame.get_frame_timestamp_domain());
            if (actual_profile.stream_type() != profile.stream_type() ||
                stream.actual_width != stream.width ||
                stream.actual_height != stream.height ||
                stream.actual_fps != stream.fps ||
                actual_profile.format() != profile.format()) {
                throw std::runtime_error(
                    "received frame does not match requested profile");
            }
            samples.add(stream.frame_number, stream.frame_timestamp_ms);
        } while (samples.elapsed_seconds() < options.measure_seconds);
        finish_fps_measurement(samples, options.fps_tolerance_percent, stream);
    } catch (const std::exception &error) {
        append_error(stream.verification_error, error.what());
    }

    if (streaming) {
        try {
            sensor.stop();
        } catch (const std::exception &error) {
            stream.verified = false;
            append_error(stream.verification_error,
                         std::string("stop failed: ") + error.what());
        }
    }
    if (opened) {
        try {
            sensor.close();
        } catch (const std::exception &error) {
            stream.verified = false;
            append_error(stream.verification_error,
                         std::string("close failed: ") + error.what());
        }
    }
    stream.verification_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started_at)
                                 .count();
}

void measure_orbbec_fps(const std::shared_ptr<ob::Device> &device,
                        const std::shared_ptr<ob::VideoStreamProfile> &profile,
                        const Options &options, StreamInfo &stream,
                        std::optional<double> &depth_scale_m)
{
    const auto started_at = std::chrono::steady_clock::now();
    bool streaming = false;
    stream.verification_attempted = true;
    std::shared_ptr<ob::Pipeline> pipeline;
    try {
        pipeline = std::make_shared<ob::Pipeline>(device);
        auto config = std::make_shared<ob::Config>();
        config->enableStream(profile);
        pipeline->start(config);
        streaming = true;

        const auto read_frame = [&]() -> std::shared_ptr<ob::VideoFrame> {
            const auto frameset =
                pipeline->waitForFrames(options.frame_timeout_ms);
            if (!frameset) {
                throw std::runtime_error(
                    "no frame arrived within " +
                    std::to_string(options.frame_timeout_ms) + " ms");
            }
            if (profile->getType() == OB_STREAM_COLOR) {
                return frameset->getColorFrame();
            }
            if (profile->getType() == OB_STREAM_DEPTH) {
                const auto depth = frameset->getDepthFrame();
                if (depth) {
                    const double scale_m = depth->getValueScale() * 0.001;
                    if (std::isfinite(scale_m) && scale_m > 0.0) {
                        depth_scale_m = scale_m;
                    }
                }
                return depth;
            }
            return nullptr;
        };

        for (unsigned int index = 0; index < options.warmup_frames; ++index) {
            if (!read_frame()) {
                throw std::runtime_error(
                    "frameset does not contain the requested stream");
            }
        }

        FpsSamples samples;
        do {
            const auto frame = read_frame();
            if (!frame) {
                throw std::runtime_error(
                    "frameset does not contain the requested stream");
            }
            const auto actual_profile =
                frame->getStreamProfile()->as<ob::VideoStreamProfile>();
            stream.actual_width = static_cast<int>(frame->getWidth());
            stream.actual_height = static_cast<int>(frame->getHeight());
            stream.actual_fps = static_cast<int>(actual_profile->getFps());
            stream.actual_format =
                ob::TypeHelper::convertOBFormatTypeToString(frame->getFormat());
            stream.frame_number = frame->getIndex();
            stream.frame_timestamp_ms =
                static_cast<double>(frame->getTimeStampUs()) / 1000.0;
            stream.timestamp_domain = "device hardware clock";
            if (actual_profile->getType() != profile->getType() ||
                stream.actual_width != stream.width ||
                stream.actual_height != stream.height ||
                stream.actual_fps != stream.fps ||
                frame->getFormat() != profile->getFormat()) {
                throw std::runtime_error(
                    "received frame does not match requested profile");
            }
            samples.add(stream.frame_number, stream.frame_timestamp_ms);
        } while (samples.elapsed_seconds() < options.measure_seconds);
        finish_fps_measurement(samples, options.fps_tolerance_percent, stream);
    } catch (const std::exception &error) {
        append_error(stream.verification_error, error.what());
    }

    if (pipeline && streaming) {
        try {
            pipeline->stop();
        } catch (const std::exception &error) {
            stream.verified = false;
            append_error(stream.verification_error,
                         std::string("stop failed: ") + error.what());
        }
    }
    stream.verification_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started_at)
                                 .count();
}

Report inspect_realsense(const Identity &identity, const Options &options,
                         bool validation_enabled,
                         const InventoryCallback &inventory_callback,
                         const ProgressCallback &progress_callback)
{
    Report report;
    report.identity = identity;
    std::vector<PendingProfile> pending;
    rs2::context context;
    rs2::device selected;
    for (auto device : context.query_devices()) {
        try {
            if (identity.serial ==
                device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
                selected = device;
                break;
            }
        } catch (const std::exception &) {
        }
    }
    if (!selected) {
        report.warnings.push_back("device disappeared before inspection");
        if (inventory_callback) {
            inventory_callback(report);
        }
        return report;
    }

    const auto add_device_info = [&](const std::string &label,
                                     rs2_camera_info key) {
        try {
            if (selected.supports(key)) {
                report.device_details.emplace_back(label,
                                                   selected.get_info(key));
            }
        } catch (const std::exception &) {
        }
    };
    add_device_info("firmware", RS2_CAMERA_INFO_FIRMWARE_VERSION);
    add_device_info("recommended firmware",
                    RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION);
    add_device_info("product id", RS2_CAMERA_INFO_PRODUCT_ID);
    add_device_info("product line", RS2_CAMERA_INFO_PRODUCT_LINE);
    add_device_info("connection", RS2_CAMERA_INFO_CONNECTION_TYPE);
    add_device_info("USB type", RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);
    add_device_info("physical port", RS2_CAMERA_INFO_PHYSICAL_PORT);

    for (auto sensor : selected.query_sensors()) {
        std::string sensor_name = "unknown sensor";
        try {
            if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
                sensor_name = sensor.get_info(RS2_CAMERA_INFO_NAME);
            }
        } catch (const std::exception &) {
        }

        try {
            if (sensor.supports(RS2_OPTION_DEPTH_UNITS)) {
                const double scale = sensor.get_option(RS2_OPTION_DEPTH_UNITS);
                if (std::isfinite(scale) && scale > 0.0) {
                    report.depth_scale_m = scale;
                }
            }
        } catch (const std::exception &error) {
            report.warnings.push_back(
                sensor_name + " depth units unavailable: " + error.what());
        }

        try {
            const auto profiles = sensor.get_stream_profiles();
            for (const auto &profile : profiles) {
                if (profile.stream_type() != RS2_STREAM_COLOR &&
                    profile.stream_type() != RS2_STREAM_DEPTH) {
                    continue;
                }
                const auto video = profile.as<rs2::video_stream_profile>();
                if (!video) {
                    continue;
                }
                StreamInfo stream;
                stream.sensor = sensor_name;
                stream.type = rs2_stream_to_string(profile.stream_type());
                stream.width = video.width();
                stream.height = video.height();
                stream.fps = profile.fps();
                stream.format = rs2_format_to_string(profile.format());
                stream.is_default = profile.is_default();
                stream.stream_index = profile.stream_index();
                stream.unique_id = profile.unique_id();
                try {
                    const auto intrinsic = video.get_intrinsics();
                    stream.fx = intrinsic.fx;
                    stream.fy = intrinsic.fy;
                    stream.cx = intrinsic.ppx;
                    stream.cy = intrinsic.ppy;
                    stream.has_intrinsics = true;
                    stream.distortion_model =
                        rs2_distortion_to_string(intrinsic.model);
                    for (int index = 0; index < 5; ++index) {
                        stream.distortion.emplace_back(
                            "c" + std::to_string(index),
                            intrinsic.coeffs[index]);
                    }
                } catch (const std::exception &error) {
                    stream.calibration_error = error.what();
                }
                PendingProfile item;
                item.stream = std::move(stream);
                item.verify = [sensor, profile, options](
                                  StreamInfo &target, std::optional<double> &) {
                    if (kFpsTestProgram) {
                        measure_realsense_fps(sensor, profile, options, target);
                    } else {
                        verify_realsense_profile(
                            sensor, profile, options.frame_timeout_ms, target);
                    }
                };
                pending.push_back(std::move(item));
            }
        } catch (const std::exception &error) {
            report.warnings.push_back(
                sensor_name + " profile enumeration failed: " + error.what());
        }
    }
    prepare_and_verify_profiles(report, pending, validation_enabled,
                                inventory_callback, progress_callback);
    return report;
}

Report inspect_orbbec(const Identity &identity, const Options &options,
                      bool validation_enabled,
                      const InventoryCallback &inventory_callback,
                      const ProgressCallback &progress_callback)
{
    Report report;
    report.identity = identity;
    std::vector<PendingProfile> pending;
    try {
        auto context = std::make_shared<ob::Context>();
        auto devices = context->queryDeviceList();
        auto device = devices->getDeviceBySN(identity.serial.c_str());
        const auto device_info = device->getDeviceInfo();
        report.device_details.emplace_back("firmware",
                                           device_info->getFirmwareVersion());
        report.device_details.emplace_back("hardware",
                                           device_info->getHardwareVersion());
        report.device_details.emplace_back("connection",
                                           device_info->getConnectionType());
        report.device_details.emplace_back(
            "VID:PID", hexadecimal_id(device_info->getVid()) + ":" +
                           hexadecimal_id(device_info->getPid()));

        auto pipeline = std::make_shared<ob::Pipeline>(device);

        const auto enumerate_profiles = [&](OBSensorType sensor_type,
                                            const std::string &type) {
            try {
                const auto list = pipeline->getStreamProfileList(sensor_type);
                std::shared_ptr<ob::VideoStreamProfile> default_profile;
                try {
                    default_profile = list->getVideoStreamProfile();
                } catch (const std::exception &error) {
                    report.warnings.push_back(
                        type + " default profile unavailable: " + error.what());
                }
                for (std::uint32_t index = 0; index < list->getCount();
                     ++index) {
                    try {
                        const auto profile = list->getProfile(index)
                                                 ->as<ob::VideoStreamProfile>();
                        StreamInfo stream;
                        stream.sensor =
                            ob::TypeHelper::convertOBSensorTypeToString(
                                sensor_type);
                        stream.type = type;
                        stream.width = static_cast<int>(profile->getWidth());
                        stream.height = static_cast<int>(profile->getHeight());
                        stream.fps = static_cast<int>(profile->getFps());
                        stream.format =
                            ob::TypeHelper::convertOBFormatTypeToString(
                                profile->getFormat());
                        stream.is_default =
                            same_orbbec_profile(profile, default_profile);
                        try {
                            const auto intrinsic = profile->getIntrinsic();
                            stream.fx = intrinsic.fx;
                            stream.fy = intrinsic.fy;
                            stream.cx = intrinsic.cx;
                            stream.cy = intrinsic.cy;
                            stream.has_intrinsics = true;
                            const auto distortion = profile->getDistortion();
                            stream.distortion_model =
                                orbbec_distortion_name(distortion.model);
                            stream.distortion = {
                                { "k1", distortion.k1 },
                                { "k2", distortion.k2 },
                                { "p1", distortion.p1 },
                                { "p2", distortion.p2 },
                                { "k3", distortion.k3 },
                                { "k4", distortion.k4 },
                                { "k5", distortion.k5 },
                                { "k6", distortion.k6 },
                            };
                        } catch (const std::exception &error) {
                            stream.calibration_error = error.what();
                        }
                        PendingProfile item;
                        item.stream = std::move(stream);
                        item.verify = [device, profile, options](
                                          StreamInfo &target,
                                          std::optional<double> &depth_scale) {
                            if (kFpsTestProgram) {
                                measure_orbbec_fps(device, profile, options,
                                                   target, depth_scale);
                            } else {
                                verify_orbbec_profile(device, profile,
                                                      options.frame_timeout_ms,
                                                      target, depth_scale);
                            }
                        };
                        pending.push_back(std::move(item));
                    } catch (const std::exception &error) {
                        report.warnings.push_back(
                            type + " profile #" + std::to_string(index) +
                            " unavailable: " + error.what());
                    }
                }
            } catch (const std::exception &error) {
                report.warnings.push_back(
                    type + " profile enumeration failed: " + error.what());
            }
        };

        enumerate_profiles(OB_SENSOR_COLOR, "COLOR");
        enumerate_profiles(OB_SENSOR_DEPTH, "DEPTH");
    } catch (const std::exception &error) {
        report.warnings.push_back(std::string("cannot open/read device: ") +
                                  error.what());
    }
    prepare_and_verify_profiles(report, pending, validation_enabled,
                                inventory_callback, progress_callback);
    return report;
}

Report inspect(const Identity &identity, const Options &options,
               bool validation_enabled,
               const InventoryCallback &inventory_callback,
               const ProgressCallback &progress_callback)
{
    return identity.backend == "realsense" ?
               inspect_realsense(identity, options, validation_enabled,
                                 inventory_callback, progress_callback) :
               inspect_orbbec(identity, options, validation_enabled,
                              inventory_callback, progress_callback);
}

std::string profile_description(const StreamInfo &stream)
{
    std::ostringstream output;
    output << stream.type << " " << stream.width << "x" << stream.height
           << " @ " << stream.fps << " FPS " << stream.format;
    if (stream.is_default) {
        output << " [default]";
    }
    return output.str();
}

std::string calibration_key(const StreamInfo &stream)
{
    if (!stream.has_intrinsics && stream.calibration_error.empty()) {
        return {};
    }
    std::ostringstream key;
    key << std::setprecision(17) << stream.type << '|' << stream.width << '|'
        << stream.height << '|' << stream.has_intrinsics << '|' << stream.fx
        << '|' << stream.fy << '|' << stream.cx << '|' << stream.cy << '|'
        << stream.distortion_model << '|' << stream.calibration_error;
    for (const auto &coefficient : stream.distortion) {
        key << '|' << coefficient.first << '=' << coefficient.second;
    }
    return key.str();
}

std::string numbered_label(char prefix, std::size_t number)
{
    std::ostringstream output;
    output << prefix << std::setfill('0') << std::setw(3) << number;
    return output.str();
}

void print_inventory(const Report &report, bool validation_follows)
{
    std::cout << std::defaultfloat << std::setprecision(9)
              << "\n=== Camera profile inventory (enumeration complete) ===\n"
              << "[" << report.identity.backend << "] " << report.identity.name
              << '\n'
              << "  serial : " << report.identity.serial << '\n';
    if (!report.identity.uid.empty()) {
        std::cout << "  uid    : " << report.identity.uid << '\n';
    }
    for (const auto &detail : report.device_details) {
        std::cout << "  " << detail.first << ": " << detail.second << '\n';
    }

    std::cout << "  supported profiles: " << report.supported_profiles.size()
              << " (validation pending)\n";
    if (report.supported_profiles.empty()) {
        std::cout << "    unavailable (device may be in use)\n";
    }

    std::map<std::string, std::size_t> calibration_ids;
    std::vector<const StreamInfo *> calibrations;
    std::vector<std::string> profile_calibrations;
    profile_calibrations.reserve(report.supported_profiles.size());
    for (const auto &stream : report.supported_profiles) {
        const auto key = calibration_key(stream);
        if (key.empty()) {
            profile_calibrations.emplace_back("-");
            continue;
        }
        auto [entry, inserted] =
            calibration_ids.emplace(key, calibration_ids.size() + 1);
        if (inserted) {
            calibrations.push_back(&stream);
        }
        profile_calibrations.push_back(numbered_label('K', entry->second));
    }

    std::string previous_type;
    for (std::size_t index = 0; index < report.supported_profiles.size();
         ++index) {
        const auto &stream = report.supported_profiles[index];
        if (stream.type != previous_type) {
            previous_type = stream.type;
            std::cout << "\n  " << stream.type << " profiles:\n"
                      << "    ID   D resolution    FPS format  calibration "
                         "sensor\n";
        }
        std::ostringstream resolution;
        resolution << stream.width << 'x' << stream.height;
        std::cout << "    " << numbered_label('P', index + 1) << ' '
                  << (stream.is_default ? '*' : ' ') << ' ' << std::left
                  << std::setw(13) << resolution.str() << std::right
                  << std::setw(3) << stream.fps << ' ' << std::left
                  << std::setw(7) << stream.format << std::setw(12)
                  << profile_calibrations[index] << stream.sensor << std::right;
        if (stream.stream_index >= 0) {
            std::cout << " index=" << stream.stream_index;
        }
        if (stream.unique_id >= 0) {
            std::cout << " uid=" << stream.unique_id;
        }
        std::cout << '\n';
    }

    if (!calibrations.empty()) {
        std::cout << "\n  Calibration records (referenced by profile table):\n";
    }
    for (std::size_t index = 0; index < calibrations.size(); ++index) {
        const auto &stream = *calibrations[index];
        std::cout << "    " << numbered_label('K', index + 1) << ' '
                  << stream.type << ' ' << stream.width << 'x' << stream.height;
        if (stream.has_intrinsics) {
            std::cout << " intrinsics fx=" << stream.fx << " fy=" << stream.fy
                      << " cx=" << stream.cx << " cy=" << stream.cy;
            const auto fov = field_of_view(stream);
            if (fov) {
                std::cout << " FOV=" << std::fixed << std::setprecision(2)
                          << fov->first << "x" << fov->second << " deg"
                          << std::defaultfloat << std::setprecision(9);
            }
            std::cout << '\n';
        } else if (!stream.calibration_error.empty()) {
            std::cout
                << " calibration unavailable: " << stream.calibration_error
                << '\n';
        }
        if (!stream.distortion_model.empty()) {
            std::cout << "         distortion model="
                      << stream.distortion_model;
            for (const auto &coefficient : stream.distortion) {
                std::cout << " " << coefficient.first << "="
                          << coefficient.second;
            }
            std::cout << '\n';
        }
    }
    for (const auto &warning : report.warnings) {
        std::cout << "  warning: " << warning << '\n';
    }
    if (validation_follows) {
        std::cout << "\nEnumeration is complete. Starting "
                  << (kFpsTestProgram ? "sustained FPS measurement" :
                                        "one-frame validation")
                  << "...\n";
    } else {
        std::cout << "\nEnumeration is complete. No stream was started.\n";
    }
    std::cout << std::flush;
}

struct VerificationCounts {
    std::size_t attempted{ 0 };
    std::size_t passed{ 0 };
    std::size_t failed{ 0 };
};

VerificationCounts verification_counts(const Report &report)
{
    VerificationCounts result;
    for (const auto &stream : report.supported_profiles) {
        if (!stream.verification_attempted) {
            continue;
        }
        ++result.attempted;
        if (stream.verified) {
            ++result.passed;
        } else {
            ++result.failed;
        }
    }
    return result;
}

void print_validation_summary(const Report &report)
{
    const auto counts = verification_counts(report);
    const auto pending = report.supported_profiles.size() - counts.attempted;
    const bool passed = !report.supported_profiles.empty() &&
                        counts.passed == report.supported_profiles.size();
    std::cout << std::defaultfloat << std::setprecision(9) << "\n=== "
              << (kFpsTestProgram ? "Sustained FPS test result" :
                                    "One-frame validation result")
              << " ===\n"
              << "  camera : [" << report.identity.backend << "] "
              << report.identity.name << " serial=" << report.identity.serial
              << '\n'
              << "  total  : " << report.supported_profiles.size() << '\n'
              << "  PASS   : " << counts.passed << '\n'
              << "  FAIL   : " << counts.failed << '\n'
              << "  pending: " << pending << '\n';
    if (report.depth_scale_m) {
        std::cout << "  depth scale: raw x " << *report.depth_scale_m << " m ("
                  << *report.depth_scale_m * 1000.0 << " mm/unit)\n";
    }
    if (kFpsTestProgram && counts.attempted > 0) {
        std::cout << "\n  Measured profile rates:\n"
                  << "    ID   state nominal host_FPS device_FPS frames "
                     "drops p95_ms max_ms\n";
        for (std::size_t index = 0; index < report.supported_profiles.size();
             ++index) {
            const auto &stream = report.supported_profiles[index];
            if (!stream.verification_attempted) {
                continue;
            }
            std::cout << "    " << numbered_label('P', index + 1) << ' '
                      << (stream.verified ? "PASS " : "FAIL ") << std::fixed
                      << std::setprecision(2) << std::setw(7) << stream.fps
                      << std::setw(9) << stream.host_measured_fps
                      << std::setw(11) << stream.device_measured_fps
                      << std::setw(7) << stream.measured_frames << std::setw(6)
                      << stream.dropped_frames << std::setw(8)
                      << stream.p95_interval_ms << std::setw(7)
                      << stream.max_interval_ms << std::defaultfloat
                      << std::setprecision(9) << ' '
                      << profile_description(stream) << '\n';
            if (!stream.verification_error.empty()) {
                std::cout << "         error=\"" << stream.verification_error
                          << "\"\n";
            }
        }
    } else if (counts.failed > 0) {
        std::cout << "\n  Failed profiles:\n";
        for (std::size_t index = 0; index < report.supported_profiles.size();
             ++index) {
            const auto &stream = report.supported_profiles[index];
            if (!stream.verification_attempted || stream.verified) {
                continue;
            }
            std::cout << "    " << numbered_label('P', index + 1) << ' '
                      << profile_description(stream)
                      << " elapsed=" << std::fixed << std::setprecision(1)
                      << stream.verification_ms << " ms" << std::defaultfloat
                      << std::setprecision(9) << " error=\""
                      << stream.verification_error << "\"\n";
        }
    }
    std::cout << "\n  overall verification: " << (passed ? "PASS" : "FAIL")
              << '\n'
              << std::flush;
}

class ValidationDisplay {
public:
    explicit ValidationDisplay(Options options)
        : options_(std::move(options))
    {
        const char *term = std::getenv("TERM");
        interactive_ = ::isatty(STDOUT_FILENO) != 0 && term != nullptr &&
                       std::string(term) != "dumb";
    }

    ~ValidationDisplay()
    {
        leave_tui();
    }

    void begin(const Report &report)
    {
        print_inventory(report, true);
        if (kFpsTestProgram) {
            std::cout << "FPS test settings: warmup=" << options_.warmup_frames
                      << " frames, measure=" << options_.measure_seconds
                      << " s, tolerance=" << options_.fps_tolerance_percent
                      << "%, frame timeout=" << options_.frame_timeout_ms
                      << " ms\n"
                      << std::flush;
        }
        started_at_ = std::chrono::steady_clock::now();
        if (interactive_ && !report.supported_profiles.empty()) {
            std::cout << "\033[?1049h\033[?25l" << std::flush;
            active_ = true;
            render(report, std::nullopt);
        }
    }

    void update(const Report &report, std::size_t completed_index)
    {
        if (interactive_) {
            render(report, completed_index);
            return;
        }
        const auto &stream = report.supported_profiles[completed_index];
        std::cout << '[' << completed_index + 1 << '/'
                  << report.supported_profiles.size() << "] "
                  << (stream.verified ? "PASS " : "FAIL ")
                  << profile_description(stream) << " elapsed=" << std::fixed
                  << std::setprecision(1) << stream.verification_ms << " ms"
                  << std::defaultfloat << std::setprecision(9);
        if (!stream.verification_error.empty()) {
            std::cout << " error=\"" << stream.verification_error << '"';
        }
        if (kFpsTestProgram) {
            std::cout << std::fixed << std::setprecision(2)
                      << " frames=" << stream.measured_frames
                      << " host_fps=" << stream.host_measured_fps
                      << " device_fps=" << stream.device_measured_fps
                      << " p95=" << stream.p95_interval_ms << "ms"
                      << " max=" << stream.max_interval_ms << "ms"
                      << " drops=" << stream.dropped_frames << std::defaultfloat
                      << std::setprecision(9);
        }
        std::cout << '\n' << std::flush;
    }

    void finish(const Report &report)
    {
        leave_tui();
        print_validation_summary(report);
    }

private:
    void leave_tui()
    {
        if (!active_) {
            return;
        }
        std::cout << "\033[?25h\033[?1049l" << std::flush;
        active_ = false;
    }

    void render(const Report &report,
                const std::optional<std::size_t> &completed_index)
    {
        const auto counts = verification_counts(report);
        const auto total = report.supported_profiles.size();
        constexpr std::size_t bar_width = 48;
        const std::size_t filled =
            total == 0 ? 0 : counts.attempted * bar_width / total;
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started_at_)
                .count();

        std::ostringstream screen;
        screen
            << "\033[2J\033[H\033[1;36mCamera "
            << (kFpsTestProgram ? "sustained FPS test" : "profile validation")
            << "\033[0m\n\n"
            << '[' << report.identity.backend << "] " << report.identity.name
            << "  serial=" << report.identity.serial << "\n\n["
            << std::string(filled, '#') << std::string(bar_width - filled, '-')
            << "] " << counts.attempted << '/' << total << "\n\n"
            << "\033[1;32mPASS " << counts.passed << "\033[0m    "
            << "\033[1;31mFAIL " << counts.failed << "\033[0m    "
            << "PENDING " << total - counts.attempted << "    elapsed "
            << std::fixed << std::setprecision(1) << elapsed << " s\n";

        if (completed_index) {
            const auto &stream = report.supported_profiles[*completed_index];
            screen << "\nLatest: " << numbered_label('P', *completed_index + 1)
                   << ' ' << profile_description(stream) << "\nResult: "
                   << (stream.verified ? "\033[1;32mPASS\033[0m" :
                                         "\033[1;31mFAIL\033[0m")
                   << "  " << stream.verification_ms << " ms\n";
            if (!stream.verification_error.empty()) {
                screen << "Error : " << stream.verification_error << "\n";
            }
            if (kFpsTestProgram) {
                screen << "FPS   : frames=" << stream.measured_frames
                       << " host=" << stream.host_measured_fps
                       << " device=" << stream.device_measured_fps
                       << " drops=" << stream.dropped_frames
                       << "\nJitter: p95=" << stream.p95_interval_ms
                       << " ms max=" << stream.max_interval_ms << " ms\n";
            }
        } else {
            screen << "\nPreparing first profile...\n";
        }

        const std::size_t focus = completed_index.value_or(0);
        const std::size_t window_begin = focus > 4 ? focus - 4 : 0;
        const std::size_t window_end =
            std::min(total, window_begin + static_cast<std::size_t>(12));
        screen
            << "\nProfiles " << window_begin + 1 << '-' << window_end << " of "
            << total << ":\n"
            << (kFpsTestProgram ?
                    "  ID    state   resolution    nominal measured frames format\n" :
                    "  ID    state   resolution    FPS format\n");
        for (std::size_t index = window_begin; index < window_end; ++index) {
            const auto &stream = report.supported_profiles[index];
            const char *state = !stream.verification_attempted ?
                                    "PENDING " :
                                    (stream.verified ?
                                         "\033[32mPASS\033[0m    " :
                                         "\033[31mFAIL\033[0m    ");
            std::ostringstream resolution;
            resolution << stream.width << 'x' << stream.height;
            screen << (index == focus ? "> " : "  ")
                   << numbered_label('P', index + 1) << "  " << state << " "
                   << std::left << std::setw(14) << resolution.str();
            if (kFpsTestProgram) {
                screen << std::right << std::setw(7) << stream.fps << ' ';
                if (stream.verification_attempted) {
                    screen << std::fixed << std::setprecision(2) << std::setw(8)
                           << stream.host_measured_fps << std::defaultfloat
                           << std::setprecision(9) << std::setw(7)
                           << stream.measured_frames;
                } else {
                    screen << std::setw(8) << "-" << std::setw(7) << "-";
                }
                screen << ' ';
            } else {
                screen << std::right << std::setw(3) << stream.fps << ' ';
            }
            screen << stream.format << (stream.is_default ? " *" : "") << '\n';
        }

        screen << "\nRecent failures:\n";
        std::size_t shown = 0;
        for (std::size_t offset = counts.attempted; offset > 0 && shown < 5;
             --offset) {
            const auto index = offset - 1;
            const auto &stream = report.supported_profiles[index];
            if (!stream.verification_attempted || stream.verified) {
                continue;
            }
            screen << "  " << numbered_label('P', index + 1) << ' '
                   << profile_description(stream) << " — "
                   << stream.verification_error << '\n';
            ++shown;
        }
        if (shown == 0) {
            screen << "  none\n";
        }
        screen << "\nCtrl-C stops after the current profile.\n";
        std::cout << screen.str() << std::flush;
    }

    bool interactive_{ false };
    bool active_{ false };
    Options options_;
    std::chrono::steady_clock::time_point started_at_{};
};

Report inspect_with_display(const Identity &identity, const Options &options)
{
    ValidationDisplay display(options);
    auto report = inspect(
        identity, options, true,
        [&display](const Report &inventory) { display.begin(inventory); },
        [&display](const Report &progress, std::size_t completed_index) {
            display.update(progress, completed_index);
        });
    display.finish(report);
    return report;
}

Report inspect_inventory(const Identity &identity)
{
    return inspect(identity, Options{}, false,
                   [](const Report &report) { print_inventory(report, false); },
                   {});
}

Report run_selected_program(const Identity &identity, const Options &options)
{
    return kValidationProgram ? inspect_with_display(identity, options) :
                                inspect_inventory(identity);
}

bool report_verified(const Report &report)
{
    return !report.supported_profiles.empty() &&
           std::all_of(report.supported_profiles.begin(),
                       report.supported_profiles.end(),
                       [](const StreamInfo &stream) {
                           return stream.verification_attempted &&
                                  stream.verified;
                       });
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const auto options = parse_options(argc, argv);
        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_OFF);
        ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
        ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        auto previous = scan(options.serial);
        bool verification_failed = false;
        if (previous.empty()) {
            std::cout << "No matching supported camera found.\n";
        } else {
            for (const auto &entry : previous) {
                const auto report = run_selected_program(entry.second, options);
                verification_failed = verification_failed ||
                                      (kValidationProgram ?
                                           !report_verified(report) :
                                           report.supported_profiles.empty());
            }
        }
        if (!options.watch) {
            return previous.empty() || verification_failed ? 1 : 0;
        }
        std::cout << "Watching camera hot-plug events; press Ctrl-C to stop.\n";
        while (running.load()) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(options.interval));
            const auto current = scan(options.serial);
            for (const auto &entry : previous) {
                if (current.count(entry.first) == 0) {
                    std::cout << "\n[removed] " << entry.second.backend
                              << " serial=" << entry.second.serial << '\n';
                }
            }
            for (const auto &entry : current) {
                if (previous.count(entry.first) == 0) {
                    std::cout << "\n[added] " << entry.second.backend
                              << " serial=" << entry.second.serial << '\n';
                    run_selected_program(entry.second, options);
                }
            }
            previous = current;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << kProgramName << ": " << error.what() << '\n';
        return 2;
    }
}
