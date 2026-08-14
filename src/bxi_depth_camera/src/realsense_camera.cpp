#include "bxi_depth_camera/camera_worker.hpp"

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace bxi_depth_camera
{
namespace
{

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

class RealSenseCamera final : public CameraWorker {
public:
    RealSenseCamera(rclcpp::Node &node, DeviceDescriptor descriptor,
                    std::string logical_name, CameraConfig config)
        : CameraWorker(node, std::move(descriptor), std::move(logical_name),
                       std::move(config))
        , pipeline_(context_)
        , align_to_color_(RS2_STREAM_COLOR)
        , hole_filter_(config_.hole_filling_mode)
        , second_hole_filter_(config_.second_hole_filling_mode)
    {
        start_pipeline();
        video_thread_ = std::thread(&RealSenseCamera::run_video_worker, this);
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
        try {
            pipeline_.stop();
        } catch (const std::exception &error) {
            warn_throttled("realsense-stop", error.what());
        }
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
    void start_pipeline()
    {
        rs2::config config;
        config.enable_device(descriptor_.serial);
        if (config_.enable_depth) {
            if (config_.depth_profile.automatic()) {
                config.enable_stream(RS2_STREAM_DEPTH);
            } else {
                config.enable_stream(RS2_STREAM_DEPTH,
                                     config_.depth_profile.width,
                                     config_.depth_profile.height,
                                     RS2_FORMAT_Z16, config_.depth_profile.fps);
            }
        }
        if (config_.enable_color) {
            if (config_.color_profile.automatic()) {
                config.enable_stream(RS2_STREAM_COLOR);
            } else {
                config.enable_stream(RS2_STREAM_COLOR,
                                     config_.color_profile.width,
                                     config_.color_profile.height,
                                     RS2_FORMAT_BGR8,
                                     config_.color_profile.fps);
            }
        }
        if (config_.enable_infra1) {
            if (config_.depth_profile.automatic()) {
                config.enable_stream(RS2_STREAM_INFRARED, 1);
            } else {
                config.enable_stream(RS2_STREAM_INFRARED, 1,
                                     config_.depth_profile.width,
                                     config_.depth_profile.height,
                                     RS2_FORMAT_Y8, config_.depth_profile.fps);
            }
        }
        if (config_.enable_infra2) {
            if (config_.depth_profile.automatic()) {
                config.enable_stream(RS2_STREAM_INFRARED, 2);
            } else {
                config.enable_stream(RS2_STREAM_INFRARED, 2,
                                     config_.depth_profile.width,
                                     config_.depth_profile.height,
                                     RS2_FORMAT_Y8, config_.depth_profile.fps);
            }
        }
        if (config_.enable_gyro) {
            config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
        }
        if (config_.enable_accel) {
            config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
        }

        auto profile = pipeline_.start(config, [this](rs2::frame frame) {
            on_frame(std::move(frame));
        });
        configure_device(profile.get_device());
        configure_filters();
        RCLCPP_INFO(node_.get_logger(), "started RealSense %s serial=%s",
                    descriptor_.name.c_str(), descriptor_.serial.c_str());
    }

    void configure_device(const rs2::device &device)
    {
        for (auto sensor : device.query_sensors()) {
            try {
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
            } catch (const std::exception &error) {
                warn_throttled("realsense-options",
                               "cannot configure RealSense " +
                                   descriptor_.serial + ": " + error.what());
            }
        }
    }

    void configure_filters()
    {
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
            mark_frame();
            if (auto frameset = frame.as<rs2::frameset>()) {
                if (!video_consumers_requested()) {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(video_mutex_);
                    latest_frameset_ = std::move(frameset);
                }
                video_cv_.notify_one();
                return;
            }
            auto motion = frame.as<rs2::motion_frame>();
            if (!motion) {
                return;
            }
            const auto stream = motion.get_profile().stream_type();
            if (stream == RS2_STREAM_GYRO && pub_gyro_ &&
                pub_gyro_->get_subscription_count() > 0) {
                const auto value = motion.get_motion_data();
                publish_imu(pub_gyro_, gyro_frame_id(), true, value.x, value.y,
                            value.z);
            } else if (stream == RS2_STREAM_ACCEL && pub_accel_ &&
                       pub_accel_->get_subscription_count() > 0) {
                const auto value = motion.get_motion_data();
                publish_imu(pub_accel_, accel_frame_id(), false, value.x,
                            value.y, value.z);
            }
        } catch (const std::exception &error) {
            error_throttled("realsense-callback",
                            "RealSense " + descriptor_.serial +
                                " callback failed: " + error.what());
        }
    }

    void run_video_worker()
    {
        while (true) {
            rs2::frameset frameset;
            {
                std::unique_lock<std::mutex> lock(video_mutex_);
                video_cv_.wait(lock, [this] {
                    return video_stopping_ || latest_frameset_.has_value();
                });
                if (video_stopping_) {
                    return;
                }
                frameset = std::move(*latest_frameset_);
                latest_frameset_.reset();
            }
            try {
                publish_frameset(frameset);
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

    void publish_frameset(const rs2::frameset &frameset)
    {
        const bool need_depth = depth_requested();
        const bool need_color = color_requested();
        const bool need_aligned = aligned_depth_requested();
        const bool need_infra1 = infra1_requested();
        const bool need_infra2 = infra2_requested();
        const bool need_pointcloud = pointcloud_requested();
        if (!(need_depth || need_color || need_aligned || need_infra1 ||
              need_infra2 || need_pointcloud)) {
            return;
        }
        const auto stamp = node_.now();
        rs2::frameset processed = frameset;
        if (config_.enable_depth &&
            (need_depth || need_aligned || need_pointcloud)) {
            processed = process_depth(frameset);
        }
        if (need_depth) {
            publish_depth(processed.get_depth_frame(), stamp, false);
        }
        if (need_color) {
            publish_video(frameset.get_color_frame(), stamp, pub_color_,
                          pub_color_info_, config_.rectify_color, "bgr8",
                          color_frame_id(), CV_8UC3);
        }
        if (need_aligned) {
            auto aligned =
                align_to_color_.process(processed).as<rs2::frameset>();
            auto depth = aligned.get_depth_frame();
            auto color = frameset.get_color_frame();
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
        if (need_infra1) {
            publish_video(frameset.get_infrared_frame(1), stamp, pub_infra1_,
                          pub_infra1_info_, config_.rectify_infra1, "mono8",
                          infra1_frame_id(), CV_8UC1);
        }
        if (need_infra2) {
            publish_video(frameset.get_infrared_frame(2), stamp, pub_infra2_,
                          pub_infra2_info_, config_.rectify_infra2, "mono8",
                          infra2_frame_id(), CV_8UC1);
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
        publish_calibrated_image(
            frame.get_data(), frame.get_width(), frame.get_height(),
            static_cast<std::size_t>(frame.get_stride_in_bytes()), CV_16UC1,
            "16UC1", aligned ? color_frame_id() : depth_frame_id(), stamp,
            aligned ? pub_aligned_depth_ : pub_depth_,
            aligned ? pub_aligned_depth_info_ : pub_depth_info_,
            calibration_from(profile.get_intrinsics()),
            aligned ? config_.rectify_color : config_.rectify_depth, true);
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
        publish_calibrated_image(
            frame.get_data(), frame.get_width(), frame.get_height(),
            static_cast<std::size_t>(frame.get_stride_in_bytes()), cv_type,
            encoding, frame_id, stamp, image, info,
            calibration_from(profile.get_intrinsics()), rectify, false);
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
    rs2::pipeline pipeline_;
    rs2::align align_to_color_;
    rs2::decimation_filter decimation_filter_;
    rs2::spatial_filter spatial_filter_;
    rs2::temporal_filter temporal_filter_;
    rs2::hole_filling_filter hole_filter_;
    rs2::hole_filling_filter second_hole_filter_;
    rs2::pointcloud pointcloud_;

    std::mutex video_mutex_;
    std::condition_variable video_cv_;
    std::optional<rs2::frameset> latest_frameset_;
    bool video_stopping_{ false };
    std::thread video_thread_;
    std::atomic<bool> local_stopped_{ false };
};

} // namespace

std::vector<DeviceDescriptor> discover_realsense()
{
    std::vector<DeviceDescriptor> result;
    rs2::context context;
    for (auto device : context.query_devices()) {
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

std::unique_ptr<CameraDevice> make_realsense_camera(
    rclcpp::Node &node, const DeviceDescriptor &descriptor,
    const std::string &logical_name, const CameraConfig &config)
{
    return std::make_unique<RealSenseCamera>(node, descriptor, logical_name,
                                             config);
}

} // namespace bxi_depth_camera
