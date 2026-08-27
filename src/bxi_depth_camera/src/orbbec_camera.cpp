#include "bxi_depth_camera/camera_worker.hpp"

#include <libobsensor/ObSensor.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace bxi_depth_camera
{
namespace
{

constexpr int kOrbbecVendorId = 0x2BC5;
constexpr int kGemini335ProductId = 0x0800;

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

void configure_orbbec_logging()
{
    static std::once_flag configured;
    std::call_once(configured, [] {
        ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
        ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
    });
}

struct OrbbecDiscoveryContext {
    // Context destruction unregisters the callback before generation dies.
    std::atomic<std::uint64_t> generation{ 1 };
    ob::Context context;

    OrbbecDiscoveryContext()
    {
        context.registerDeviceChangedCallback(
            [this](std::shared_ptr<ob::DeviceList>,
                   std::shared_ptr<ob::DeviceList>) {
                generation.fetch_add(1, std::memory_order_relaxed);
            });
    }
};

OrbbecDiscoveryContext &orbbec_discovery_context()
{
    configure_orbbec_logging();
    static OrbbecDiscoveryContext instance;
    return instance;
}

std::shared_ptr<ob::Context> make_orbbec_context()
{
    configure_orbbec_logging();
    return std::make_shared<ob::Context>();
}

Calibration
calibration_from(const std::shared_ptr<ob::VideoStreamProfile> &profile,
                 double fallback_hfov, double fallback_vfov)
{
    Calibration calibration;
    try {
        const auto intrinsic = profile->getIntrinsic();
        const auto distortion = profile->getDistortion();
        calibration.fx = intrinsic.fx;
        calibration.fy = intrinsic.fy;
        calibration.cx = intrinsic.cx;
        calibration.cy = intrinsic.cy;
        calibration.distortion = { distortion.k1, distortion.k2, distortion.p1,
                                   distortion.p2, distortion.k3, distortion.k4,
                                   distortion.k5, distortion.k6 };
    } catch (const std::exception &) {
        const double width = profile->getWidth();
        const double height = profile->getHeight();
        calibration.fx = width / (2.0 * std::tan(fallback_hfov * M_PI / 360.0));
        calibration.fy =
            height / (2.0 * std::tan(fallback_vfov * M_PI / 360.0));
        calibration.cx = (width - 1.0) / 2.0;
        calibration.cy = (height - 1.0) / 2.0;
        calibration.distortion.assign(5, 0.0);
    }
    return calibration;
}

std::shared_ptr<ob::VideoStreamProfile> select_profile(
    const std::shared_ptr<ob::StreamProfileList> &profiles,
    const StreamProfile &requested, const std::vector<OBFormat> &formats)
{
    std::exception_ptr last_error;
    for (const auto format : formats) {
        try {
            return profiles->getVideoStreamProfile(
                requested.automatic() ? OB_WIDTH_ANY : requested.width,
                requested.automatic() ? OB_HEIGHT_ANY : requested.height,
                format, requested.automatic() ? OB_FPS_ANY : requested.fps);
        } catch (...) {
            last_error = std::current_exception();
        }
    }
    if (last_error) {
        std::rethrow_exception(last_error);
    }
    throw std::runtime_error("no compatible Orbbec stream profile");
}

class OrbbecCamera final : public CameraWorker {
public:
    OrbbecCamera(rclcpp::Node &node, DeviceDescriptor descriptor,
                 std::string logical_name, CameraConfig config)
        : CameraWorker(node, std::move(descriptor), std::move(logical_name),
                       std::move(config))
        , context_(make_orbbec_context())
    {
        auto devices = context_->queryDeviceList();
        device_ = devices->getDeviceBySN(descriptor_.serial.c_str());
        configure_device_options();
        start_pipeline();
        video_thread_ = std::thread(&OrbbecCamera::run_video_worker, this);
    }

    ~OrbbecCamera() override
    {
        stop();
    }

    void stop() noexcept override
    {
        if (local_stopped_.exchange(true)) {
            return;
        }
        if (imu_pipeline_) {
            try {
                imu_pipeline_->stop();
            } catch (const std::exception &error) {
                warn_throttled("orbbec-imu-stop", error.what());
            }
            imu_pipeline_.reset();
        }
        if (pipeline_) {
            try {
                pipeline_->stop();
            } catch (const std::exception &error) {
                warn_throttled("orbbec-stop", error.what());
            }
            pipeline_.reset();
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
        align_to_color_.reset();
        pointcloud_filter_.reset();
        depth_filters_.clear();
        device_.reset();
        context_.reset();
        CameraWorker::stop();
    }

private:
    struct MotionSample {
        rclcpp::Time stamp;
        float x{ 0.0F };
        float y{ 0.0F };
        float z{ 0.0F };
    };

    bool roi_configured() const noexcept
    {
        return config_.auto_exposure_roi_left >= 0;
    }

    void set_bool_property(OBPropertyID property, bool value,
                           const std::string &name)
    {
        if (!device_->isPropertySupported(property, OB_PERMISSION_WRITE)) {
            warn_throttled("orbbec-property-unsupported-" + name,
                           "Orbbec " + descriptor_.serial +
                               " does not support writable property " + name);
            return;
        }
        device_->setBoolProperty(property, value);
    }

    void set_int_property(OBPropertyID property, int value,
                          const std::string &name)
    {
        if (!device_->isPropertySupported(property, OB_PERMISSION_WRITE)) {
            warn_throttled("orbbec-property-unsupported-" + name,
                           "Orbbec " + descriptor_.serial +
                               " does not support writable property " + name);
            return;
        }
        const auto range = device_->getIntPropertyRange(property);
        const int clamped = std::clamp(value, range.min, range.max);
        device_->setIntProperty(property, clamped);
    }

    void set_roi_property(OBPropertyID property, const std::string &name)
    {
        if (!roi_configured()) {
            return;
        }
        if (!device_->isPropertySupported(property, OB_PERMISSION_WRITE)) {
            warn_throttled("orbbec-roi-unsupported-" + name,
                           "Orbbec " + descriptor_.serial +
                               " does not support writable AE ROI " + name);
            return;
        }
        OBRegionOfInterest roi{};
        roi.x0_left = static_cast<int16_t>(config_.auto_exposure_roi_left);
        roi.y0_top = static_cast<int16_t>(config_.auto_exposure_roi_top);
        roi.x1_right = static_cast<int16_t>(config_.auto_exposure_roi_right);
        roi.y1_bottom = static_cast<int16_t>(config_.auto_exposure_roi_bottom);
        device_->setStructuredData(property,
                                   reinterpret_cast<const uint8_t *>(&roi),
                                   sizeof(roi));
    }

    void configure_device_options()
    {
        try {
            if (!config_.color_enable_auto_exposure ||
                config_.color_exposure >= 0.0 || config_.color_gain >= 0.0) {
                set_bool_property(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                                  config_.color_enable_auto_exposure,
                                  "color_auto_exposure");
            }
            if (config_.color_exposure >= 0.0) {
                set_int_property(OB_PROP_COLOR_EXPOSURE_INT,
                                 static_cast<int>(std::lround(
                                     config_.color_exposure)),
                                 "color_exposure");
            }
            if (config_.color_gain >= 0.0) {
                set_int_property(OB_PROP_COLOR_GAIN_INT,
                                 static_cast<int>(
                                     std::lround(config_.color_gain)),
                                 "color_gain");
            }
            if (!config_.depth_enable_auto_exposure ||
                config_.depth_exposure >= 0.0 || config_.depth_gain >= 0.0) {
                set_bool_property(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
                                  config_.depth_enable_auto_exposure,
                                  "depth_auto_exposure");
            }
            if (config_.depth_exposure >= 0.0) {
                set_int_property(OB_PROP_DEPTH_EXPOSURE_INT,
                                 static_cast<int>(std::lround(
                                     config_.depth_exposure)),
                                 "depth_exposure");
            }
            if (config_.depth_gain >= 0.0) {
                set_int_property(OB_PROP_DEPTH_GAIN_INT,
                                 static_cast<int>(
                                     std::lround(config_.depth_gain)),
                                 "depth_gain");
            }
            set_roi_property(OB_STRUCT_COLOR_AE_ROI, "color");
            set_roi_property(OB_STRUCT_DEPTH_AE_ROI, "depth");
        } catch (const std::exception &error) {
            warn_throttled("orbbec-options",
                           "cannot configure Orbbec " + descriptor_.serial +
                               ": " + error.what());
        }
    }

    void start_pipeline()
    {
        pipeline_ = std::make_shared<ob::Pipeline>(device_);
        auto config = std::make_shared<ob::Config>();
        if (config_.enable_depth) {
            auto profiles = pipeline_->getStreamProfileList(OB_SENSOR_DEPTH);
            depth_profile_ = select_profile(profiles, config_.depth_profile,
                                            { OB_FORMAT_Y16, OB_FORMAT_ANY });
            config->enableStream(depth_profile_);
        }
        if (config_.enable_color) {
            auto profiles = pipeline_->getStreamProfileList(OB_SENSOR_COLOR);
            color_profile_ =
                select_profile(profiles, config_.color_profile,
                               { OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_MJPG,
                                 OB_FORMAT_YUYV, OB_FORMAT_UYVY });
            config->enableStream(color_profile_);
        }
        configure_ir(config, OB_SENSOR_IR_LEFT, OB_STREAM_IR_LEFT,
                     config_.enable_infra1);
        configure_ir(config, OB_SENSOR_IR_RIGHT, OB_STREAM_IR_RIGHT,
                     config_.enable_infra2);
        if ((config_.enable_infra1 || config_.enable_infra2) &&
            !infra1_profile_ && !infra2_profile_) {
            configure_ir(config, OB_SENSOR_IR, OB_STREAM_IR, true);
        }

        if (config_.align_depth || config_.enable_sync ||
            config_.enable_rgbd ||
            (config_.pointcloud_enabled && config_.enable_color)) {
            config->setFrameAggregateOutputMode(
                OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
            pipeline_->enableFrameSync();
        }
        align_to_color_ = config_.align_depth ?
                              std::make_shared<ob::Align>(OB_STREAM_COLOR) :
                              nullptr;
        if (config_.pointcloud_enabled) {
            pointcloud_filter_ = std::make_shared<ob::PointCloudFilter>();
            pointcloud_filter_->setCreatePointFormat(OB_FORMAT_POINT);
        }
        initialize_depth_filters();
        pipeline_->start(config,
                         [this](std::shared_ptr<ob::FrameSet> frameset) {
                             on_video_frames(std::move(frameset));
                         });
        if (config_.enable_gyro || config_.enable_accel) {
            start_imu_pipeline();
        }
        RCLCPP_INFO(node_.get_logger(), "started Orbbec %s serial=%s",
                    descriptor_.name.c_str(), descriptor_.serial.c_str());
    }

    void configure_ir(const std::shared_ptr<ob::Config> &config,
                      OBSensorType sensor, OBStreamType, bool enabled)
    {
        if (!enabled) {
            return;
        }
        try {
            auto profile = select_profile(
                pipeline_->getStreamProfileList(sensor), config_.depth_profile,
                { OB_FORMAT_Y8, OB_FORMAT_Y16, OB_FORMAT_ANY });
            config->enableStream(profile);
            if (sensor == OB_SENSOR_IR_RIGHT) {
                infra2_profile_ = profile;
            } else {
                infra1_profile_ = profile;
            }
        } catch (const std::exception &) {
        }
    }

    void initialize_depth_filters()
    {
        if (!config_.enable_depth) {
            return;
        }
        if (config_.threshold_enabled) {
            auto threshold = std::make_shared<ob::ThresholdFilter>();
            threshold->enable(true);
            const auto min_mm = static_cast<uint16_t>(std::clamp(
                std::lround(config_.threshold_min_distance * 1000.0), 0L,
                65535L));
            const auto max_mm = static_cast<uint16_t>(std::clamp(
                std::lround(config_.threshold_max_distance * 1000.0), 0L,
                65535L));
            if (!threshold->setValueRange(min_mm, max_mm)) {
                throw std::runtime_error("invalid Orbbec threshold filter range");
            }
            RCLCPP_DEBUG(node_.get_logger(),
                         "Orbbec %s threshold filter schema: %s",
                         descriptor_.serial.c_str(),
                         threshold->getConfigSchema().c_str());
            depth_filters_.push_back(threshold);
        }
        auto sensor = device_->getSensor(OB_SENSOR_DEPTH);
        for (auto &filter : sensor->createRecommendedFilters()) {
            const std::string name = lower(filter->getName());
            RCLCPP_DEBUG(node_.get_logger(), "Orbbec %s SDK filter %s schema: %s",
                         descriptor_.serial.c_str(), filter->getName().c_str(),
                         filter->getConfigSchema().c_str());
            const bool mandatory = name.find("disparity") != std::string::npos;
            const bool selected = config_.orbbec_enable_sdk_filters &&
                                  (name.find("noise") != std::string::npos ||
                                   name.find("spatial") != std::string::npos ||
                                   name.find("temporal") != std::string::npos ||
                                   name.find("hole") != std::string::npos);
            filter->enable(mandatory || selected);
            if (mandatory || selected) {
                depth_filters_.push_back(filter);
            }
        }
    }

    void start_imu_pipeline()
    {
        imu_pipeline_ = std::make_shared<ob::Pipeline>(device_);
        auto config = std::make_shared<ob::Config>();
        if (config_.enable_gyro) {
            config->enableGyroStream();
        }
        if (config_.enable_accel) {
            config->enableAccelStream();
        }
        config->setFrameAggregateOutputMode(
            OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
        imu_pipeline_->start(config,
                             [this](std::shared_ptr<ob::FrameSet> frameset) {
                                 on_imu_frames(std::move(frameset));
                             });
    }

    void on_video_frames(std::shared_ptr<ob::FrameSet> frameset)
    {
        if (!frameset) {
            return;
        }
        mark_frame();
        if (!video_consumers_requested()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(video_mutex_);
            latest_frameset_ = std::move(frameset);
        }
        video_cv_.notify_one();
    }

    void on_imu_frames(const std::shared_ptr<ob::FrameSet> &frameset)
    {
        if (!frameset) {
            return;
        }
        try {
            mark_frame();
            std::optional<MotionSample> accel_sample;
            std::optional<MotionSample> gyro_sample;
            const auto stamp = node_.now();
            if (config_.enable_accel) {
                auto frame = frameset->getFrame(OB_FRAME_ACCEL);
                if (frame) {
                    const auto value = frame->as<ob::AccelFrame>()->getValue();
                    accel_sample = MotionSample{ stamp, value.x, value.y,
                                                 value.z };
                    publish_imu(pub_accel_, accel_frame_id(), stamp, false,
                                value.x, value.y, value.z);
                }
            }
            if (config_.enable_gyro) {
                auto frame = frameset->getFrame(OB_FRAME_GYRO);
                if (frame) {
                    const auto value = frame->as<ob::GyroFrame>()->getValue();
                    gyro_sample = MotionSample{ stamp, value.x, value.y,
                                                value.z };
                    publish_imu(pub_gyro_, gyro_frame_id(), stamp, true,
                                value.x, value.y, value.z);
                }
            }
            handle_imu_samples(gyro_sample, accel_sample, stamp);
        } catch (const std::exception &error) {
            error_throttled("orbbec-imu-callback",
                            "Orbbec " + descriptor_.serial +
                                " IMU callback failed: " + error.what());
        }
    }

    void handle_imu_samples(const std::optional<MotionSample> &gyro,
                            const std::optional<MotionSample> &accel,
                            const rclcpp::Time &stamp)
    {
        if (!pub_imu_) {
            return;
        }
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (gyro) {
            latest_gyro_ = gyro;
        }
        if (accel) {
            latest_accel_ = accel;
        }
        if (latest_gyro_ && latest_accel_) {
            publish_combined_imu(stamp, latest_gyro_->x, latest_gyro_->y,
                                 latest_gyro_->z, latest_accel_->x,
                                 latest_accel_->y, latest_accel_->z);
        }
    }

    void run_video_worker()
    {
        while (true) {
            std::shared_ptr<ob::FrameSet> frameset;
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
                error_throttled("orbbec-frameset",
                                "Orbbec " + descriptor_.serial +
                                    " frame failed: " + error.what());
            }
        }
    }

    void publish_frameset(const std::shared_ptr<ob::FrameSet> &frameset)
    {
        const bool need_depth = depth_requested();
        const bool need_color = color_requested();
        const bool need_aligned = aligned_depth_requested();
        const bool need_rgbd = rgbd_requested();
        const bool need_infra1 = infra1_requested();
        const bool need_infra2 = infra2_requested();
        const bool need_pointcloud = pointcloud_requested();
        if (!(need_depth || need_color || need_aligned || need_infra1 ||
              need_infra2 || need_rgbd || need_pointcloud)) {
            return;
        }
        const auto stamp = node_.now();
        if (need_depth) {
            publish_depth(frameset->getDepthFrame(), stamp, pub_depth_,
                          pub_depth_info_, depth_frame_id(),
                          config_.rectify_depth, true);
        }
        if (need_color) {
            auto color = frameset->getColorFrame();
            if (color) {
                auto image = color_bgr(color);
                const auto profile =
                    color->getStreamProfile()->as<ob::VideoStreamProfile>();
                publish_calibrated_image(
                    image.data, image.cols, image.rows, image.step, CV_8UC3,
                    "bgr8", color_frame_id(), stamp, pub_color_,
                    pub_color_info_,
                    calibration_from(profile, config_.orbbec_fallback_hfov,
                                     config_.orbbec_fallback_vfov),
                    config_.rectify_color, false);
            }
        }
        std::shared_ptr<ob::FrameSet> aligned;
        if ((need_aligned || need_rgbd) && align_to_color_) {
            auto aligned_frame = align_to_color_->process(frameset);
            aligned = aligned_frame ? aligned_frame->as<ob::FrameSet>() :
                                      nullptr;
        }
        if (need_aligned && align_to_color_) {
            if (!aligned || !aligned->getDepthFrame()) {
                warn_throttled(
                    "orbbec-align-no-output",
                    "Orbbec " + descriptor_.serial +
                        " align filter returned no compatible depth frame");
            } else {
                publish_depth(aligned->getDepthFrame(), stamp,
                              pub_aligned_depth_, pub_aligned_depth_info_,
                              color_frame_id(), config_.rectify_color, false);
            }
        }
        if (need_rgbd) {
            publish_rgbd(frameset->getColorFrame(),
                         aligned ? aligned->getDepthFrame() : nullptr, stamp);
        }
        if (need_infra1) {
            auto frame = frameset->getFrame(OB_FRAME_IR_LEFT);
            if (!frame) {
                frame = frameset->getFrame(OB_FRAME_IR);
            }
            publish_ir(frame, stamp, pub_infra1_, pub_infra1_info_,
                       infra1_frame_id(), config_.rectify_infra1);
        }
        if (need_infra2) {
            publish_ir(frameset->getFrame(OB_FRAME_IR_RIGHT), stamp,
                       pub_infra2_, pub_infra2_info_, infra2_frame_id(),
                       config_.rectify_infra2);
        }
        if (need_pointcloud) {
            enqueue_pointcloud([this, frameset, stamp] {
                create_pointcloud(frameset, stamp);
            });
        }
    }

    std::vector<std::uint16_t>
    depth_millimeters(const std::shared_ptr<ob::DepthFrame> &depth) const
    {
        if (!depth) {
            return {};
        }
        const int width = static_cast<int>(depth->getWidth());
        const int height = static_cast<int>(depth->getHeight());
        const auto pixels = static_cast<std::size_t>(width) * height;
        if (depth->getDataSize() < pixels * sizeof(std::uint16_t)) {
            throw std::runtime_error("malformed Orbbec depth frame");
        }
        const auto *raw =
            reinterpret_cast<const std::uint16_t *>(depth->getData());
        const float scale = depth->getValueScale();
        std::vector<std::uint16_t> millimeters(pixels);
        for (std::size_t index = 0; index < pixels; ++index) {
            millimeters[index] = static_cast<std::uint16_t>(
                std::clamp(std::lround(static_cast<double>(raw[index]) * scale),
                           0L, 65535L));
        }
        return millimeters;
    }

    void publish_rgbd(const std::shared_ptr<ob::ColorFrame> &color,
                      const std::shared_ptr<ob::DepthFrame> &depth,
                      const rclcpp::Time &stamp)
    {
#if BXI_DEPTH_CAMERA_HAS_RGBD_MSG
        if (!has_rgbd_inputs(color, depth)) {
            return;
        }
        auto color_image = color_bgr(color);
        auto depth_image = depth_millimeters(depth);
        const auto color_profile =
            color->getStreamProfile()->as<ob::VideoStreamProfile>();
        const auto calibration =
            calibration_from(color_profile, config_.orbbec_fallback_hfov,
                             config_.orbbec_fallback_vfov);
        realsense2_camera_msgs::msg::RGBD message;
        message.header.stamp = stamp;
        message.header.frame_id = color_frame_id();
        message.rgb = make_image(color_image.data, color_image.cols,
                                 color_image.rows, color_image.step, CV_8UC3,
                                 "bgr8", color_frame_id(), stamp);
        message.depth = make_image(
            depth_image.data(), static_cast<int>(depth->getWidth()),
            static_cast<int>(depth->getHeight()),
            static_cast<std::size_t>(depth->getWidth()) *
                sizeof(std::uint16_t),
            CV_16UC1, "16UC1", color_frame_id(), stamp);
        message.rgb_camera_info = make_camera_info(
            color_image.cols, color_image.rows, color_frame_id(), stamp,
            calibration, false);
        message.depth_camera_info = make_camera_info(
            static_cast<int>(depth->getWidth()),
            static_cast<int>(depth->getHeight()), color_frame_id(), stamp,
            calibration, false);
        pub_rgbd_->publish(std::move(message));
#else
        (void)color;
        (void)depth;
        (void)stamp;
#endif
    }

    bool has_rgbd_inputs(const std::shared_ptr<ob::ColorFrame> &color,
                         const std::shared_ptr<ob::DepthFrame> &depth) const
    {
#if BXI_DEPTH_CAMERA_HAS_RGBD_MSG
        return pub_rgbd_ && color && depth;
#else
        (void)color;
        (void)depth;
        return false;
#endif
    }

    std::shared_ptr<ob::DepthFrame>
    process_depth(const std::shared_ptr<ob::DepthFrame> &depth)
    {
        if (!depth) {
            return nullptr;
        }
        std::shared_ptr<ob::Frame> processed = depth;
        for (const auto &filter : depth_filters_) {
            processed = filter->process(processed);
            if (!processed) {
                throw std::runtime_error("Orbbec depth filter " +
                                         filter->getName() +
                                         " returned no frame");
            }
        }
        return processed->as<ob::DepthFrame>();
    }

    void publish_depth(const std::shared_ptr<ob::DepthFrame> &source,
                       const rclcpp::Time &stamp,
                       const ImagePublisher::SharedPtr &image,
                       const InfoPublisher::SharedPtr &info,
                       const std::string &frame_id, bool rectify,
                       bool apply_filters)
    {
        auto depth = apply_filters ? process_depth(source) : source;
        if (!depth) {
            return;
        }
        const int width = static_cast<int>(depth->getWidth());
        const int height = static_cast<int>(depth->getHeight());
        auto millimeters = depth_millimeters(depth);
        const auto profile =
            depth->getStreamProfile()->as<ob::VideoStreamProfile>();
        publish_calibrated_image(
            millimeters.data(), width, height,
            static_cast<std::size_t>(width) * sizeof(std::uint16_t), CV_16UC1,
            "16UC1", frame_id, stamp, image, info,
            calibration_from(profile, config_.orbbec_fallback_hfov,
                             config_.orbbec_fallback_vfov),
            rectify, true);
    }

    void publish_ir(const std::shared_ptr<ob::Frame> &source,
                    const rclcpp::Time &stamp,
                    const ImagePublisher::SharedPtr &image,
                    const InfoPublisher::SharedPtr &info,
                    const std::string &frame_id, bool rectify)
    {
        if (!source) {
            return;
        }
        auto frame = source->as<ob::VideoFrame>();
        const int width = static_cast<int>(frame->getWidth());
        const int height = static_cast<int>(frame->getHeight());
        const bool eight_bit = source->getFormat() == OB_FORMAT_Y8;
        const auto profile =
            source->getStreamProfile()->as<ob::VideoStreamProfile>();
        publish_calibrated_image(
            source->getData(), width, height,
            static_cast<std::size_t>(width) * (eight_bit ? 1U : 2U),
            eight_bit ? CV_8UC1 : CV_16UC1, eight_bit ? "mono8" : "mono16",
            frame_id, stamp, image, info,
            calibration_from(profile, config_.orbbec_fallback_hfov,
                             config_.orbbec_fallback_vfov),
            rectify, false);
    }

    cv::Mat color_bgr(const std::shared_ptr<ob::ColorFrame> &frame) const
    {
        const int width = static_cast<int>(frame->getWidth());
        const int height = static_cast<int>(frame->getHeight());
        const auto format = frame->getFormat();
        const auto *data = static_cast<const std::uint8_t *>(frame->getData());
        if (format == OB_FORMAT_BGR) {
            return cv::Mat(height, width, CV_8UC3,
                           const_cast<std::uint8_t *>(data));
        }
        cv::Mat output;
        if (format == OB_FORMAT_RGB) {
            cv::cvtColor(cv::Mat(height, width, CV_8UC3,
                                 const_cast<std::uint8_t *>(data)),
                         output, cv::COLOR_RGB2BGR);
        } else if (format == OB_FORMAT_MJPG) {
            const cv::Mat encoded(1, static_cast<int>(frame->getDataSize()),
                                  CV_8UC1, const_cast<std::uint8_t *>(data));
            output = cv::imdecode(encoded, cv::IMREAD_COLOR);
        } else if (format == OB_FORMAT_YUYV || format == OB_FORMAT_UYVY) {
            cv::cvtColor(cv::Mat(height, width, CV_8UC2,
                                 const_cast<std::uint8_t *>(data)),
                         output,
                         format == OB_FORMAT_YUYV ? cv::COLOR_YUV2BGR_YUY2 :
                                                    cv::COLOR_YUV2BGR_UYVY);
        } else {
            throw std::runtime_error("unsupported Orbbec color format");
        }
        if (output.empty()) {
            throw std::runtime_error("cannot decode Orbbec color frame");
        }
        return output;
    }

    void create_pointcloud(const std::shared_ptr<ob::FrameSet> &frameset,
                           const rclcpp::Time &stamp)
    {
        auto depth = frameset->getDepthFrame();
        if (!depth || !pointcloud_filter_) {
            return;
        }
        pointcloud_filter_->setCreatePointFormat(OB_FORMAT_POINT);
        auto generated = pointcloud_filter_->process(depth);
        if (!generated) {
            throw std::runtime_error(
                "Orbbec point cloud filter returned no frame");
        }
        auto points_frame = generated->as<ob::PointsFrame>();
        const auto count = generated->getDataSize() / sizeof(OBPoint);
        const auto *sdk_points =
            reinterpret_cast<const OBPoint *>(generated->getData());
        const float scale_to_m =
            points_frame->getCoordinateValueScale() * 0.001F;
        std::vector<PointXYZRGB> points(count);
        for (std::size_t index = 0; index < count; ++index) {
            points[index].x = sdk_points[index].x * scale_to_m;
            points[index].y = sdk_points[index].y * scale_to_m;
            points[index].z = sdk_points[index].z * scale_to_m;
        }

        auto color = frameset->getColorFrame();
        if (color && depth->getStreamProfile() && color->getStreamProfile()) {
            const auto depth_profile =
                depth->getStreamProfile()->as<ob::VideoStreamProfile>();
            const auto color_profile =
                color->getStreamProfile()->as<ob::VideoStreamProfile>();
            const auto extrinsic = depth_profile->getExtrinsicTo(color_profile);
            const auto color_intrinsic = color_profile->getIntrinsic();
            const auto color_distortion = color_profile->getDistortion();
            cv::Mat rotation(3, 3, CV_32F, const_cast<float *>(extrinsic.rot));
            cv::Mat rotation_vector;
            cv::Rodrigues(rotation, rotation_vector);
            cv::Mat translation(3, 1, CV_32F,
                                const_cast<float *>(extrinsic.trans));
            const cv::Mat camera =
                (cv::Mat_<double>(3, 3) << color_intrinsic.fx, 0.0,
                 color_intrinsic.cx, 0.0, color_intrinsic.fy,
                 color_intrinsic.cy, 0.0, 0.0, 1.0);
            const cv::Mat distortion =
                (cv::Mat_<double>(1, 8) << color_distortion.k1,
                 color_distortion.k2, color_distortion.p1, color_distortion.p2,
                 color_distortion.k3, color_distortion.k4, color_distortion.k5,
                 color_distortion.k6);
            const float scale_to_mm = points_frame->getCoordinateValueScale();
            std::vector<cv::Point3f> object_points(count);
            for (std::size_t index = 0; index < count; ++index) {
                object_points[index] =
                    cv::Point3f(sdk_points[index].x * scale_to_mm,
                                sdk_points[index].y * scale_to_mm,
                                sdk_points[index].z * scale_to_mm);
            }
            std::vector<cv::Point2f> pixels;
            cv::projectPoints(object_points, rotation_vector, translation,
                              camera, distortion, pixels);
            const cv::Mat image = color_bgr(color);
            for (std::size_t index = 0; index < count; ++index) {
                auto &output = points[index];
                output.has_color = true;
                const int x = static_cast<int>(std::floor(pixels[index].x));
                const int y = static_cast<int>(std::floor(pixels[index].y));
                output.texture_valid = std::isfinite(pixels[index].x) &&
                                       std::isfinite(pixels[index].y) &&
                                       x >= 0 && y >= 0 && x < image.cols &&
                                       y < image.rows;
                if (output.texture_valid) {
                    const auto pixel = image.at<cv::Vec3b>(y, x);
                    output.r = pixel[2];
                    output.g = pixel[1];
                    output.b = pixel[0];
                }
            }
        }
        publish_pointcloud(points,
                           static_cast<std::uint32_t>(depth->getWidth()),
                           static_cast<std::uint32_t>(depth->getHeight()),
                           stamp);
    }

    std::shared_ptr<ob::Context> context_;
    std::shared_ptr<ob::Device> device_;
    std::shared_ptr<ob::Pipeline> pipeline_;
    std::shared_ptr<ob::Pipeline> imu_pipeline_;
    std::shared_ptr<ob::Align> align_to_color_;
    std::shared_ptr<ob::PointCloudFilter> pointcloud_filter_;
    std::vector<std::shared_ptr<ob::Filter>> depth_filters_;
    std::shared_ptr<ob::VideoStreamProfile> depth_profile_;
    std::shared_ptr<ob::VideoStreamProfile> color_profile_;
    std::shared_ptr<ob::VideoStreamProfile> infra1_profile_;
    std::shared_ptr<ob::VideoStreamProfile> infra2_profile_;

    std::mutex imu_mutex_;
    std::optional<MotionSample> latest_gyro_;
    std::optional<MotionSample> latest_accel_;

    std::mutex video_mutex_;
    std::condition_variable video_cv_;
    std::optional<std::shared_ptr<ob::FrameSet>> latest_frameset_;
    bool video_stopping_{ false };
    std::thread video_thread_;
    std::atomic<bool> local_stopped_{ false };
};

} // namespace

std::vector<DeviceDescriptor> discover_orbbec()
{
    std::vector<DeviceDescriptor> result;
    auto devices = orbbec_discovery_context().context.queryDeviceList();
    for (std::uint32_t index = 0; index < devices->getCount(); ++index) {
        try {
            if (devices->getVid(index) != kOrbbecVendorId ||
                devices->getPid(index) != kGemini335ProductId) {
                continue;
            }
            const std::string serial = devices->getSerialNumber(index);
            if (serial.empty()) {
                continue;
            }
            result.push_back(DeviceDescriptor{ "orbbec", serial,
                                               devices->getName(index),
                                               devices->getUid(index) });
        } catch (const std::exception &) {
        }
    }
    return result;
}

std::uint64_t orbbec_device_generation()
{
    return orbbec_discovery_context().generation.load(
        std::memory_order_relaxed);
}

std::unique_ptr<CameraDevice>
make_orbbec_camera(rclcpp::Node &node, const DeviceDescriptor &descriptor,
                   const std::string &logical_name, const CameraConfig &config)
{
    return std::make_unique<OrbbecCamera>(node, descriptor, logical_name,
                                          config);
}

} // namespace bxi_depth_camera
