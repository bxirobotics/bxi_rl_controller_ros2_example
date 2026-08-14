#include "bxi_depth_camera/camera_worker.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>

namespace bxi_depth_camera
{
namespace
{

std::string trim_slashes(std::string value)
{
    while (!value.empty() && value.front() == '/') {
        value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string normalize_profile(std::string value)
{
    std::replace(value.begin(), value.end(), 'x', ',');
    std::replace(value.begin(), value.end(), 'X', ',');
    return value;
}

template <typename PublisherT>
bool has_subscribers(const std::shared_ptr<PublisherT> &publisher)
{
    return publisher && publisher->get_subscription_count() > 0;
}

std::int64_t steady_ns(std::chrono::steady_clock::time_point value)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               value.time_since_epoch())
        .count();
}

} // namespace

void CameraConfig::validate() const
{
    if (camera_namespace.empty() || trim_slashes(camera_namespace).empty()) {
        throw std::invalid_argument(
            "camera_namespace must be a non-empty string");
    }
    if (align_depth && !(enable_depth && enable_color)) {
        throw std::invalid_argument(
            "align_depth.enable requires both enable_depth and enable_color");
    }
    if (pointcloud_enabled && !enable_depth) {
        throw std::invalid_argument("pointcloud.enable requires enable_depth");
    }
    if (!std::isfinite(pointcloud_max_fps) || pointcloud_max_fps <= 0.0) {
        throw std::invalid_argument(
            "pointcloud.max_fps must be greater than zero");
    }
    if (!std::isfinite(device_timeout_sec) || device_timeout_sec <= 0.0) {
        throw std::invalid_argument(
            "device_timeout_sec must be greater than zero");
    }
}

StreamProfile parse_profile(const std::string &value, const std::string &name)
{
    const std::string normalized = normalize_profile(value);
    std::smatch match;
    static const std::regex pattern(R"(^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$)");
    if (!std::regex_match(normalized, match, pattern)) {
        throw std::invalid_argument(
            name + " must use WIDTHxHEIGHTxFPS or WIDTH,HEIGHT,FPS");
    }
    StreamProfile profile{ std::stoi(match[1].str()), std::stoi(match[2].str()),
                           std::stoi(match[3].str()) };
    if (profile.automatic()) {
        return profile;
    }
    if (profile.width <= 0 || profile.height <= 0 || profile.fps <= 0) {
        throw std::invalid_argument(
            name + " must be 0,0,0 or contain positive values");
    }
    return profile;
}

std::string camera_name_token(const std::string &value)
{
    static const std::regex pattern(R"(^[A-Za-z][A-Za-z0-9_]*$)");
    if (!std::regex_match(value, pattern)) {
        throw std::invalid_argument(
            "logical camera name must start with a letter and contain only letters, digits, or underscores");
    }
    return value;
}

std::string topic_token(const std::string &serial)
{
    static const std::regex pattern(R"(^[A-Za-z0-9_]+$)");
    if (serial.empty() || !std::regex_match(serial, pattern)) {
        throw std::invalid_argument(
            "camera serial must contain only letters, digits, or underscores");
    }
    return "SN_" + serial;
}

std::string node_token(const std::string &serial)
{
    return "camera_" + topic_token(serial);
}

CameraWorker::CameraWorker(rclcpp::Node &node, DeviceDescriptor descriptor,
                           std::string logical_name, CameraConfig config)
    : node_(node)
    , descriptor_(std::move(descriptor))
    , logical_name_(camera_name_token(logical_name))
    , config_(std::move(config))
    , started_(std::chrono::steady_clock::now())
    , last_pointcloud_(std::chrono::steady_clock::time_point::min())
{
    config_.validate();
    const auto qos = rclcpp::SensorDataQoS();
    if (config_.enable_depth) {
        pub_depth_ = node_.create_publisher<sensor_msgs::msg::Image>(
            topic("depth/image_rect_raw"), qos);
        pub_depth_info_ = node_.create_publisher<sensor_msgs::msg::CameraInfo>(
            topic("depth/camera_info"), qos);
    }
    if (config_.enable_color) {
        pub_color_ = node_.create_publisher<sensor_msgs::msg::Image>(
            topic("color/image_raw"), qos);
        pub_color_info_ = node_.create_publisher<sensor_msgs::msg::CameraInfo>(
            topic("color/camera_info"), qos);
    }
    if (config_.align_depth) {
        pub_aligned_depth_ = node_.create_publisher<sensor_msgs::msg::Image>(
            topic("aligned_depth_to_color/image_raw"), qos);
        pub_aligned_depth_info_ =
            node_.create_publisher<sensor_msgs::msg::CameraInfo>(
                topic("aligned_depth_to_color/camera_info"), qos);
    }
    if (config_.enable_infra1) {
        pub_infra1_ = node_.create_publisher<sensor_msgs::msg::Image>(
            topic("infra1/image_rect_raw"), qos);
        pub_infra1_info_ = node_.create_publisher<sensor_msgs::msg::CameraInfo>(
            topic("infra1/camera_info"), qos);
    }
    if (config_.enable_infra2) {
        pub_infra2_ = node_.create_publisher<sensor_msgs::msg::Image>(
            topic("infra2/image_rect_raw"), qos);
        pub_infra2_info_ = node_.create_publisher<sensor_msgs::msg::CameraInfo>(
            topic("infra2/camera_info"), qos);
    }
    if (config_.enable_gyro) {
        pub_gyro_ = node_.create_publisher<sensor_msgs::msg::Imu>(
            topic("gyro/sample"), qos);
    }
    if (config_.enable_accel) {
        pub_accel_ = node_.create_publisher<sensor_msgs::msg::Imu>(
            topic("accel/sample"), qos);
    }
    if (config_.pointcloud_enabled) {
        pub_pointcloud_ = node_.create_publisher<sensor_msgs::msg::PointCloud2>(
            topic("depth/color/points"), qos);
        pointcloud_thread_ =
            std::thread(&CameraWorker::run_pointcloud_worker, this);
    }
}

CameraWorker::~CameraWorker()
{
    CameraWorker::stop();
}

void CameraWorker::stop() noexcept
{
    if (stopped_.exchange(true)) {
        return;
    }
    stop_pointcloud_worker();
}

void CameraWorker::mark_frame() noexcept
{
    last_frame_ns_.store(steady_ns(std::chrono::steady_clock::now()),
                         std::memory_order_relaxed);
}

bool CameraWorker::stale(std::chrono::steady_clock::time_point now) const
{
    const auto raw = last_frame_ns_.load(std::memory_order_relaxed);
    const auto reference = raw == 0 ? started_ :
                                      std::chrono::steady_clock::time_point(
                                          std::chrono::nanoseconds(raw));
    return std::chrono::duration<double>(now - reference).count() >
           config_.device_timeout_sec;
}

bool CameraWorker::requested(const rclcpp::PublisherBase::SharedPtr &publisher)
{
    return publisher && publisher->get_subscription_count() > 0;
}

bool CameraWorker::depth_requested() const
{
    return has_subscribers(pub_depth_) || has_subscribers(pub_depth_info_);
}
bool CameraWorker::color_requested() const
{
    return has_subscribers(pub_color_) || has_subscribers(pub_color_info_);
}
bool CameraWorker::aligned_depth_requested() const
{
    return has_subscribers(pub_aligned_depth_) ||
           has_subscribers(pub_aligned_depth_info_);
}
bool CameraWorker::infra1_requested() const
{
    return has_subscribers(pub_infra1_) || has_subscribers(pub_infra1_info_);
}
bool CameraWorker::infra2_requested() const
{
    return has_subscribers(pub_infra2_) || has_subscribers(pub_infra2_info_);
}
bool CameraWorker::video_consumers_requested() const
{
    return depth_requested() || color_requested() ||
           aligned_depth_requested() || infra1_requested() ||
           infra2_requested() || has_subscribers(pub_pointcloud_);
}

bool CameraWorker::pointcloud_requested()
{
    if (!has_subscribers(pub_pointcloud_)) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto period =
        std::chrono::duration<double>(1.0 / config_.pointcloud_max_fps);
    if (now - last_pointcloud_ < period) {
        return false;
    }
    last_pointcloud_ = now;
    return true;
}

std::string CameraWorker::topic(const std::string &suffix) const
{
    return "/" + trim_slashes(config_.camera_namespace) + "/" + logical_name_ +
           "/" + suffix;
}
std::string CameraWorker::depth_frame_id() const
{
    return logical_name_ + "_depth_optical_frame";
}
std::string CameraWorker::color_frame_id() const
{
    return logical_name_ + "_color_optical_frame";
}
std::string CameraWorker::infra1_frame_id() const
{
    return logical_name_ + "_infra1_optical_frame";
}
std::string CameraWorker::infra2_frame_id() const
{
    return logical_name_ + "_infra2_optical_frame";
}
std::string CameraWorker::gyro_frame_id() const
{
    return logical_name_ + "_gyro_optical_frame";
}
std::string CameraWorker::accel_frame_id() const
{
    return logical_name_ + "_accel_optical_frame";
}

const CameraWorker::RectificationMaps &
CameraWorker::rectification_maps(const std::string &key, int width, int height,
                                 const Calibration &calibration)
{
    auto found = rectification_maps_.find(key);
    if (found != rectification_maps_.end()) {
        return found->second;
    }
    cv::Mat camera = (cv::Mat_<double>(3, 3) << calibration.fx, 0.0,
                      calibration.cx, 0.0, calibration.fy, calibration.cy, 0.0,
                      0.0, 1.0);
    cv::Mat distortion(calibration.distortion, true);
    RectificationMaps maps;
    const cv::Size size(width, height);
    if (calibration.distortion_model == "equidistant") {
        cv::fisheye::initUndistortRectifyMap(camera, distortion,
                                             cv::Mat::eye(3, 3, CV_64F), camera,
                                             size, CV_32FC1, maps.x, maps.y);
    } else {
        cv::initUndistortRectifyMap(camera, distortion,
                                    cv::Mat::eye(3, 3, CV_64F), camera, size,
                                    CV_32FC1, maps.x, maps.y);
    }
    return rectification_maps_.emplace(key, std::move(maps)).first->second;
}

sensor_msgs::msg::CameraInfo CameraWorker::make_camera_info(
    int width, int height, const std::string &frame_id,
    const rclcpp::Time &stamp, const Calibration &calibration,
    bool rectified) const
{
    sensor_msgs::msg::CameraInfo info;
    info.header.stamp = stamp;
    info.header.frame_id = frame_id;
    info.width = static_cast<std::uint32_t>(width);
    info.height = static_cast<std::uint32_t>(height);
    info.distortion_model = calibration.distortion_model;
    info.d =
        rectified ?
            std::vector<double>(
                std::max<std::size_t>(5, calibration.distortion.size()), 0.0) :
            calibration.distortion;
    if (info.d.empty()) {
        info.d.assign(5, 0.0);
    }
    info.k = { calibration.fx,
               0.0,
               calibration.cx,
               0.0,
               calibration.fy,
               calibration.cy,
               0.0,
               0.0,
               1.0 };
    info.r = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    info.p = { calibration.fx,
               0.0,
               calibration.cx,
               0.0,
               0.0,
               calibration.fy,
               calibration.cy,
               0.0,
               0.0,
               0.0,
               1.0,
               0.0 };
    return info;
}

void CameraWorker::publish_calibrated_image(
    const void *data, int width, int height, std::size_t source_step,
    int cv_type, const std::string &encoding, const std::string &frame_id,
    const rclcpp::Time &stamp, const ImagePublisher::SharedPtr &image_publisher,
    const InfoPublisher::SharedPtr &info_publisher,
    const Calibration &calibration, bool rectify, bool depth)
{
    if (!image_publisher || !info_publisher || data == nullptr || width <= 0 ||
        height <= 0) {
        return;
    }
    cv::Mat source(height, width, cv_type, const_cast<void *>(data),
                   source_step);
    cv::Mat output = source;
    bool rectified = false;
    if (rectify && !calibration.distortion.empty() &&
        std::any_of(calibration.distortion.begin(),
                    calibration.distortion.end(),
                    [](double value) { return std::abs(value) > 1e-12; })) {
        try {
            const auto &maps =
                rectification_maps(frame_id + ":" + std::to_string(width) +
                                       "x" + std::to_string(height),
                                   width, height, calibration);
            cv::remap(source, output, maps.x, maps.y,
                      depth ? cv::INTER_NEAREST : cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT);
            rectified = true;
        } catch (const cv::Exception &error) {
            warn_throttled("rectification-" + frame_id,
                           "cannot rectify " + frame_id +
                               "; publishing SDK frame: " + error.what());
            output = source;
        }
    }

    sensor_msgs::msg::Image image;
    image.header.stamp = stamp;
    image.header.frame_id = frame_id;
    image.height = static_cast<std::uint32_t>(height);
    image.width = static_cast<std::uint32_t>(width);
    image.encoding = encoding;
    image.is_bigendian = false;
    const auto packed_step =
        static_cast<std::size_t>(width) * output.elemSize();
    image.step = static_cast<std::uint32_t>(packed_step);
    image.data.resize(packed_step * static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        std::memcpy(image.data.data() +
                        static_cast<std::size_t>(row) * packed_step,
                    output.ptr(row), packed_step);
    }
    info_publisher->publish(make_camera_info(width, height, frame_id, stamp,
                                             calibration, rectified));
    image_publisher->publish(std::move(image));
}

void CameraWorker::publish_imu(const ImuPublisher::SharedPtr &publisher,
                               const std::string &frame_id, bool angular,
                               float x, float y, float z)
{
    if (!has_subscribers(publisher)) {
        return;
    }
    sensor_msgs::msg::Imu message;
    message.header.stamp = node_.now();
    message.header.frame_id = frame_id;
    if (angular) {
        message.angular_velocity.x = x;
        message.angular_velocity.y = y;
        message.angular_velocity.z = z;
    } else {
        message.linear_acceleration.x = x;
        message.linear_acceleration.y = y;
        message.linear_acceleration.z = z;
    }
    publisher->publish(std::move(message));
}

void CameraWorker::publish_pointcloud(const std::vector<PointXYZRGB> &points,
                                      std::uint32_t width, std::uint32_t height,
                                      const rclcpp::Time &stamp)
{
    if (!has_subscribers(pub_pointcloud_) || points.empty()) {
        return;
    }
    const bool with_color =
        std::any_of(points.begin(), points.end(),
                    [](const PointXYZRGB &point) { return point.has_color; });
    const std::size_t point_step = with_color ? 16U : 12U;
    std::vector<std::size_t> selected;
    selected.reserve(points.size());
    if (config_.pointcloud_ordered) {
        if (points.size() != static_cast<std::size_t>(width) * height) {
            throw std::runtime_error(
                "ordered point cloud size does not match frame dimensions");
        }
        selected.resize(points.size());
        for (std::size_t index = 0; index < points.size(); ++index) {
            selected[index] = index;
        }
    } else {
        for (std::size_t index = 0; index < points.size(); ++index) {
            const auto &point = points[index];
            const bool geometric = std::isfinite(point.x) &&
                                   std::isfinite(point.y) &&
                                   std::isfinite(point.z) && point.z > 0.0F;
            const bool textured = point.texture_valid ||
                                  config_.pointcloud_allow_no_texture_points;
            if (geometric && textured) {
                selected.push_back(index);
            }
        }
        width = static_cast<std::uint32_t>(selected.size());
        height = 1;
    }

    sensor_msgs::msg::PointCloud2 message;
    message.header.stamp = stamp;
    message.header.frame_id = depth_frame_id();
    message.width = width;
    message.height = height;
    message.is_bigendian = false;
    message.is_dense = !config_.pointcloud_ordered;
    message.point_step = static_cast<std::uint32_t>(point_step);
    message.row_step = message.point_step * message.width;
    for (std::uint32_t index = 0; index < 3; ++index) {
        sensor_msgs::msg::PointField field;
        field.name = index == 0 ? "x" : (index == 1 ? "y" : "z");
        field.offset = index * 4;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        message.fields.push_back(field);
    }
    if (with_color) {
        sensor_msgs::msg::PointField field;
        field.name = "rgb";
        field.offset = 12;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        message.fields.push_back(field);
    }
    message.data.assign(selected.size() * point_step, 0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t output_index = 0; output_index < selected.size();
         ++output_index) {
        const auto &point = points[selected[output_index]];
        const bool valid =
            std::isfinite(point.x) && std::isfinite(point.y) &&
            std::isfinite(point.z) && point.z > 0.0F &&
            (point.texture_valid || config_.pointcloud_allow_no_texture_points);
        const float xyz[3] = { valid ? point.x : nan, valid ? point.y : nan,
                               valid ? point.z : nan };
        auto *destination = message.data.data() + output_index * point_step;
        std::memcpy(destination, xyz, sizeof(xyz));
        if (with_color) {
            const std::uint32_t rgb =
                point.has_color && point.texture_valid ?
                    (static_cast<std::uint32_t>(point.r) << 16U) |
                        (static_cast<std::uint32_t>(point.g) << 8U) |
                        static_cast<std::uint32_t>(point.b) :
                    0U;
            std::memcpy(destination + 12, &rgb, sizeof(rgb));
        }
    }
    pub_pointcloud_->publish(std::move(message));
}

void CameraWorker::enqueue_pointcloud(std::function<void()> task)
{
    if (!pub_pointcloud_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pointcloud_mutex_);
        if (pointcloud_stopping_) {
            return;
        }
        pointcloud_task_ = std::move(task);
    }
    pointcloud_cv_.notify_one();
}

void CameraWorker::run_pointcloud_worker()
{
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(pointcloud_mutex_);
            pointcloud_cv_.wait(lock, [this] {
                return pointcloud_stopping_ || pointcloud_task_.has_value();
            });
            if (pointcloud_stopping_) {
                return;
            }
            task = std::move(*pointcloud_task_);
            pointcloud_task_.reset();
        }
        if (!has_subscribers(pub_pointcloud_)) {
            continue;
        }
        try {
            task();
        } catch (const std::exception &error) {
            error_throttled("pointcloud",
                            descriptor_.backend + " " + descriptor_.serial +
                                " point cloud failed: " + error.what());
        }
    }
}

void CameraWorker::stop_pointcloud_worker() noexcept
{
    {
        std::lock_guard<std::mutex> lock(pointcloud_mutex_);
        pointcloud_stopping_ = true;
        pointcloud_task_.reset();
    }
    pointcloud_cv_.notify_all();
    if (pointcloud_thread_.joinable()) {
        pointcloud_thread_.join();
    }
}

void CameraWorker::warn_throttled(const std::string &key,
                                  const std::string &message) const
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(log_mutex_);
    const auto found = log_times_.find(key);
    if (found != log_times_.end() &&
        now - found->second < std::chrono::seconds(5)) {
        return;
    }
    log_times_[key] = now;
    RCLCPP_WARN(node_.get_logger(), "%s", message.c_str());
}

void CameraWorker::error_throttled(const std::string &key,
                                   const std::string &message) const
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(log_mutex_);
    const auto found = log_times_.find(key);
    if (found != log_times_.end() &&
        now - found->second < std::chrono::seconds(5)) {
        return;
    }
    log_times_[key] = now;
    RCLCPP_ERROR(node_.get_logger(), "%s", message.c_str());
}

} // namespace bxi_depth_camera
