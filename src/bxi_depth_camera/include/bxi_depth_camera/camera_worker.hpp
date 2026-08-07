#pragma once

#include "bxi_depth_camera/types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace bxi_depth_camera
{

class CameraWorker : public CameraDevice {
public:
    CameraWorker(rclcpp::Node &node, DeviceDescriptor descriptor,
                 std::string logical_name, CameraConfig config);
    ~CameraWorker() override;

    const DeviceDescriptor &descriptor() const noexcept override
    {
        return descriptor_;
    }
    const std::string &logical_name() const noexcept override
    {
        return logical_name_;
    }
    bool stale(std::chrono::steady_clock::time_point now) const override;
    void stop() noexcept override;

protected:
    using ImagePublisher = rclcpp::Publisher<sensor_msgs::msg::Image>;
    using InfoPublisher = rclcpp::Publisher<sensor_msgs::msg::CameraInfo>;
    using ImuPublisher = rclcpp::Publisher<sensor_msgs::msg::Imu>;
    using PointCloudPublisher =
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;

    void mark_frame() noexcept;
    bool video_consumers_requested() const;
    bool depth_requested() const;
    bool color_requested() const;
    bool aligned_depth_requested() const;
    bool infra1_requested() const;
    bool infra2_requested() const;
    bool pointcloud_requested();

    void publish_calibrated_image(
        const void *data, int width, int height, std::size_t source_step,
        int cv_type, const std::string &encoding, const std::string &frame_id,
        const rclcpp::Time &stamp,
        const ImagePublisher::SharedPtr &image_publisher,
        const InfoPublisher::SharedPtr &info_publisher,
        const Calibration &calibration, bool rectify, bool depth);
    void publish_imu(const ImuPublisher::SharedPtr &publisher,
                     const std::string &frame_id, bool angular, float x,
                     float y, float z);
    void publish_pointcloud(const std::vector<PointXYZRGB> &points,
                            std::uint32_t width, std::uint32_t height,
                            const rclcpp::Time &stamp);

    void enqueue_pointcloud(std::function<void()> task);
    void stop_pointcloud_worker() noexcept;
    void warn_throttled(const std::string &key,
                        const std::string &message) const;
    void error_throttled(const std::string &key,
                         const std::string &message) const;

    std::string topic(const std::string &suffix) const;
    std::string depth_frame_id() const;
    std::string color_frame_id() const;
    std::string infra1_frame_id() const;
    std::string infra2_frame_id() const;
    std::string gyro_frame_id() const;
    std::string accel_frame_id() const;

    rclcpp::Node &node_;
    DeviceDescriptor descriptor_;
    std::string logical_name_;
    CameraConfig config_;

    ImagePublisher::SharedPtr pub_depth_;
    InfoPublisher::SharedPtr pub_depth_info_;
    ImagePublisher::SharedPtr pub_color_;
    InfoPublisher::SharedPtr pub_color_info_;
    ImagePublisher::SharedPtr pub_aligned_depth_;
    InfoPublisher::SharedPtr pub_aligned_depth_info_;
    ImagePublisher::SharedPtr pub_infra1_;
    InfoPublisher::SharedPtr pub_infra1_info_;
    ImagePublisher::SharedPtr pub_infra2_;
    InfoPublisher::SharedPtr pub_infra2_info_;
    ImuPublisher::SharedPtr pub_gyro_;
    ImuPublisher::SharedPtr pub_accel_;
    PointCloudPublisher::SharedPtr pub_pointcloud_;

private:
    struct RectificationMaps {
        cv::Mat x;
        cv::Mat y;
    };

    sensor_msgs::msg::CameraInfo
    make_camera_info(int width, int height, const std::string &frame_id,
                     const rclcpp::Time &stamp, const Calibration &calibration,
                     bool rectified) const;
    const RectificationMaps &rectification_maps(const std::string &key,
                                                int width, int height,
                                                const Calibration &calibration);
    void run_pointcloud_worker();
    static bool requested(const rclcpp::PublisherBase::SharedPtr &publisher);

    std::chrono::steady_clock::time_point started_;
    std::atomic<std::int64_t> last_frame_ns_{ 0 };
    std::chrono::steady_clock::time_point last_pointcloud_{};
    std::atomic<bool> stopped_{ false };

    mutable std::mutex log_mutex_;
    mutable std::map<std::string, std::chrono::steady_clock::time_point>
        log_times_;
    std::map<std::string, RectificationMaps> rectification_maps_;

    std::mutex pointcloud_mutex_;
    std::condition_variable pointcloud_cv_;
    std::optional<std::function<void()>> pointcloud_task_;
    bool pointcloud_stopping_{ false };
    std::thread pointcloud_thread_;
};

std::vector<DeviceDescriptor> discover_realsense();
std::vector<DeviceDescriptor> discover_orbbec();

std::unique_ptr<CameraDevice> make_realsense_camera(
    rclcpp::Node &node, const DeviceDescriptor &descriptor,
    const std::string &logical_name, const CameraConfig &config);
std::unique_ptr<CameraDevice>
make_orbbec_camera(rclcpp::Node &node, const DeviceDescriptor &descriptor,
                   const std::string &logical_name, const CameraConfig &config);

} // namespace bxi_depth_camera
