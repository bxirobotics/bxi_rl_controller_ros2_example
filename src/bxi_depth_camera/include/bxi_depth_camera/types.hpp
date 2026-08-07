#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bxi_depth_camera
{

struct StreamProfile {
    int width{ 0 };
    int height{ 0 };
    int fps{ 0 };

    bool automatic() const noexcept
    {
        return width == 0 && height == 0 && fps == 0;
    }
};

struct DeviceDescriptor {
    std::string backend;
    std::string serial;
    std::string name;
    std::string uid;

    std::string key() const
    {
        return backend + ":" + serial;
    }
};

struct CameraConfig {
    std::string camera_namespace{ "hardware" };
    StreamProfile depth_profile{};
    StreamProfile color_profile{};
    bool enable_depth{ true };
    bool enable_color{ true };
    bool enable_infra1{ false };
    bool enable_infra2{ false };
    bool enable_gyro{ false };
    bool enable_accel{ false };
    bool align_depth{ false };
    bool pointcloud_enabled{ false };
    bool pointcloud_ordered{ false };
    bool pointcloud_allow_no_texture_points{ false };
    double pointcloud_max_fps{ 10.0 };
    bool rectify_depth{ false };
    bool rectify_color{ true };
    bool rectify_infra1{ false };
    bool rectify_infra2{ false };
    double device_timeout_sec{ 3.0 };
    bool decimation_enabled{ false };
    int decimation_magnitude{ 1 };
    bool spatial_enabled{ true };
    double spatial_alpha{ 0.45 };
    double spatial_delta{ 20.0 };
    int spatial_holes_fill{ 2 };
    bool temporal_enabled{ true };
    double temporal_alpha{ 0.45 };
    double temporal_delta{ 20.0 };
    int temporal_holes_fill{ 4 };
    bool hole_filling_enabled{ true };
    int hole_filling_mode{ 1 };
    bool second_hole_filling_enabled{ true };
    int second_hole_filling_mode{ 2 };
    bool orbbec_enable_sdk_filters{ true };
    double orbbec_fallback_hfov{ 90.0 };
    double orbbec_fallback_vfov{ 65.0 };

    void validate() const;
};

struct Calibration {
    double fx{ 0.0 };
    double fy{ 0.0 };
    double cx{ 0.0 };
    double cy{ 0.0 };
    std::vector<double> distortion;
    std::string distortion_model{ "plumb_bob" };
};

struct PointXYZRGB {
    float x{ 0.0F };
    float y{ 0.0F };
    float z{ 0.0F };
    std::uint8_t r{ 0 };
    std::uint8_t g{ 0 };
    std::uint8_t b{ 0 };
    bool has_color{ false };
    bool texture_valid{ true };
};

StreamProfile parse_profile(const std::string &value, const std::string &name);
std::string camera_name_token(const std::string &value);
std::string topic_token(const std::string &serial);
std::string node_token(const std::string &serial);

class CameraDevice {
public:
    virtual ~CameraDevice() = default;
    virtual const DeviceDescriptor &descriptor() const noexcept = 0;
    virtual const std::string &logical_name() const noexcept = 0;
    virtual bool stale(std::chrono::steady_clock::time_point now) const = 0;
    virtual void stop() noexcept = 0;
};

} // namespace bxi_depth_camera
