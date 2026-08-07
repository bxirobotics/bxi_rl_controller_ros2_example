#pragma once

#include "bxi_depth_camera/camera_worker.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace bxi_depth_camera
{

class CameraManager final : public rclcpp::Node {
public:
    explicit CameraManager(
        const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~CameraManager() override;

private:
    using Proposed = std::unordered_map<std::string, rclcpp::Parameter>;

    struct Failure {
        std::chrono::steady_clock::time_point retry_at;
        std::string message;
        std::chrono::steady_clock::time_point last_log;
    };

    void declare_defaults();
    void reconcile();
    std::vector<DeviceDescriptor> discover();
    void start_camera(const DeviceDescriptor &descriptor,
                      std::chrono::steady_clock::time_point now);
    void remove_camera(const std::string &key, const std::string &reason);
    void close() noexcept;

    CameraConfig read_config(const std::string &logical_name = {},
                             const Proposed *proposed = nullptr) const;
    rclcpp::Parameter effective_parameter(const std::string &leaf,
                                          const std::string &logical_name,
                                          const Proposed *proposed) const;
    std::set<std::string> camera_names(const Proposed *proposed = nullptr) const;
    std::map<std::string, std::string>
    serial_mappings(const Proposed *proposed = nullptr) const;
    std::string logical_name_for(const std::string &serial) const;
    void update_single_camera_fallback(
        const std::vector<DeviceDescriptor> &selected);
    rcl_interfaces::msg::SetParametersResult
    on_parameters(const std::vector<rclcpp::Parameter> &parameters);

    static bool camera_parameter_parts(const std::string &name,
                                       std::string &logical_name,
                                       std::string &leaf);
    static bool supported_config_parameter(const std::string &name);
    static void validate_parameter(const std::string &name,
                                   const rclcpp::Parameter &parameter);

    rclcpp::TimerBase::SharedPtr timer_;
    OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
    std::unordered_map<std::string, std::unique_ptr<CameraDevice>> workers_;
    std::unordered_map<std::string, Failure> failures_;
    std::set<std::string> pending_restarts_;
    std::optional<std::string> single_unmapped_serial_;
    std::chrono::steady_clock::time_point next_discovery_{};
    bool closed_{ false };
};

} // namespace bxi_depth_camera
