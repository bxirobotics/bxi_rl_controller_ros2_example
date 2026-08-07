#include "bxi_depth_camera/manager.hpp"

#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>

namespace bxi_depth_camera
{
namespace
{

const std::map<std::string, rclcpp::ParameterValue> &config_defaults()
{
    static const std::map<std::string, rclcpp::ParameterValue> values = {
        { "camera_namespace", rclcpp::ParameterValue("hardware") },
        { "device_timeout_sec", rclcpp::ParameterValue(3.0) },
        { "enable_depth", rclcpp::ParameterValue(true) },
        { "enable_color", rclcpp::ParameterValue(true) },
        { "enable_infra1", rclcpp::ParameterValue(false) },
        { "enable_infra2", rclcpp::ParameterValue(false) },
        { "enable_gyro", rclcpp::ParameterValue(false) },
        { "enable_accel", rclcpp::ParameterValue(false) },
        { "align_depth.enable", rclcpp::ParameterValue(false) },
        { "pointcloud.enable", rclcpp::ParameterValue(false) },
        { "pointcloud.ordered_pc", rclcpp::ParameterValue(false) },
        { "pointcloud.allow_no_texture_points", rclcpp::ParameterValue(false) },
        { "pointcloud.max_fps", rclcpp::ParameterValue(10.0) },
        { "depth_module.depth_profile", rclcpp::ParameterValue("0,0,0") },
        { "depth_module.rectification.enable", rclcpp::ParameterValue(false) },
        { "rgb_camera.color_profile", rclcpp::ParameterValue("0,0,0") },
        { "rgb_camera.rectification.enable", rclcpp::ParameterValue(true) },
        { "infra1.rectification.enable", rclcpp::ParameterValue(false) },
        { "infra2.rectification.enable", rclcpp::ParameterValue(false) },
        { "decimation_filter.enable", rclcpp::ParameterValue(false) },
        { "decimation_filter.filter_magnitude", rclcpp::ParameterValue(1) },
        { "spatial_filter.enable", rclcpp::ParameterValue(true) },
        { "spatial_filter.filter_smooth_alpha", rclcpp::ParameterValue(0.45) },
        { "spatial_filter.filter_smooth_delta", rclcpp::ParameterValue(20.0) },
        { "spatial_filter.holes_fill", rclcpp::ParameterValue(2) },
        { "temporal_filter.enable", rclcpp::ParameterValue(true) },
        { "temporal_filter.filter_smooth_alpha", rclcpp::ParameterValue(0.45) },
        { "temporal_filter.filter_smooth_delta", rclcpp::ParameterValue(20.0) },
        { "temporal_filter.holes_fill", rclcpp::ParameterValue(4) },
        { "hole_filling_filter.enable", rclcpp::ParameterValue(true) },
        { "hole_filling_filter.holes_fill", rclcpp::ParameterValue(1) },
        { "second_hole_filling_filter.enable", rclcpp::ParameterValue(true) },
        { "second_hole_filling_filter.holes_fill", rclcpp::ParameterValue(2) },
        { "orbbec.enable_sdk_filters", rclcpp::ParameterValue(true) },
        { "orbbec.fallback_hfov", rclcpp::ParameterValue(90.0) },
        { "orbbec.fallback_vfov", rclcpp::ParameterValue(65.0) },
    };
    return values;
}

const std::map<std::string, rclcpp::ParameterValue> &manager_defaults()
{
    static const std::map<std::string, rclcpp::ParameterValue> values = {
        { "serial_no", rclcpp::ParameterValue("") },
        { "single_camera_name", rclcpp::ParameterValue("head_depth_camera") },
        { "discovery_interval_sec", rclcpp::ParameterValue(1.0) },
        { "retry_interval_sec", rclcpp::ParameterValue(2.0) },
    };
    return values;
}

double number(const rclcpp::Parameter &parameter)
{
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
        return parameter.as_double();
    }
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
        return static_cast<double>(parameter.as_int());
    }
    throw std::invalid_argument(parameter.get_name() + " must be a number");
}

double positive(const rclcpp::Parameter &parameter)
{
    const double value = number(parameter);
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(parameter.get_name() +
                                    " must be greater than zero");
    }
    return value;
}

int bounded_integer(const rclcpp::Parameter &parameter, int minimum,
                    int maximum)
{
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
        throw std::invalid_argument(parameter.get_name() +
                                    " must be an integer");
    }
    const auto value = parameter.as_int();
    if (value < minimum || value > maximum) {
        throw std::invalid_argument(parameter.get_name() +
                                    " is outside the supported range");
    }
    return static_cast<int>(value);
}

bool boolean(const rclcpp::Parameter &parameter)
{
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
        throw std::invalid_argument(parameter.get_name() +
                                    " must be a boolean");
    }
    return parameter.as_bool();
}

std::string text(const rclcpp::Parameter &parameter)
{
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
        throw std::invalid_argument(parameter.get_name() + " must be a string");
    }
    return parameter.as_string();
}

std::string strip_leading_underscores(std::string value)
{
    value.erase(0, value.find_first_not_of('_'));
    return value;
}

} // namespace

CameraManager::CameraManager(const rclcpp::NodeOptions &options)
    : rclcpp::Node("depth_camera_manager",
                   rclcpp::NodeOptions(options)
                       .allow_undeclared_parameters(true)
                       .automatically_declare_parameters_from_overrides(true))
{
    declare_defaults();
    read_config().validate();
    camera_name_token(get_parameter("single_camera_name").as_string());
    serial_mappings();
    parameter_callback_ = add_on_set_parameters_callback(
        std::bind(&CameraManager::on_parameters, this, std::placeholders::_1));
    timer_ = create_wall_timer(std::chrono::milliseconds(100),
                               std::bind(&CameraManager::reconcile, this));
    RCLCPP_INFO(
        get_logger(),
        "camera SDK providers: RealSense:librealsense2, Orbbec:OrbbecSDK_v2 2.9.3");
}

CameraManager::~CameraManager()
{
    close();
}

void CameraManager::declare_defaults()
{
    for (const auto &entry : manager_defaults()) {
        if (!has_parameter(entry.first)) {
            declare_parameter(entry.first, entry.second);
        }
    }
    for (const auto &entry : config_defaults()) {
        if (!has_parameter(entry.first)) {
            declare_parameter(entry.first, entry.second);
        }
    }
}

bool CameraManager::supported_config_parameter(const std::string &name)
{
    return config_defaults().count(name) != 0;
}

bool CameraManager::camera_parameter_parts(const std::string &name,
                                           std::string &logical_name,
                                           std::string &leaf)
{
    static const std::regex pattern(
        R"(^cameras\.([A-Za-z][A-Za-z0-9_]*)\.(.+)$)");
    std::smatch match;
    if (!std::regex_match(name, match, pattern)) {
        return false;
    }
    logical_name = match[1].str();
    leaf = match[2].str();
    return leaf == "serial_no" || supported_config_parameter(leaf);
}

rclcpp::Parameter CameraManager::effective_parameter(
    const std::string &leaf, const std::string &logical_name,
    const Proposed *proposed) const
{
    if (!logical_name.empty()) {
        const std::string specific = "cameras." + logical_name + "." + leaf;
        if (proposed) {
            const auto found = proposed->find(specific);
            if (found != proposed->end() &&
                found->second.get_type() !=
                    rclcpp::ParameterType::PARAMETER_NOT_SET) {
                return found->second;
            }
            if (found != proposed->end()) {
                return effective_parameter(leaf, {}, proposed);
            }
        }
        rclcpp::Parameter value;
        if (get_parameter(specific, value)) {
            return value;
        }
    }
    if (proposed) {
        const auto found = proposed->find(leaf);
        if (found != proposed->end() &&
            found->second.get_type() !=
                rclcpp::ParameterType::PARAMETER_NOT_SET) {
            return found->second;
        }
    }
    return get_parameter(leaf);
}

CameraConfig CameraManager::read_config(const std::string &logical_name,
                                        const Proposed *proposed) const
{
    auto value = [this, &logical_name, proposed](const std::string &name) {
        return effective_parameter(name, logical_name, proposed);
    };
    CameraConfig config;
    config.camera_namespace = text(value("camera_namespace"));
    config.depth_profile =
        parse_profile(text(value("depth_module.depth_profile")),
                      "depth_module.depth_profile");
    config.color_profile = parse_profile(
        text(value("rgb_camera.color_profile")), "rgb_camera.color_profile");
    config.enable_depth = boolean(value("enable_depth"));
    config.enable_color = boolean(value("enable_color"));
    config.enable_infra1 = boolean(value("enable_infra1"));
    config.enable_infra2 = boolean(value("enable_infra2"));
    config.enable_gyro = boolean(value("enable_gyro"));
    config.enable_accel = boolean(value("enable_accel"));
    config.align_depth = boolean(value("align_depth.enable"));
    config.pointcloud_enabled = boolean(value("pointcloud.enable"));
    config.pointcloud_ordered = boolean(value("pointcloud.ordered_pc"));
    config.pointcloud_allow_no_texture_points =
        boolean(value("pointcloud.allow_no_texture_points"));
    config.pointcloud_max_fps = positive(value("pointcloud.max_fps"));
    config.rectify_depth = boolean(value("depth_module.rectification.enable"));
    config.rectify_color = boolean(value("rgb_camera.rectification.enable"));
    config.rectify_infra1 = boolean(value("infra1.rectification.enable"));
    config.rectify_infra2 = boolean(value("infra2.rectification.enable"));
    config.device_timeout_sec = positive(value("device_timeout_sec"));
    config.decimation_enabled = boolean(value("decimation_filter.enable"));
    config.decimation_magnitude =
        bounded_integer(value("decimation_filter.filter_magnitude"), 1, 8);
    config.spatial_enabled = boolean(value("spatial_filter.enable"));
    config.spatial_alpha = number(value("spatial_filter.filter_smooth_alpha"));
    config.spatial_delta = number(value("spatial_filter.filter_smooth_delta"));
    config.spatial_holes_fill =
        bounded_integer(value("spatial_filter.holes_fill"), 0, 5);
    config.temporal_enabled = boolean(value("temporal_filter.enable"));
    config.temporal_alpha =
        number(value("temporal_filter.filter_smooth_alpha"));
    config.temporal_delta =
        number(value("temporal_filter.filter_smooth_delta"));
    config.temporal_holes_fill =
        bounded_integer(value("temporal_filter.holes_fill"), 0, 8);
    config.hole_filling_enabled = boolean(value("hole_filling_filter.enable"));
    config.hole_filling_mode =
        bounded_integer(value("hole_filling_filter.holes_fill"), 0, 2);
    config.second_hole_filling_enabled =
        boolean(value("second_hole_filling_filter.enable"));
    config.second_hole_filling_mode =
        bounded_integer(value("second_hole_filling_filter.holes_fill"), 0, 2);
    config.orbbec_enable_sdk_filters =
        boolean(value("orbbec.enable_sdk_filters"));
    config.orbbec_fallback_hfov = number(value("orbbec.fallback_hfov"));
    config.orbbec_fallback_vfov = number(value("orbbec.fallback_vfov"));
    config.validate();
    return config;
}

std::set<std::string> CameraManager::camera_names(const Proposed *proposed) const
{
    std::set<std::string> names;
    const auto listed = list_parameters({ "cameras" }, 10);
    for (const auto &name : listed.names) {
        std::string logical;
        std::string leaf;
        if (!camera_parameter_parts(name, logical, leaf)) {
            throw std::invalid_argument("invalid per-camera parameter " + name);
        }
        names.insert(logical);
    }
    if (proposed) {
        for (const auto &entry : *proposed) {
            std::string logical;
            std::string leaf;
            if (camera_parameter_parts(entry.first, logical, leaf)) {
                names.insert(logical);
            }
        }
    }
    return names;
}

std::map<std::string, std::string>
CameraManager::serial_mappings(const Proposed *proposed) const
{
    std::map<std::string, std::string> result;
    for (const auto &logical : camera_names(proposed)) {
        const std::string parameter_name = "cameras." + logical + ".serial_no";
        rclcpp::Parameter parameter;
        if (proposed) {
            const auto found = proposed->find(parameter_name);
            if (found != proposed->end()) {
                if (found->second.get_type() ==
                    rclcpp::ParameterType::PARAMETER_NOT_SET) {
                    continue;
                }
                parameter = found->second;
            } else if (!get_parameter(parameter_name, parameter)) {
                continue;
            }
        } else if (!get_parameter(parameter_name, parameter)) {
            continue;
        }
        std::string serial;
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
            serial = parameter.as_string();
        } else if (parameter.get_type() ==
                   rclcpp::ParameterType::PARAMETER_INTEGER) {
            serial = std::to_string(parameter.as_int());
        } else {
            throw std::invalid_argument(parameter_name +
                                        " must be a string or integer");
        }
        if (serial.empty()) {
            continue;
        }
        const auto inserted = result.emplace(serial, logical);
        if (!inserted.second && inserted.first->second != logical) {
            throw std::invalid_argument("camera serial " + serial +
                                        " is assigned more than once");
        }
    }
    return result;
}

std::string CameraManager::logical_name_for(const std::string &serial) const
{
    const auto mappings = serial_mappings();
    const auto found = mappings.find(serial);
    if (found != mappings.end()) {
        return found->second;
    }
    if (single_unmapped_serial_ && *single_unmapped_serial_ == serial) {
        return camera_name_token(
            get_parameter("single_camera_name").as_string());
    }
    return topic_token(serial);
}

void CameraManager::update_single_camera_fallback(
    const std::vector<DeviceDescriptor> &selected)
{
    single_unmapped_serial_.reset();
    if (selected.size() != 1) {
        return;
    }
    const auto mappings = serial_mappings();
    const auto &serial = selected.front().serial;
    const auto fallback =
        camera_name_token(get_parameter("single_camera_name").as_string());
    if (mappings.count(serial) != 0) {
        return;
    }
    const bool reserved = std::any_of(mappings.begin(), mappings.end(),
                                      [&fallback](const auto &entry) {
                                          return entry.second == fallback;
                                      });
    if (!reserved) {
        single_unmapped_serial_ = serial;
    }
}

std::vector<DeviceDescriptor> CameraManager::discover()
{
    std::vector<DeviceDescriptor> descriptors;
    try {
        auto values = discover_realsense();
        descriptors.insert(descriptors.end(), values.begin(), values.end());
    } catch (const std::exception &error) {
        RCLCPP_WARN(get_logger(), "RealSense discovery failed: %s",
                    error.what());
    }
    try {
        auto values = discover_orbbec();
        descriptors.insert(descriptors.end(), values.begin(), values.end());
    } catch (const std::exception &error) {
        RCLCPP_WARN(get_logger(), "Orbbec discovery failed: %s", error.what());
    }
    const std::string selected_serial =
        strip_leading_underscores(get_parameter("serial_no").as_string());
    if (!selected_serial.empty()) {
        descriptors.erase(
            std::remove_if(
                descriptors.begin(), descriptors.end(),
                [&selected_serial](const DeviceDescriptor &descriptor) {
                    return descriptor.serial != selected_serial;
                }),
            descriptors.end());
    }
    update_single_camera_fallback(descriptors);
    return descriptors;
}

void CameraManager::reconcile()
{
    const auto now = std::chrono::steady_clock::now();
    if (!pending_restarts_.empty()) {
        std::vector<std::string> removals;
        for (const auto &entry : workers_) {
            if (pending_restarts_.count("*") != 0 ||
                pending_restarts_.count(entry.second->logical_name()) != 0) {
                removals.push_back(entry.first);
            }
        }
        for (const auto &key : removals) {
            remove_camera(key, "configuration changed");
            failures_.erase(key);
        }
        pending_restarts_.clear();
        next_discovery_ = std::chrono::steady_clock::time_point::min();
    }
    if (now < next_discovery_) {
        return;
    }
    next_discovery_ =
        now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(
                      positive(get_parameter("discovery_interval_sec"))));
    const auto descriptors = discover();
    std::map<std::string, DeviceDescriptor> discovered;
    std::map<std::string, std::string> topic_owners;
    for (const auto &descriptor : descriptors) {
        const auto logical = logical_name_for(descriptor.serial);
        const auto owner = topic_owners.find(logical);
        if (owner != topic_owners.end() && owner->second != descriptor.key()) {
            RCLCPP_ERROR(get_logger(), "camera topic prefix collision for %s",
                         logical.c_str());
            continue;
        }
        topic_owners[logical] = descriptor.key();
        discovered[descriptor.key()] = descriptor;
    }

    std::vector<std::string> removals;
    for (const auto &entry : workers_) {
        const auto found = discovered.find(entry.first);
        if (found == discovered.end()) {
            removals.push_back(entry.first);
        } else if (entry.second->logical_name() !=
                   logical_name_for(found->second.serial)) {
            removals.push_back(entry.first);
            failures_.erase(entry.first);
        } else if (entry.second->stale(now)) {
            removals.push_back(entry.first);
            failures_[entry.first] =
                Failure{ now + std::chrono::duration_cast<
                                   std::chrono::steady_clock::duration>(
                                   std::chrono::duration<double>(positive(
                                       get_parameter("retry_interval_sec")))),
                         {},
                         {} };
        }
    }
    for (const auto &key : removals) {
        remove_camera(key, discovered.count(key) == 0 ? "disconnected" :
                                                        "stream timed out");
    }
    for (const auto &entry : discovered) {
        if (workers_.count(entry.first) != 0) {
            continue;
        }
        const auto failure = failures_.find(entry.first);
        if (failure != failures_.end() && now < failure->second.retry_at) {
            continue;
        }
        start_camera(entry.second, now);
    }
    for (auto iterator = failures_.begin(); iterator != failures_.end();) {
        if (discovered.count(iterator->first) == 0 &&
            workers_.count(iterator->first) == 0) {
            iterator = failures_.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

void CameraManager::start_camera(const DeviceDescriptor &descriptor,
                                 std::chrono::steady_clock::time_point now)
{
    try {
        const auto logical = logical_name_for(descriptor.serial);
        const auto config = read_config(logical);
        std::unique_ptr<CameraDevice> worker =
            descriptor.backend == "realsense" ?
                make_realsense_camera(*this, descriptor, logical, config) :
                make_orbbec_camera(*this, descriptor, logical, config);
        workers_[descriptor.key()] = std::move(worker);
        failures_.erase(descriptor.key());
        RCLCPP_INFO(
            get_logger(),
            "camera online: backend=%s, serial=%s, logical_name=%s, topics=/%s/%s",
            descriptor.backend.c_str(), descriptor.serial.c_str(),
            logical.c_str(), config.camera_namespace.c_str(), logical.c_str());
    } catch (const std::exception &error) {
        const double retry = positive(get_parameter("retry_interval_sec"));
        auto &failure = failures_[descriptor.key()];
        const bool should_log =
            failure.message != error.what() ||
            failure.last_log.time_since_epoch().count() == 0 ||
            now - failure.last_log >= std::chrono::seconds(30);
        failure.retry_at =
            now +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(retry));
        failure.message = error.what();
        if (should_log) {
            failure.last_log = now;
            RCLCPP_ERROR(
                get_logger(),
                "camera start failed; retrying: backend=%s, serial=%s: %s",
                descriptor.backend.c_str(), descriptor.serial.c_str(),
                error.what());
        }
    }
}

void CameraManager::remove_camera(const std::string &key,
                                  const std::string &reason)
{
    auto found = workers_.find(key);
    if (found == workers_.end()) {
        return;
    }
    const auto descriptor = found->second->descriptor();
    found->second->stop();
    workers_.erase(found);
    if (rclcpp::ok()) {
        RCLCPP_WARN(get_logger(),
                    "camera offline: backend=%s, serial=%s, reason=%s",
                    descriptor.backend.c_str(), descriptor.serial.c_str(),
                    reason.c_str());
    }
}

void CameraManager::validate_parameter(const std::string &name,
                                       const rclcpp::Parameter &parameter)
{
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        return;
    }
    if (name == "single_camera_name") {
        camera_name_token(text(parameter));
    } else if (name == "serial_no") {
        text(parameter);
    } else if (name == "discovery_interval_sec" ||
               name == "retry_interval_sec" || name == "device_timeout_sec" ||
               name == "pointcloud.max_fps") {
        positive(parameter);
    }
}

rcl_interfaces::msg::SetParametersResult
CameraManager::on_parameters(const std::vector<rclcpp::Parameter> &parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    Proposed proposed;
    std::set<std::string> affected;
    try {
        for (const auto &parameter : parameters) {
            const auto &name = parameter.get_name();
            if (name == "use_sim_time") {
                continue;
            }
            std::string logical;
            std::string leaf;
            if (supported_config_parameter(name)) {
                if (parameter.get_type() ==
                    rclcpp::ParameterType::PARAMETER_NOT_SET) {
                    throw std::invalid_argument("global parameter " + name +
                                                " cannot be unset");
                }
                proposed[name] = parameter;
                affected.insert("*");
            } else if (camera_parameter_parts(name, logical, leaf)) {
                proposed[name] = parameter;
                affected.insert(leaf == "serial_no" ? "*" : logical);
            } else if (manager_defaults().count(name) != 0) {
                if (parameter.get_type() ==
                    rclcpp::ParameterType::PARAMETER_NOT_SET) {
                    throw std::invalid_argument("manager parameter " + name +
                                                " cannot be unset");
                }
                proposed[name] = parameter;
                if (name == "serial_no" || name == "single_camera_name") {
                    affected.insert("*");
                }
            } else {
                throw std::invalid_argument("unsupported parameter: " + name);
            }
            validate_parameter(leaf.empty() ? name : leaf, parameter);
        }
        read_config({}, &proposed);
        for (const auto &logical : camera_names(&proposed)) {
            read_config(logical, &proposed);
        }
        serial_mappings(&proposed);
        for (const auto &parameter : parameters) {
            if (parameter.get_name() == "single_camera_name") {
                camera_name_token(parameter.as_string());
            }
        }
    } catch (const std::exception &error) {
        result.reason = error.what();
        return result;
    }
    pending_restarts_.insert(affected.begin(), affected.end());
    if (affected.count("*") != 0) {
        single_unmapped_serial_.reset();
    }
    result.successful = true;
    return result;
}

void CameraManager::close() noexcept
{
    if (closed_) {
        return;
    }
    closed_ = true;
    timer_.reset();
    for (auto &entry : workers_) {
        entry.second->stop();
    }
    workers_.clear();
}

} // namespace bxi_depth_camera
