#include "remote_controller/config.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace remote_controller {
namespace {

template <typename T>
T get_or(const YAML::Node &node, const std::string &key, const T &fallback)
{
    if (!node || !node[key]) {
        return fallback;
    }
    return node[key].as<T>();
}

AxisConfig load_axis(const YAML::Node &node, const AxisConfig &fallback)
{
    AxisConfig axis = fallback;
    axis.index = get_or<int>(node, "index", axis.index);
    axis.direction = get_or<int>(node, "direction", axis.direction);
    axis.deadzone = get_or<double>(node, "deadzone", axis.deadzone);
    axis.min_value = get_or<double>(node, "min", axis.min_value);
    axis.max_value = get_or<double>(node, "max", axis.max_value);
    axis.alpha = get_or<double>(node, "alpha", axis.alpha);
    return axis;
}

std::vector<std::string> load_string_list(const YAML::Node &node)
{
    std::vector<std::string> values;
    if (!node) {
        return values;
    }
    for (const auto &item : node) {
        values.push_back(item.as<std::string>());
    }
    return values;
}

void load_system_commands(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error("system must be a YAML map");
    }

    for (const auto &item : node) {
        const std::string action = item.first.as<std::string>();
        const YAML::Node action_node = item.second;

        if (!action_node.IsSequence()) {
            throw std::runtime_error("system." + action + " must be a command list");
        }
        config.system_commands[action] = load_string_list(action_node);
    }
}

void load_system_mutexes(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error("system_mutexes must be a YAML map");
    }

    for (const auto &item : node) {
        SystemMutexConfig mutex;
        mutex.name = item.first.as<std::string>();
        if (!item.second.IsMap()) {
            throw std::runtime_error("system_mutexes." + mutex.name + " must be a YAML map");
        }
        mutex.acquire = get_or<std::string>(item.second, "acquire", "");
        mutex.release = get_or<std::string>(item.second, "release", "");
        if (mutex.name.empty() || mutex.acquire.empty() || mutex.release.empty()) {
            throw std::runtime_error("system_mutexes." + mutex.name + " must contain acquire and release");
        }
        config.system_mutexes.push_back(mutex);
    }
}

}  // namespace

bool starts_with(const std::string &value, const std::string &prefix)
{
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

char key_from_name(const std::string &name)
{
    if (name == "space") {
        return ' ';
    }
    if (name == "tab") {
        return '\t';
    }
    if (name.size() == 1) {
        return name[0];
    }
    return '\0';
}

RemoteConfig load_remote_config(const std::string &path)
{
    RemoteConfig config;
    const YAML::Node root = YAML::LoadFile(path);

    const YAML::Node device = root["device"];
    config.js_device = get_or<std::string>(device, "js", config.js_device);
    config.vel_offset = get_or<double>(device, "vel_offset", config.vel_offset);

    AxisConfig vx_default;
    vx_default.index = 3;
    vx_default.direction = -1;
    AxisConfig vy_default;
    vy_default.index = 0;
    vy_default.direction = -1;
    AxisConfig yaw_default;
    yaw_default.index = 6;
    yaw_default.direction = -1;
    yaw_default.alpha = 0.05;

    const YAML::Node axes = root["axes"];
    config.vx = load_axis(axes["vx"], vx_default);
    config.vy = load_axis(axes["vy"], vy_default);
    config.yaw = load_axis(axes["yaw"], yaw_default);

    const YAML::Node buttons = root["buttons"];
    for (const auto &item : buttons) {
        const std::string control_name = item.first.as<std::string>();
        const int index = item.second.as<int>();
        config.control_by_button_index[index] = control_name;
        config.control_down[control_name] = false;
    }

    for (const auto &modifier : load_string_list(root["modifiers"])) {
        config.modifiers.insert(modifier);
    }

    const YAML::Node axis_buttons = root["axis_buttons"];
    if (axis_buttons) {
        for (const auto &item : axis_buttons) {
            AxisButtonConfig axis_button;
            axis_button.control = item.first.as<std::string>();
            axis_button.index = get_or<int>(item.second, "index", axis_button.index);
            axis_button.direction = get_or<int>(item.second, "direction", axis_button.direction);
            axis_button.threshold = get_or<double>(item.second, "threshold", axis_button.threshold);
            axis_button.release_outputs = load_string_list(item.second["release_outputs"]);
            config.axis_buttons.push_back(axis_button);
            config.control_down[axis_button.control] = false;
        }
    }

    for (const auto &item : root["bindings"]) {
        Binding binding;
        binding.output = item["output"].as<std::string>();
        binding.controls = load_string_list(item["when"]);
        if (binding.output.empty() || binding.controls.empty()) {
            throw std::runtime_error("remote binding must contain output and when");
        }
        config.bindings.push_back(binding);
    }

    const YAML::Node keyboard = root["keyboard"];
    config.keyboard.timeout_us = get_or<int>(keyboard, "timeout_us", config.keyboard.timeout_us);
    const YAML::Node movement = keyboard["movement"];
    config.keyboard.forward = key_from_name(get_or<std::string>(movement, "forward", "w"));
    config.keyboard.backward = key_from_name(get_or<std::string>(movement, "backward", "s"));
    config.keyboard.yaw_left = key_from_name(get_or<std::string>(movement, "yaw_left", "a"));
    config.keyboard.yaw_right = key_from_name(get_or<std::string>(movement, "yaw_right", "d"));
    config.keyboard.strafe_left = key_from_name(get_or<std::string>(movement, "strafe_left", "q"));
    config.keyboard.strafe_right = key_from_name(get_or<std::string>(movement, "strafe_right", "e"));
    config.keyboard.stop = key_from_name(get_or<std::string>(movement, "stop", "space"));

    const YAML::Node keyboard_bindings = keyboard["bindings"];
    for (const auto &item : keyboard_bindings) {
        const char key = key_from_name(item.first.as<std::string>());
        const std::string output = item.second.as<std::string>();
        if (key != '\0') {
            config.keyboard.bindings[key] = output;
        }
    }

    load_system_commands(root["system"], config);
    load_system_mutexes(root["system_mutexes"], config);
    for (const auto &action : load_string_list(root["system_reset_motion_after"])) {
        config.reset_motion_after_system.insert(action);
    }

    return config;
}

}  // namespace remote_controller
