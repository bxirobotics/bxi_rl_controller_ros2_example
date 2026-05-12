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

std::vector<std::string> load_string_list(const YAML::Node &node)
{
    std::vector<std::string> values;
    if (!node) {
        return values;
    }
    if (node.IsScalar()) {
        values.push_back(node.as<std::string>());
        return values;
    }
    for (const auto &item : node) {
        values.push_back(item.as<std::string>());
    }
    return values;
}

std::string source_value_as_string(const YAML::Node &node, const std::string &prefix)
{
    if (node.IsScalar()) {
        try {
            return prefix + std::to_string(node.as<int>());
        } catch (const YAML::BadConversion &) {
            return node.as<std::string>();
        }
    }
    return get_or<std::string>(node, "source", "");
}

void add_source_alias(RemoteConfig &config, const std::string &group, const std::string &name, const std::string &source)
{
    if (source.empty()) {
        throw std::runtime_error("sources." + group + "." + name + " must contain a source");
    }
    config.source_aliases[group + "." + name] = source;
}

std::string resolve_source(const RemoteConfig &config, const std::string &source)
{
    const auto alias_it = config.source_aliases.find(source);
    if (alias_it == config.source_aliases.end()) {
        return source;
    }
    return alias_it->second;
}

void load_keyboard_source(const std::string &group, const YAML::Node &node, RemoteConfig &config)
{
    config.keyboard.poll_timeout_us = get_or<int>(
        node,
        "poll_timeout_us",
        get_or<int>(node, "timeout_us", config.keyboard.poll_timeout_us));
    config.keyboard.hold_ms = get_or<int>(node, "hold_ms", config.keyboard.hold_ms);

    const YAML::Node movement = node["movement"];
    config.keyboard.forward = key_from_name(get_or<std::string>(movement, "forward", "w"));
    config.keyboard.backward = key_from_name(get_or<std::string>(movement, "backward", "s"));
    config.keyboard.yaw_left = key_from_name(get_or<std::string>(movement, "yaw_left", "a"));
    config.keyboard.yaw_right = key_from_name(get_or<std::string>(movement, "yaw_right", "d"));
    config.keyboard.strafe_left = key_from_name(get_or<std::string>(movement, "strafe_left", "q"));
    config.keyboard.strafe_right = key_from_name(get_or<std::string>(movement, "strafe_right", "e"));
    config.keyboard.stop = key_from_name(get_or<std::string>(movement, "stop", "space"));

    const YAML::Node axes = node["axes"];
    if (axes) {
        if (!axes.IsMap()) {
            throw std::runtime_error("sources." + group + ".axes must be a YAML map");
        }
        for (const auto &item : axes) {
            add_source_alias(
                config,
                group,
                item.first.as<std::string>(),
                source_value_as_string(item.second, ""));
        }
    }

    const YAML::Node keys = node["keys"];
    if (!keys) {
        return;
    }
    if (!keys.IsMap()) {
        throw std::runtime_error("sources." + group + ".keys must be a YAML map");
    }
    for (const auto &item : keys) {
        const std::string name = item.first.as<std::string>();
        const YAML::Node key_node = item.second;
        std::string key_name;
        std::string signal;

        if (key_node.IsScalar()) {
            key_name = key_node.as<std::string>();
            signal = "key." + name;
        } else {
            key_name = get_or<std::string>(key_node, "key", "");
            signal = get_or<std::string>(key_node, "source", "key." + name);
        }

        const char key = key_from_name(key_name);
        if (key != '\0') {
            config.keyboard.signals_by_key[key] = {signal};
        }
        add_source_alias(config, group, name, signal);
    }
}

void load_source_declarations(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error("sources must be a YAML map");
    }

    for (const auto &group_item : node) {
        const std::string group = group_item.first.as<std::string>();
        const YAML::Node group_node = group_item.second;
        const std::string type = get_or<std::string>(group_node, "type", group);

        if (type == "joystick" || type == "gamepad") {
            config.js_device = get_or<std::string>(
                group_node,
                "device",
                get_or<std::string>(group_node, "js", config.js_device));

            const YAML::Node axes = group_node["axes"];
            if (axes) {
                for (const auto &item : axes) {
                    add_source_alias(
                        config,
                        group,
                        item.first.as<std::string>(),
                        source_value_as_string(item.second, "js.axis."));
                }
            }

            const YAML::Node buttons = group_node["buttons"];
            if (buttons) {
                for (const auto &item : buttons) {
                    add_source_alias(
                        config,
                        group,
                        item.first.as<std::string>(),
                        source_value_as_string(item.second, "js.button."));
                }
            }
        } else if (type == "keyboard") {
            load_keyboard_source(group, group_node, config);
        } else {
            const YAML::Node signals = group_node["signals"];
            if (!signals || !signals.IsMap()) {
                throw std::runtime_error(
                    "sources." + group + " with custom type must contain signals");
            }
            for (const auto &item : signals) {
                add_source_alias(
                    config,
                    group,
                    item.first.as<std::string>(),
                    source_value_as_string(item.second, ""));
            }
        }
    }
}

SignalSourceConfig load_signal_source(const YAML::Node &node)
{
    SignalSourceConfig source;
    if (node.IsScalar()) {
        source.source = node.as<std::string>();
        return source;
    }

    source.source = get_or<std::string>(node, "source", "");
    source.direction = get_or<double>(node, "direction", source.direction);
    source.scale = get_or<double>(node, "scale", source.scale);
    if (source.source.empty()) {
        throw std::runtime_error("control source must contain source");
    }
    return source;
}

std::vector<SignalSourceConfig> load_signal_sources(const YAML::Node &node)
{
    std::vector<SignalSourceConfig> sources;
    if (!node) {
        return sources;
    }

    if (node.IsScalar() || (node.IsMap() && node["source"])) {
        sources.push_back(load_signal_source(node));
        return sources;
    }

    for (const auto &item : node) {
        sources.push_back(load_signal_source(item));
    }
    return sources;
}

void load_controls(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        throw std::runtime_error("remote config must contain controls");
    }
    if (!node.IsMap()) {
        throw std::runtime_error("controls must be a YAML map");
    }

    for (const auto &item : node) {
        ControlConfig control;
        control.name = item.first.as<std::string>();
        const YAML::Node control_node = item.second;
        control.type = get_or<std::string>(control_node, "type", control.type);
        control.deadzone = get_or<double>(control_node, "deadzone", control.deadzone);
        control.min_value = get_or<double>(control_node, "min", control.min_value);
        control.max_value = get_or<double>(control_node, "max", control.max_value);
        control.alpha = get_or<double>(control_node, "alpha", control.alpha);
        control.threshold = get_or<double>(control_node, "threshold", control.threshold);
        control.hysteresis = get_or<double>(control_node, "hysteresis", control.hysteresis);

        if (control_node["sources"]) {
            control.sources = load_signal_sources(control_node["sources"]);
        } else if (control_node["source"]) {
            control.sources = load_signal_sources(control_node["source"]);
        }
        if (control.sources.size() == 1) {
            control.sources[0].direction = get_or<double>(
                control_node,
                "direction",
                control.sources[0].direction);
            control.sources[0].scale = get_or<double>(
                control_node,
                "scale",
                control.sources[0].scale);
        }
        for (auto &source : control.sources) {
            source.source = resolve_source(config, source.source);
        }

        const YAML::Node positions = control_node["positions"];
        if (positions) {
            if (!positions.IsMap()) {
                throw std::runtime_error("controls." + control.name + ".positions must be a YAML map");
            }
            for (const auto &position_item : positions) {
                EnumPositionConfig position;
                position.value = position_item.first.as<std::string>();
                const YAML::Node range = position_item.second;
                if (!range.IsSequence() || range.size() != 2) {
                    throw std::runtime_error(
                        "controls." + control.name + ".positions." + position.value +
                        " must be [min, max]");
                }
                position.min = range[0].as<double>();
                position.max = range[1].as<double>();
                control.positions.push_back(position);
            }
        }

        if (control.sources.empty()) {
            throw std::runtime_error("controls." + control.name + " must contain source or sources");
        }
        config.controls.push_back(control);
    }
}

void load_analog_outputs(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error("analog_outputs must be a YAML map");
    }

    for (const auto &item : node) {
        AnalogOutputConfig output;
        output.field = item.first.as<std::string>();
        const YAML::Node output_node = item.second;
        if (output_node.IsScalar()) {
            output.controls.push_back(output_node.as<std::string>());
        } else {
            if (output_node["control"]) {
                output.controls = load_string_list(output_node["control"]);
            }
            if (output_node["controls"]) {
                output.controls = load_string_list(output_node["controls"]);
            }
            output.offset = get_or<double>(output_node, "offset", output.offset);
        }
        if (output.field.empty() || output.controls.empty()) {
            throw std::runtime_error("analog_outputs entries must contain a control");
        }
        config.analog_outputs.push_back(output);
    }
}

BindingCondition load_condition(const YAML::Node &node)
{
    BindingCondition condition;
    if (node.IsScalar()) {
        const std::string text = node.as<std::string>();
        const auto eq_pos = text.find('=');
        if (eq_pos == std::string::npos) {
            condition.kind = "pressed";
            condition.control = text;
        } else {
            condition.kind = "equals";
            condition.control = text.substr(0, eq_pos);
            condition.value = text.substr(eq_pos + 1);
        }
        return condition;
    }

    if (!node.IsMap() || node.size() != 1) {
        throw std::runtime_error("binding condition must be scalar or single-key map");
    }

    const YAML::Node::const_iterator item = node.begin();
    condition.kind = item->first.as<std::string>();
    const YAML::Node value_node = item->second;

    if (condition.kind == "pressed" || condition.kind == "released") {
        condition.control = value_node.as<std::string>();
    } else if (condition.kind == "equals") {
        condition.control = get_or<std::string>(value_node, "control", "");
        condition.value = get_or<std::string>(value_node, "value", "");
    } else if (condition.kind == "range") {
        condition.control = get_or<std::string>(value_node, "control", "");
        condition.min = get_or<double>(value_node, "min", -1.0);
        condition.max = get_or<double>(value_node, "max", 1.0);
    } else {
        throw std::runtime_error("unknown binding condition kind: " + condition.kind);
    }

    if (condition.control.empty()) {
        throw std::runtime_error("binding condition must contain control");
    }
    return condition;
}

std::vector<BindingCondition> load_condition_list(const YAML::Node &node)
{
    std::vector<BindingCondition> conditions;
    if (!node) {
        return conditions;
    }

    if (node.IsScalar() || (node.IsMap() && !node["all"])) {
        conditions.push_back(load_condition(node));
        return conditions;
    }

    const YAML::Node list_node = node["all"] ? node["all"] : node;
    for (const auto &item : list_node) {
        conditions.push_back(load_condition(item));
    }
    return conditions;
}

std::vector<std::vector<BindingCondition>> load_condition_groups(const YAML::Node &node)
{
    std::vector<std::vector<BindingCondition>> groups;
    if (!node) {
        return groups;
    }

    if (node.IsMap() && node["any"]) {
        for (const auto &item : node["any"]) {
            groups.push_back(load_condition_list(item));
        }
    } else {
        groups.push_back(load_condition_list(node));
    }

    return groups;
}

void load_bindings(const YAML::Node &node, RemoteConfig &config, const std::string &default_mode)
{
    if (!node) {
        return;
    }
    if (!node.IsSequence()) {
        throw std::runtime_error("bindings must be a YAML list");
    }

    for (const auto &item : node) {
        Binding binding;
        binding.output = item["output"].as<std::string>();
        binding.mode = get_or<std::string>(item, "mode", default_mode);
        binding.condition_groups = load_condition_groups(item["when"]);
        if (binding.mode.empty()) {
            binding.mode = starts_with(binding.output, "system.") ? "edge" : "level";
        }
        if (binding.output.empty() || binding.condition_groups.empty()) {
            throw std::runtime_error("binding must contain output and when");
        }
        if (binding.mode != "level" && binding.mode != "edge") {
            throw std::runtime_error("binding mode must be level or edge");
        }
        config.bindings.push_back(binding);
    }
}

void load_outputs(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error("outputs must be a YAML map");
    }

    load_analog_outputs(node["analog"], config);
    load_bindings(node["level"], config, "level");
    load_bindings(node["edge"], config, "edge");
}

void load_keyboard(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }

    config.keyboard.poll_timeout_us = get_or<int>(
        node,
        "poll_timeout_us",
        get_or<int>(node, "timeout_us", config.keyboard.poll_timeout_us));
    config.keyboard.hold_ms = get_or<int>(node, "hold_ms", config.keyboard.hold_ms);

    const YAML::Node movement = node["movement"];
    config.keyboard.forward = key_from_name(get_or<std::string>(movement, "forward", "w"));
    config.keyboard.backward = key_from_name(get_or<std::string>(movement, "backward", "s"));
    config.keyboard.yaw_left = key_from_name(get_or<std::string>(movement, "yaw_left", "a"));
    config.keyboard.yaw_right = key_from_name(get_or<std::string>(movement, "yaw_right", "d"));
    config.keyboard.strafe_left = key_from_name(get_or<std::string>(movement, "strafe_left", "q"));
    config.keyboard.strafe_right = key_from_name(get_or<std::string>(movement, "strafe_right", "e"));
    config.keyboard.stop = key_from_name(get_or<std::string>(movement, "stop", "space"));

    const YAML::Node keys = node["keys"] ? node["keys"] : node["controls"];
    if (!keys) {
        return;
    }
    for (const auto &item : keys) {
        const char key = key_from_name(item.first.as<std::string>());
        if (key == '\0') {
            continue;
        }
        config.keyboard.signals_by_key[key] = load_string_list(item.second);
    }
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
    if (name == "esc" || name == "escape") {
        return '\x1b';
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

    load_source_declarations(root["sources"], config);
    load_keyboard(root["keyboard"], config);
    load_controls(root["controls"], config);
    load_outputs(root["outputs"], config);
    load_analog_outputs(root["analog_outputs"], config);
    load_bindings(root["bindings"], config, "");
    load_system_commands(root["system"], config);
    load_system_mutexes(root["system_mutexes"], config);
    for (const auto &action : load_string_list(root["system_reset_motion_after"])) {
        config.reset_motion_after_system.insert(action);
    }

    return config;
}

}  // namespace remote_controller
