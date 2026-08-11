#include "remote_controller/config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "remote_controller/motion_commands_adapter.hpp"

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

void require_map(const YAML::Node &node, const std::string &name)
{
    if (!node || !node.IsMap()) {
        throw std::runtime_error(name + " must be a YAML map");
    }
}

void require_sequence(const YAML::Node &node, const std::string &name)
{
    if (!node || !node.IsSequence()) {
        throw std::runtime_error(name + " must be a YAML list");
    }
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

std::string tail_name(const std::string &name)
{
    const auto dot_pos = name.rfind('.');
    if (dot_pos == std::string::npos) {
        return name;
    }
    return name.substr(dot_pos + 1);
}

std::string resolve_source(const RemoteConfig &config, const std::string &source)
{
    const auto alias_it = config.source_aliases.find(source);
    if (alias_it == config.source_aliases.end()) {
        return source;
    }
    return alias_it->second;
}

void set_source_alias(RemoteConfig &config, const std::string &signal, const std::string &from)
{
    if (signal.empty()) {
        throw std::runtime_error("source signal name must not be empty");
    }
    if (from.empty()) {
        throw std::runtime_error("source " + signal + " must contain from");
    }
    if (config.source_aliases.count(signal) > 0) {
        throw std::runtime_error("duplicate source signal: " + signal);
    }
    config.source_aliases[signal] = from;
}

void add_diagnostic(RemoteConfig &config, const std::string &severity, const std::string &message)
{
    ConfigDiagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.message = message;
    config.diagnostics.push_back(diagnostic);
}

bool is_valid_name(const std::string &name)
{
    if (name.empty()) {
        return false;
    }
    for (char ch : name) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!std::isalnum(c) && ch != '_' && ch != '-' && ch != '.') {
            return false;
        }
    }
    return true;
}

std::string join_ints(const std::set<int> &values)
{
    std::ostringstream stream;
    bool first = true;
    for (const int value : values) {
        if (!first) {
            stream << ", ";
        }
        stream << value;
        first = false;
    }
    return stream.str();
}

bool parse_int_text(const std::string &text, int &value)
{
    if (text.empty()) {
        return false;
    }

    std::size_t index = 0;
    if (text[index] == '-' || text[index] == '+') {
        ++index;
    }
    if (index >= text.size()) {
        return false;
    }
    for (; index < text.size(); ++index) {
        if (!std::isdigit(static_cast<unsigned char>(text[index]))) {
            return false;
        }
    }
    value = std::atoi(text.c_str());
    return true;
}

char required_key_from_name(const std::string &name, const std::string &field)
{
    const char key = key_from_name(name);
    if (key == '\0') {
        throw std::runtime_error(field + " contains unsupported key name: " + name);
    }
    return key;
}

char optional_key_from_name(const YAML::Node &node, const std::string &key, const std::string &field)
{
    if (!node || !node[key]) {
        return '\0';
    }
    return required_key_from_name(node[key].as<std::string>(), field);
}

void load_keyboard_signal(
    const std::string &signal,
    const YAML::Node &signal_node,
    RemoteConfig &config,
    KeyboardConfig &keyboard)
{
    const std::string from = get_or<std::string>(signal_node, "from", "");
    if (from == "keyboard.axis") {
        set_source_alias(config, signal, signal);

        const char negative = optional_key_from_name(
            signal_node,
            "negative",
            "keyboard axis source " + signal + ".negative");
        const char positive = optional_key_from_name(
            signal_node,
            "positive",
            "keyboard axis source " + signal + ".positive");
        if (negative == '\0' && positive == '\0') {
            throw std::runtime_error("keyboard axis source " + signal + " must contain negative or positive");
        }
        keyboard.hold_ms_by_signal[signal] = get_or<int>(
            signal_node,
            "hold_ms",
            keyboard.hold_ms);
        if (keyboard.hold_ms_by_signal[signal] < 0) {
            throw std::runtime_error("keyboard axis source " + signal + ".hold_ms must be >= 0");
        }

        const std::string role = tail_name(signal);
        if (role == "vx") {
            keyboard.vx_source = signal;
            if (negative != '\0') {
                keyboard.forward = negative;
            }
            if (positive != '\0') {
                keyboard.backward = positive;
            }
        } else if (role == "vy") {
            keyboard.vy_source = signal;
            if (negative != '\0') {
                keyboard.strafe_left = negative;
            }
            if (positive != '\0') {
                keyboard.strafe_right = positive;
            }
        } else if (role == "yaw") {
            keyboard.yaw_source = signal;
            if (negative != '\0') {
                keyboard.yaw_left = negative;
            }
            if (positive != '\0') {
                keyboard.yaw_right = positive;
            }
        } else {
            throw std::runtime_error(
                "keyboard axis source " + signal + " must end with vx, vy, or yaw");
        }
        return;
    }

    if (from == "keyboard.key") {
        if (!signal_node["key"]) {
            throw std::runtime_error("keyboard key source " + signal + " must contain key");
        }
        const char key = required_key_from_name(
            signal_node["key"].as<std::string>(),
            "keyboard key source " + signal + ".key");
        set_source_alias(config, signal, signal);
        keyboard.signals_by_key[key].push_back(signal);
        keyboard.hold_ms_by_signal[signal] = get_or<int>(
            signal_node,
            "hold_ms",
            keyboard.hold_ms);
        if (keyboard.hold_ms_by_signal[signal] < 0) {
            throw std::runtime_error("keyboard key source " + signal + ".hold_ms must be >= 0");
        }
        return;
    }

    throw std::runtime_error(
        "keyboard source " + signal + " has unsupported from: " + from);
}

void load_input_selection(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    require_map(node, "inputs");
    const YAML::Node selection = node["selection"] ? node["selection"] : node;
    require_map(selection, "inputs.selection");
    config.input_selection.scan_interval_ms = get_or<int>(
        selection,
        "scan_interval_ms",
        config.input_selection.scan_interval_ms);
    config.input_selection.promote_stable_ms = get_or<int>(
        selection,
        "promote_stable_ms",
        config.input_selection.promote_stable_ms);
    if (config.input_selection.scan_interval_ms <= 0) {
        throw std::runtime_error("inputs.selection.scan_interval_ms must be > 0");
    }
    if (config.input_selection.promote_stable_ms < 0) {
        throw std::runtime_error("inputs.selection.promote_stable_ms must be >= 0");
    }
}

void load_sources(const YAML::Node &node, RemoteConfig &config)
{
    require_map(node, "sources");

    for (const auto &group_item : node) {
        const std::string group = group_item.first.as<std::string>();
        const YAML::Node group_node = group_item.second;
        require_map(group_node, "sources." + group);

        const std::string type = get_or<std::string>(group_node, "type", "");
        if (type.empty()) {
            throw std::runtime_error("sources." + group + " must contain type");
        }

        InputDeviceConfig device;
        device.name = group;
        device.type = type;
        device.priority = get_or<int>(group_node, "priority", device.priority);
        device.ready_timeout_ms = get_or<int>(
            group_node,
            "ready_timeout_ms",
            device.ready_timeout_ms);
        device.loss_timeout_ms = get_or<int>(
            group_node,
            "loss_timeout_ms",
            device.loss_timeout_ms);
        device.cooldown_ms = get_or<int>(group_node, "cooldown_ms", device.cooldown_ms);
        if (device.ready_timeout_ms <= 0 || device.loss_timeout_ms < 0 || device.cooldown_ms < 0) {
            throw std::runtime_error(
                "sources." + group +
                " ready_timeout_ms must be > 0; loss_timeout_ms and cooldown_ms must be >= 0");
        }

        const std::set<std::string> generic_keys = {
            "type", "device", "js", "priority", "ready_timeout_ms", "loss_timeout_ms",
            "cooldown_ms", "signals", "poll_timeout_us", "hold_ms", "stop"};
        for (const auto &option_item : group_node) {
            const std::string option_name = option_item.first.as<std::string>();
            if (generic_keys.count(option_name) == 0 && option_item.second.IsScalar()) {
                device.options[option_name] = option_item.second.as<std::string>();
            }
        }

        if (type == "joystick" || type == "gamepad") {
            config.js_device = get_or<std::string>(
                group_node,
                "device",
                get_or<std::string>(group_node, "js", config.js_device));
            device.device = config.js_device;
        } else if (type == "keyboard") {
            device.keyboard.poll_timeout_us = get_or<int>(
                group_node,
                "poll_timeout_us",
                device.keyboard.poll_timeout_us);
            device.keyboard.hold_ms = get_or<int>(
                group_node,
                "hold_ms",
                device.keyboard.hold_ms);
            if (device.keyboard.poll_timeout_us < 0) {
                throw std::runtime_error("sources." + group + ".poll_timeout_us must be >= 0");
            }
            if (device.keyboard.hold_ms < 0) {
                throw std::runtime_error("sources." + group + ".hold_ms must be >= 0");
            }
            device.keyboard.stop = required_key_from_name(
                get_or<std::string>(group_node, "stop", "space"),
                "sources." + group + ".stop");
        } else {
            device.device = get_or<std::string>(group_node, "device", "");
        }

        const YAML::Node signals = group_node["signals"];
        require_map(signals, "sources." + group + ".signals");

        for (const auto &signal_item : signals) {
            const std::string signal = signal_item.first.as<std::string>();
            const YAML::Node signal_node = signal_item.second;
            require_map(signal_node, "sources." + group + ".signals." + signal);

            if (type == "keyboard") {
                load_keyboard_signal(signal, signal_node, config, device.keyboard);
                device.signals[signal] = signal;
                device.raw_sources.insert(signal);
                continue;
            }

            const std::string from = get_or<std::string>(signal_node, "from", "");
            if (from.empty()) {
                throw std::runtime_error(
                    "sources." + group + ".signals." + signal + " must contain from");
            }
            set_source_alias(config, signal, from);
            device.signals[signal] = from;
            device.raw_sources.insert(from);
            const int timeout_ms = get_or<int>(signal_node, "timeout_ms", 0);
            if (timeout_ms < 0) {
                throw std::runtime_error(
                    "sources." + group + ".signals." + signal + ".timeout_ms must be >= 0");
            }
            if (timeout_ms > 0) {
                SourceRuntimeConfig runtime;
                runtime.source = from;
                runtime.timeout_ms = timeout_ms;
                runtime.failsafe = get_or<double>(signal_node, "failsafe", 0.0);
                config.source_runtime[from] = runtime;
            }
        }

        if (type == "keyboard") {
            config.keyboard = device.keyboard;
        }
        config.input_devices.push_back(device);
    }
}

void load_curves(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    require_map(node, "curves");

    for (const auto &item : node) {
        CurveConfig curve;
        curve.name = item.first.as<std::string>();
        if (config.curves.count(curve.name) > 0) {
            throw std::runtime_error("duplicate curve: " + curve.name);
        }
        const YAML::Node curve_node = item.second;
        require_map(curve_node, "curves." + curve.name);
        curve.type = get_or<std::string>(curve_node, "type", curve.type);
        curve.deadzone = get_or<double>(curve_node, "deadzone", curve.deadzone);
        curve.expo = get_or<double>(curve_node, "expo", curve.expo);
        curve.min_value = get_or<double>(curve_node, "min", curve.min_value);
        curve.max_value = get_or<double>(curve_node, "max", curve.max_value);
        const YAML::Node limit = curve_node["limit"];
        if (limit) {
            if (!limit.IsSequence() || limit.size() != 2) {
                throw std::runtime_error("curves." + curve.name + ".limit must be [min, max]");
            }
            curve.min_value = limit[0].as<double>();
            curve.max_value = limit[1].as<double>();
        }
        const YAML::Node calibration = curve_node["calibration"];
        if (calibration) {
            require_map(calibration, "curves." + curve.name + ".calibration");
            curve.calibration.enabled = true;
            curve.calibration.clamp = get_or<bool>(calibration, "clamp", curve.calibration.clamp);
            const YAML::Node input = calibration["input"];
            if (input) {
                if (!input.IsSequence() || input.size() != 3) {
                    throw std::runtime_error("curves." + curve.name + ".calibration.input must be [min, center, max]");
                }
                curve.calibration.input_min = input[0].as<double>();
                curve.calibration.input_center = input[1].as<double>();
                curve.calibration.input_max = input[2].as<double>();
            } else {
                curve.calibration.input_min = get_or<double>(calibration, "input_min", curve.calibration.input_min);
                curve.calibration.input_center = get_or<double>(calibration, "input_center", curve.calibration.input_center);
                curve.calibration.input_max = get_or<double>(calibration, "input_max", curve.calibration.input_max);
            }
            const YAML::Node output = calibration["output"];
            if (output) {
                if (!output.IsSequence() || output.size() != 3) {
                    throw std::runtime_error("curves." + curve.name + ".calibration.output must be [min, center, max]");
                }
                curve.calibration.output_min = output[0].as<double>();
                curve.calibration.output_center = output[1].as<double>();
                curve.calibration.output_max = output[2].as<double>();
            } else {
                curve.calibration.output_min = get_or<double>(calibration, "output_min", curve.calibration.output_min);
                curve.calibration.output_center = get_or<double>(calibration, "output_center", curve.calibration.output_center);
                curve.calibration.output_max = get_or<double>(calibration, "output_max", curve.calibration.output_max);
            }
        }
        const YAML::Node points = curve_node["points"];
        if (points) {
            require_sequence(points, "curves." + curve.name + ".points");
            for (const auto &point_node : points) {
                if (!point_node.IsSequence() || point_node.size() != 2) {
                    throw std::runtime_error("curves." + curve.name + ".points[] must be [input, output]");
                }
                CurvePoint point;
                point.input = point_node[0].as<double>();
                point.output = point_node[1].as<double>();
                curve.points.push_back(point);
            }
        }
        if (curve.type != "expo" && curve.type != "piecewise") {
            throw std::runtime_error("curves." + curve.name + ".type must be expo or piecewise");
        }
        if (curve.deadzone < 0.0) {
            throw std::runtime_error("curves." + curve.name + ".deadzone must be >= 0");
        }
        if (curve.expo < 0.0 || curve.expo > 1.0) {
            throw std::runtime_error("curves." + curve.name + ".expo must be between 0 and 1");
        }
        if (curve.min_value > curve.max_value) {
            throw std::runtime_error("curves." + curve.name + " min must be <= max");
        }
        if (curve.calibration.enabled &&
            (curve.calibration.input_min >= curve.calibration.input_center ||
             curve.calibration.input_center >= curve.calibration.input_max)) {
            throw std::runtime_error(
                "curves." + curve.name + ".calibration.input must satisfy min < center < max");
        }
        for (std::size_t index = 1; index < curve.points.size(); ++index) {
            if (curve.points[index - 1].input >= curve.points[index].input) {
                throw std::runtime_error("curves." + curve.name + ".points must be sorted by input");
            }
        }
        if (curve.type == "piecewise" && curve.points.size() < 2) {
            throw std::runtime_error("curves." + curve.name + ".points must contain at least two points");
        }
        config.curves[curve.name] = curve;
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
    source.curve = get_or<std::string>(node, "curve", source.curve);
    source.direction = get_or<double>(node, "direction", source.direction);
    source.scale = get_or<double>(node, "scale", source.scale);
    source.offset = get_or<double>(node, "offset", source.offset);
    source.deadzone = get_or<double>(node, "deadzone", source.deadzone);
    source.expo = get_or<double>(node, "expo", source.expo);
    if (source.source.empty()) {
        throw std::runtime_error("control source must contain source");
    }
    return source;
}

ConditionConfig load_condition(
    const YAML::Node &node,
    const RemoteConfig &config,
    const std::string &path)
{
    if (!node) {
        throw std::runtime_error(path + " must not be empty");
    }

    if (node.IsScalar()) {
        const std::string text = node.as<std::string>();
        const auto eq_pos = text.find('=');
        ConditionConfig condition;
        if (eq_pos == std::string::npos) {
            condition.kind = "pressed";
            condition.control = text;
        } else {
            condition.kind = "equals";
            condition.control = text.substr(0, eq_pos);
            condition.value = text.substr(eq_pos + 1);
        }
        if (condition.control.empty()) {
            throw std::runtime_error(path + " contains an empty control name");
        }
        return condition;
    }

    if (node.IsSequence()) {
        ConditionConfig condition;
        condition.kind = "all";
        for (std::size_t index = 0; index < node.size(); ++index) {
            condition.children.push_back(
                load_condition(node[index], config, path + "[" + std::to_string(index) + "]"));
        }
        if (condition.children.empty()) {
            throw std::runtime_error(path + " must not be an empty condition list");
        }
        return condition;
    }

    if (!node.IsMap() || node.size() != 1) {
        throw std::runtime_error(path + " must be a condition scalar, list, or single-key map");
    }

    const YAML::Node::const_iterator item = node.begin();
    ConditionConfig condition;
    condition.kind = item->first.as<std::string>();
    const YAML::Node value_node = item->second;
    if (condition.kind == "pressed" || condition.kind == "released") {
        condition.control = value_node.as<std::string>();
    } else if (condition.kind == "equals") {
        require_map(value_node, path + ".equals");
        condition.control = get_or<std::string>(value_node, "control", "");
        condition.value = get_or<std::string>(value_node, "value", "");
    } else if (condition.kind == "range") {
        require_map(value_node, path + ".range");
        condition.control = get_or<std::string>(value_node, "control", "");
        condition.min = get_or<double>(value_node, "min", -1.0);
        condition.max = get_or<double>(value_node, "max", 1.0);
    } else if (condition.kind == "raw_range") {
        require_map(value_node, path + ".raw_range");
        condition.source = resolve_source(config, get_or<std::string>(value_node, "source", ""));
        condition.min = get_or<double>(value_node, "min", -1.0);
        condition.max = get_or<double>(value_node, "max", 1.0);
        if (condition.source.empty()) {
            throw std::runtime_error(path + ".raw_range must contain source");
        }
    } else if (condition.kind == "all" || condition.kind == "any") {
        require_sequence(value_node, path + "." + condition.kind);
        for (std::size_t index = 0; index < value_node.size(); ++index) {
            condition.children.push_back(load_condition(
                value_node[index],
                config,
                path + "." + condition.kind + "[" + std::to_string(index) + "]"));
        }
        if (condition.children.empty()) {
            throw std::runtime_error(path + "." + condition.kind + " must not be empty");
        }
    } else if (condition.kind == "not") {
        condition.children.push_back(load_condition(value_node, config, path + ".not"));
    } else {
        throw std::runtime_error(path + " has unsupported condition kind: " + condition.kind);
    }

    if ((condition.kind == "pressed" || condition.kind == "released" ||
         condition.kind == "equals" || condition.kind == "range") && condition.control.empty()) {
        throw std::runtime_error(path + "." + condition.kind + " must contain control");
    }
    if ((condition.kind == "range" || condition.kind == "raw_range") &&
        condition.min > condition.max) {
        throw std::runtime_error(path + "." + condition.kind + " min must be <= max");
    }
    return condition;
}

void load_control_input(
    const YAML::Node &node,
    const RemoteConfig &config,
    const ControlConfig &control,
    const std::string &path,
    ControlInputConfig &input)
{
    require_map(node, path);
    input.name = get_or<std::string>(node, "name", path);
    input.priority = get_or<int>(node, "priority", 0);

    const bool has_source = static_cast<bool>(node["source"]);
    const bool has_when = static_cast<bool>(node["when"]);
    const bool has_value = static_cast<bool>(node["value"]);
    if (has_source + has_when > 1 || has_source + has_value > 1) {
        throw std::runtime_error(path + " must contain exactly one of source, when, or value");
    }

    if (has_source) {
        input.kind = ControlInputKind::kSource;
        input.source = load_signal_source(node);
        input.source.source = resolve_source(config, input.source.source);
        return;
    }

    if (has_when) {
        if (!has_value) {
            throw std::runtime_error(path + ".when must contain value");
        }
        input.kind = ControlInputKind::kConditionalValue;
        input.when = load_condition(node["when"], config, path + ".when");
    } else if (has_value) {
        input.kind = ControlInputKind::kConstantValue;
    } else {
        throw std::runtime_error(path + " must contain source, when + value, or value");
    }

    const YAML::Node value = node["value"];
    if (control.type == "analog") {
        input.analog_value = value.as<double>();
    } else if (control.type == "bool") {
        input.bool_value = value.as<bool>();
    } else {
        input.enum_value = value.as<std::string>();
        if (input.enum_value.empty()) {
            throw std::runtime_error(path + ".value must not be empty for enum controls");
        }
    }
}

void load_controls(const YAML::Node &node, RemoteConfig &config)
{
    require_map(node, "controls");

    for (const auto &item : node) {
        ControlConfig control;
        control.name = item.first.as<std::string>();
        const YAML::Node control_node = item.second;
        require_map(control_node, "controls." + control.name);
        if (control_node["source"] || control_node["sources"] || control_node["expr"]) {
            throw std::runtime_error(
                "controls." + control.name +
                " uses removed source/sources/expr fields; use inputs instead");
        }

        control.type = get_or<std::string>(control_node, "type", control.type);
        if (control.type != "analog" && control.type != "bool" && control.type != "enum") {
            throw std::runtime_error(
                "controls." + control.name + ".type must be analog, bool, or enum");
        }
        const std::string default_mix = control.type == "analog" ? "max_abs" :
            (control.type == "bool" ? "any" : "first_active");
        control.mix = get_or<std::string>(control_node, "mix", default_mix);
        control.curve = get_or<std::string>(control_node, "curve", control.curve);
        control.deadzone = get_or<double>(control_node, "deadzone", control.deadzone);
        control.min_value = get_or<double>(control_node, "min", control.min_value);
        control.max_value = get_or<double>(control_node, "max", control.max_value);
        control.alpha = get_or<double>(control_node, "alpha", control.alpha);
        control.threshold = get_or<double>(control_node, "threshold", control.threshold);
        control.hysteresis = get_or<double>(control_node, "hysteresis", control.hysteresis);
        control.expo = get_or<double>(control_node, "expo", control.expo);
        control.invert = get_or<bool>(control_node, "invert", control.invert);

        if (control.type == "analog") {
            control.default_analog = get_or<double>(control_node, "default", control.default_analog);
        } else if (control.type == "bool") {
            control.default_bool = get_or<bool>(control_node, "default", control.default_bool);
        } else {
            control.default_enum = get_or<std::string>(control_node, "default", "");
        }

        const YAML::Node positions = control_node["positions"];
        if (positions) {
            require_map(positions, "controls." + control.name + ".positions");
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

        const YAML::Node inputs = control_node["inputs"];
        require_sequence(inputs, "controls." + control.name + ".inputs");
        if (inputs.size() == 0) {
            throw std::runtime_error("controls." + control.name + ".inputs must not be empty");
        }
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            ControlInputConfig input;
            load_control_input(
                inputs[index],
                config,
                control,
                "controls." + control.name + ".inputs[" + std::to_string(index) + "]",
                input);
            control.inputs.push_back(input);
        }
        config.controls.push_back(control);
    }
}

std::string normalize_output_field(const std::string &field)
{
    if (field == "vx") {
        return "vel_des.x";
    }
    if (field == "vy") {
        return "vel_des.y";
    }
    if (field == "vz") {
        return "vel_des.z";
    }
    if (field == "yaw") {
        return "yawdot_des";
    }
    if (field == "height") {
        return "height_des";
    }
    return field;
}

AnalogOutputConfig load_analog_output(
    const std::string &field,
    const YAML::Node &output_node,
    const std::string &path)
{
    AnalogOutputConfig output;
    output.field = normalize_output_field(field);
    if (output_node.IsScalar()) {
        output.controls.push_back(output_node.as<std::string>());
    } else {
        require_map(output_node, path);
        if (output_node["control"]) {
            output.controls = load_string_list(output_node["control"]);
        }
        if (output_node["controls"]) {
            output.controls = load_string_list(output_node["controls"]);
        }
        output.mix = get_or<std::string>(output_node, "mix", output.mix);
        output.scale = get_or<double>(output_node, "scale", output.scale);
        output.offset = get_or<double>(output_node, "offset", output.offset);
        output.min_value = get_or<double>(output_node, "min", output.min_value);
        output.max_value = get_or<double>(output_node, "max", output.max_value);
        const YAML::Node limit = output_node["limit"];
        if (limit) {
            if (!limit.IsSequence() || limit.size() != 2) {
                throw std::runtime_error(path + ".limit must be [min, max]");
            }
            output.min_value = limit[0].as<double>();
            output.max_value = limit[1].as<double>();
        }
    }
    if (output.field.empty() || output.controls.empty()) {
        throw std::runtime_error(path + " must contain a control");
    }
    return output;
}

void load_analog_outputs(const YAML::Node &node, RemoteConfig &config, const std::string &path)
{
    if (!node) {
        return;
    }
    require_map(node, path);

    for (const auto &item : node) {
        const std::string field = item.first.as<std::string>();
        config.analog_outputs.push_back(load_analog_output(field, item.second, path + "." + field));
    }
}

void load_bindings(const YAML::Node &node, RemoteConfig &config, const std::string &mode)
{
    if (!node) {
        return;
    }
    require_sequence(node, "outputs." + mode);

    for (const auto &item : node) {
        require_map(item, "outputs." + mode + "[]");
        Binding binding;
        if (!item["output"] || !item["when"]) {
            throw std::runtime_error("binding must contain output and when");
        }
        binding.output = item["output"].as<std::string>();
        binding.mode = mode;
        binding.when = load_condition(
            item["when"], config, "outputs." + mode + "[].when");
        if (binding.output.empty()) {
            throw std::runtime_error("binding must contain output and when");
        }
        config.bindings.push_back(binding);
    }
}

void load_outputs(const YAML::Node &node, RemoteConfig &config)
{
    require_map(node, "outputs");
    const std::set<std::string> allowed = {
        "conflict_policy",
        "publish_on_change",
        "analog",
        "level",
        "edge",
    };
    for (const auto &item : node) {
        const std::string key = item.first.as<std::string>();
        if (allowed.count(key) == 0) {
            throw std::runtime_error("outputs." + key + " is not supported; use outputs.analog");
        }
    }
    config.output_conflict_policy = get_or<std::string>(
        node,
        "conflict_policy",
        config.output_conflict_policy);
    if (config.output_conflict_policy != "last_wins" &&
        config.output_conflict_policy != "first_wins" &&
        config.output_conflict_policy != "error") {
        throw std::runtime_error("outputs.conflict_policy must be last_wins, first_wins, or error");
    }
    config.publish_on_change = get_or<bool>(
        node,
        "publish_on_change",
        config.publish_on_change);
    load_analog_outputs(node["analog"], config, "outputs.analog");
    load_bindings(node["level"], config, "level");
    load_bindings(node["edge"], config, "edge");
}

void load_system_commands(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    require_map(node, "system");

    for (const auto &item : node) {
        const std::string action = item.first.as<std::string>();
        const YAML::Node action_node = item.second;

        require_sequence(action_node, "system." + action);
        config.system_commands[action] = load_string_list(action_node);
    }
}

void load_system_mutexes(const YAML::Node &node, RemoteConfig &config)
{
    if (!node) {
        return;
    }
    require_map(node, "system_mutexes");

    for (const auto &item : node) {
        SystemMutexConfig mutex;
        mutex.name = item.first.as<std::string>();
        require_map(item.second, "system_mutexes." + mutex.name);
        mutex.acquire = get_or<std::string>(item.second, "acquire", "");
        mutex.release = get_or<std::string>(item.second, "release", "");
        if (mutex.name.empty() || mutex.acquire.empty() || mutex.release.empty()) {
            throw std::runtime_error("system_mutexes." + mutex.name + " must contain acquire and release");
        }
        config.system_mutexes.push_back(mutex);
    }
}

bool parse_btn_output(const std::string &output, int &slot, int &value)
{
    if (!starts_with(output, "btn_")) {
        return false;
    }
    const auto eq_pos = output.find('=');
    const std::string slot_text = output.substr(
        4,
        eq_pos == std::string::npos ? std::string::npos : eq_pos - 4);
    if (!parse_int_text(slot_text, slot)) {
        return false;
    }
    if (eq_pos == std::string::npos) {
        value = 1;
    } else if (!parse_int_text(output.substr(eq_pos + 1), value)) {
        return false;
    }
    return slot >= 1 && slot <= kButtonSlotCount;
}

void warn_unknown_root_keys(const YAML::Node &root, RemoteConfig &config)
{
    if (!root || !root.IsMap()) {
        return;
    }

    const std::set<std::string> allowed = {
        "inputs",
        "sources",
        "curves",
        "controls",
        "outputs",
        "system",
        "system_mutexes",
        "system_reset_motion_after",
    };
    for (const auto &item : root) {
        const std::string key = item.first.as<std::string>();
        if (allowed.count(key) == 0) {
            add_diagnostic(config, "warning", "unknown top-level config field: " + key);
        }
    }
}

void collect_condition_references(
    const ConditionConfig &condition,
    std::set<std::string> &controls,
    std::set<std::string> &raw_sources)
{
    if (condition.kind == "pressed" || condition.kind == "released" ||
        condition.kind == "equals" || condition.kind == "range") {
        controls.insert(condition.control);
    } else if (condition.kind == "raw_range") {
        raw_sources.insert(condition.source);
    }
    for (const auto &child : condition.children) {
        collect_condition_references(child, controls, raw_sources);
    }
}

void visit_control_dependency(
    const std::string &control,
    const std::map<std::string, std::set<std::string>> &dependencies,
    std::set<std::string> &visiting,
    std::set<std::string> &visited)
{
    if (visited.count(control) > 0) {
        return;
    }
    if (visiting.count(control) > 0) {
        throw std::runtime_error("derived control expression cycle detected at: " + control);
    }

    visiting.insert(control);
    const auto dependency_it = dependencies.find(control);
    if (dependency_it != dependencies.end()) {
        for (const auto &dependency : dependency_it->second) {
            if (dependencies.count(dependency) > 0) {
                visit_control_dependency(dependency, dependencies, visiting, visited);
            }
        }
    }
    visiting.erase(control);
    visited.insert(control);
}

void validate_config(RemoteConfig &config)
{
    std::set<std::string> semantic_sources;
    std::set<std::string> raw_sources;
    std::set<std::string> intentionally_exposed_sources;
    for (const auto &device : config.input_devices) {
        if (device.type != "crsf") {
            continue;
        }
        for (const auto &signal : device.signals) {
            intentionally_exposed_sources.insert(signal.first);
        }
    }
    for (const auto &item : config.source_aliases) {
        semantic_sources.insert(item.first);
        raw_sources.insert(item.second);
        if (!is_valid_name(item.first)) {
            add_diagnostic(config, "warning", "source name contains unusual characters: " + item.first);
        }
        if (!is_valid_name(item.second)) {
            add_diagnostic(config, "warning", "raw source name contains unusual characters: " + item.second);
        }
    }

    for (const auto &item : config.source_runtime) {
        const SourceRuntimeConfig &runtime = item.second;
        if (raw_sources.count(runtime.source) == 0) {
            throw std::runtime_error("runtime config references unknown source: " + runtime.source);
        }
        if (runtime.timeout_ms < 0) {
            throw std::runtime_error("source timeout_ms must be >= 0: " + runtime.source);
        }
        if (runtime.failsafe < -1.0 || runtime.failsafe > 1.0) {
            add_diagnostic(
                config,
                "warning",
                "source failsafe is outside normalized range [-1, 1]: " + runtime.source);
        }
    }

    std::set<std::string> used_raw_sources;
    std::set<std::string> used_curves;
    std::set<std::string> control_names;
    std::map<std::string, std::set<std::string>> condition_dependencies;
    for (const auto &control : config.controls) {
        if (!is_valid_name(control.name)) {
            add_diagnostic(config, "warning", "control name contains unusual characters: " + control.name);
        }
        if (control_names.count(control.name) > 0) {
            throw std::runtime_error("duplicate control name: " + control.name);
        }
        control_names.insert(control.name);

        const bool analog = control.type == "analog";
        const bool boolean = control.type == "bool";
        const bool enumeration = control.type == "enum";
        if (!analog && !boolean && !enumeration) {
            throw std::runtime_error("controls." + control.name + ".type must be analog, bool, or enum");
        }
        const bool valid_mix =
            (analog && (control.mix == "max_abs" || control.mix == "sum" ||
                        control.mix == "first_active")) ||
            (boolean && (control.mix == "any" || control.mix == "all" ||
                         control.mix == "first_active")) ||
            (enumeration && control.mix == "first_active");
        if (!valid_mix) {
            throw std::runtime_error(
                "controls." + control.name + " has an unsupported mix for type " + control.type);
        }
        if (control.deadzone < 0.0) {
            throw std::runtime_error("controls." + control.name + ".deadzone must be >= 0");
        }
        if (control.alpha < 0.0 || control.alpha > 1.0) {
            throw std::runtime_error("controls." + control.name + ".alpha must be between 0 and 1");
        }
        if (control.expo < 0.0 || control.expo > 1.0) {
            throw std::runtime_error("controls." + control.name + ".expo must be between 0 and 1");
        }
        if (control.hysteresis < 0.0) {
            throw std::runtime_error("controls." + control.name + ".hysteresis must be >= 0");
        }
        if (control.min_value > control.max_value) {
            throw std::runtime_error("controls." + control.name + " min must be <= max");
        }
        if (!control.curve.empty()) {
            used_curves.insert(control.curve);
        }
        bool enum_uses_source = false;
        for (const auto &input : control.inputs) {
            if (input.kind == ControlInputKind::kSource) {
                const SignalSourceConfig &source = input.source;
                enum_uses_source = enum_uses_source || enumeration;
                used_raw_sources.insert(source.source);
                if (!source.curve.empty()) {
                    used_curves.insert(source.curve);
                }
                if (raw_sources.count(source.source) == 0) {
                    throw std::runtime_error(
                        "controls." + control.name + " references unknown source: " + source.source);
                }
                if (source.deadzone < 0.0) {
                    throw std::runtime_error(
                        "controls." + control.name + " source " + source.source + " deadzone must be >= 0");
                }
                if (source.expo < 0.0 || source.expo > 1.0) {
                    throw std::runtime_error(
                        "controls." + control.name + " source " + source.source +
                        " expo must be between 0 and 1");
                }
            } else if (input.kind == ControlInputKind::kConditionalValue) {
                collect_condition_references(
                    input.when,
                    condition_dependencies[control.name],
                    used_raw_sources);
            }
        }

        if (enumeration) {
            if (control.default_enum.empty()) {
                throw std::runtime_error("enum control " + control.name + " must contain a non-empty default");
            }
            if (enum_uses_source && control.positions.empty()) {
                throw std::runtime_error("enum control " + control.name + " must contain positions");
            }
            std::vector<EnumPositionConfig> positions = control.positions;
            std::sort(
                positions.begin(),
                positions.end(),
                [](const EnumPositionConfig &lhs, const EnumPositionConfig &rhs) {
                    return lhs.min < rhs.min;
                });
            for (std::size_t index = 0; index < positions.size(); ++index) {
                if (positions[index].min > positions[index].max) {
                    throw std::runtime_error(
                        "enum control " + control.name + " position " +
                        positions[index].value + " has min > max");
                }
                if (index > 0 && positions[index].min < positions[index - 1].max) {
                    add_diagnostic(
                        config,
                        "warning",
                        "enum control " + control.name + " has overlapping positions: " +
                        positions[index - 1].value + " and " + positions[index].value);
                }
            }
        }
    }

    for (const auto &item : condition_dependencies) {
        for (const auto &dependency : item.second) {
            if (control_names.count(dependency) == 0) {
                throw std::runtime_error(
                    "controls." + item.first + ".inputs references unknown control: " + dependency);
            }
        }
    }
    std::set<std::string> visiting_condition_controls;
    std::set<std::string> visited_condition_controls;
    for (const auto &item : condition_dependencies) {
        visit_control_dependency(
            item.first,
            condition_dependencies,
            visiting_condition_controls,
            visited_condition_controls);
    }

    for (const auto &curve_name : used_curves) {
        if (config.curves.count(curve_name) == 0) {
            throw std::runtime_error("unknown curve referenced: " + curve_name);
        }
    }
    for (const auto &curve : config.curves) {
        if (used_curves.count(curve.first) == 0) {
            add_diagnostic(config, "warning", "curve is defined but not used: " + curve.first);
        }
    }

    std::set<std::string> used_controls;
    std::set<std::string> output_fields;
    for (const auto &output : config.analog_outputs) {
        if (!is_motion_command_field_supported(output.field)) {
            throw std::runtime_error("continuous output contains unknown MotionCommands field: " + output.field);
        }
        if (output_fields.count(output.field) > 0) {
            throw std::runtime_error("duplicate continuous output field: " + output.field);
        }
        output_fields.insert(output.field);
        if (output.mix != "max_abs" && output.mix != "sum" && output.mix != "first_active") {
            throw std::runtime_error("continuous output " + output.field + ".mix must be max_abs, sum, or first_active");
        }
        if (output.min_value > output.max_value) {
            throw std::runtime_error("continuous output " + output.field + " min must be <= max");
        }
        for (const auto &control : output.controls) {
            used_controls.insert(control);
            if (control_names.count(control) == 0) {
                throw std::runtime_error("continuous output " + output.field + " references unknown control: " + control);
            }
        }
    }

    std::map<int, std::set<int>> level_values_by_slot;
    std::map<int, std::set<int>> edge_values_by_slot;
    for (const auto &binding : config.bindings) {
        if (binding.mode != "level" && binding.mode != "edge") {
            throw std::runtime_error("binding mode must be level or edge");
        }
        if (binding.mode == "level") {
            int slot = 0;
            int value = 0;
            if (parse_btn_output(binding.output, slot, value)) {
                level_values_by_slot[slot].insert(value);
            } else if (starts_with(binding.output, "system.")) {
                throw std::runtime_error("system output must be under outputs.edge: " + binding.output);
            } else {
                throw std::runtime_error("unsupported outputs.level target: " + binding.output);
            }
        } else {
            int slot = 0;
            int value = 0;
            if (parse_btn_output(binding.output, slot, value)) {
                edge_values_by_slot[slot].insert(value);
            } else if (!starts_with(binding.output, "system.")) {
                throw std::runtime_error("unsupported outputs.edge target: " + binding.output);
            }
        }
        if (starts_with(binding.output, "system.")) {
            const std::string action = binding.output.substr(std::string("system.").size());
            if (config.system_commands.count(action) == 0) {
                throw std::runtime_error("output references unknown system action: " + binding.output);
            }
        }
        std::set<std::string> binding_controls;
        collect_condition_references(binding.when, binding_controls, used_raw_sources);
        for (const auto &control : binding_controls) {
            used_controls.insert(control);
            if (control_names.count(control) == 0) {
                throw std::runtime_error(
                    "binding " + binding.output + " references unknown control: " + control);
            }
        }
    }
    for (const auto &item : edge_values_by_slot) {
        if (item.second.size() > 1) {
            add_diagnostic(
                config,
                config.output_conflict_policy == "error" ? "warning" : "info",
                "edge btn_" + std::to_string(item.first) +
                " has multiple possible pulse values: " + join_ints(item.second));
        }
    }
    for (const auto &item : condition_dependencies) {
        for (const auto &dependency : item.second) {
            used_controls.insert(dependency);
        }
    }
    for (const auto &item : level_values_by_slot) {
        if (item.second.size() > 1) {
            add_diagnostic(
                config,
                config.output_conflict_policy == "error" ? "warning" : "info",
                "btn_" + std::to_string(item.first) +
                " has multiple possible level values: " + join_ints(item.second));
        }
    }

    for (const auto &mutex : config.system_mutexes) {
        if (config.system_commands.count(mutex.acquire) == 0) {
            throw std::runtime_error("system_mutexes." + mutex.name + ".acquire references unknown action");
        }
        if (config.system_commands.count(mutex.release) == 0) {
            throw std::runtime_error("system_mutexes." + mutex.name + ".release references unknown action");
        }
    }
    for (const auto &action : config.reset_motion_after_system) {
        if (config.system_commands.count(action) == 0) {
            throw std::runtime_error("system_reset_motion_after references unknown action: " + action);
        }
    }

    for (const auto &source : used_raw_sources) {
        if (raw_sources.count(source) == 0) {
            throw std::runtime_error("condition references unknown raw source: " + source);
        }
    }

    for (const auto &source : semantic_sources) {
        const std::string raw = config.source_aliases[source];
        if (used_raw_sources.count(raw) == 0 &&
            intentionally_exposed_sources.count(source) == 0) {
            add_diagnostic(config, "warning", "source is defined but not used by any control: " + source);
        }
    }
    for (const auto &control : control_names) {
        if (used_controls.count(control) == 0) {
            add_diagnostic(config, "warning", "control is defined but not used by any output: " + control);
        }
    }

    add_diagnostic(
        config,
        "info",
        "remote config loaded: " + std::to_string(config.input_devices.size()) +
        " input devices, " + std::to_string(config.source_aliases.size()) +
        " sources, " + std::to_string(config.controls.size()) +
        " controls, " + std::to_string(config.bindings.size()) +
        " bindings, " + std::to_string(config.analog_outputs.size()) +
        " continuous outputs");
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

    warn_unknown_root_keys(root, config);
    load_input_selection(root["inputs"], config);
    load_sources(root["sources"], config);
    load_curves(root["curves"], config);
    load_controls(root["controls"], config);
    load_outputs(root["outputs"], config);
    load_system_commands(root["system"], config);
    load_system_mutexes(root["system_mutexes"], config);
    for (const auto &action : load_string_list(root["system_reset_motion_after"])) {
        config.reset_motion_after_system.insert(action);
    }
    validate_config(config);

    return config;
}

}  // namespace remote_controller
