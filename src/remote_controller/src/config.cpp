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
    RemoteConfig &config)
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
        config.keyboard.hold_ms_by_signal[signal] = get_or<int>(
            signal_node,
            "hold_ms",
            config.keyboard.hold_ms);
        if (config.keyboard.hold_ms_by_signal[signal] < 0) {
            throw std::runtime_error("keyboard axis source " + signal + ".hold_ms must be >= 0");
        }

        const std::string role = tail_name(signal);
        if (role == "vx") {
            config.keyboard.vx_source = signal;
            if (negative != '\0') {
                config.keyboard.forward = negative;
            }
            if (positive != '\0') {
                config.keyboard.backward = positive;
            }
        } else if (role == "vy") {
            config.keyboard.vy_source = signal;
            if (negative != '\0') {
                config.keyboard.strafe_left = negative;
            }
            if (positive != '\0') {
                config.keyboard.strafe_right = positive;
            }
        } else if (role == "yaw") {
            config.keyboard.yaw_source = signal;
            if (negative != '\0') {
                config.keyboard.yaw_left = negative;
            }
            if (positive != '\0') {
                config.keyboard.yaw_right = positive;
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
        config.keyboard.signals_by_key[key].push_back(signal);
        config.keyboard.hold_ms_by_signal[signal] = get_or<int>(
            signal_node,
            "hold_ms",
            config.keyboard.hold_ms);
        if (config.keyboard.hold_ms_by_signal[signal] < 0) {
            throw std::runtime_error("keyboard key source " + signal + ".hold_ms must be >= 0");
        }
        return;
    }

    throw std::runtime_error(
        "keyboard source " + signal + " has unsupported from: " + from);
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

        if (type == "joystick" || type == "gamepad") {
            config.js_device = get_or<std::string>(
                group_node,
                "device",
                get_or<std::string>(group_node, "js", config.js_device));
        } else if (type == "keyboard") {
            config.keyboard.poll_timeout_us = get_or<int>(
                group_node,
                "poll_timeout_us",
                config.keyboard.poll_timeout_us);
            config.keyboard.hold_ms = get_or<int>(
                group_node,
                "hold_ms",
                config.keyboard.hold_ms);
            if (config.keyboard.poll_timeout_us < 0) {
                throw std::runtime_error("sources." + group + ".poll_timeout_us must be >= 0");
            }
            if (config.keyboard.hold_ms < 0) {
                throw std::runtime_error("sources." + group + ".hold_ms must be >= 0");
            }
            config.keyboard.stop = required_key_from_name(
                get_or<std::string>(group_node, "stop", "space"),
                "sources." + group + ".stop");
        }

        const YAML::Node signals = group_node["signals"];
        require_map(signals, "sources." + group + ".signals");

        for (const auto &signal_item : signals) {
            const std::string signal = signal_item.first.as<std::string>();
            const YAML::Node signal_node = signal_item.second;
            require_map(signal_node, "sources." + group + ".signals." + signal);

            if (type == "keyboard") {
                load_keyboard_signal(signal, signal_node, config);
                continue;
            }

            const std::string from = get_or<std::string>(signal_node, "from", "");
            if (from.empty()) {
                throw std::runtime_error(
                    "sources." + group + ".signals." + signal + " must contain from");
            }
            set_source_alias(config, signal, from);
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

std::vector<std::vector<BindingCondition>> load_condition_groups(const YAML::Node &node);

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
    require_map(node, "controls");

    for (const auto &item : node) {
        ControlConfig control;
        control.name = item.first.as<std::string>();
        const YAML::Node control_node = item.second;
        require_map(control_node, "controls." + control.name);

        control.type = get_or<std::string>(control_node, "type", control.type);
        control.mix = get_or<std::string>(control_node, "mix", control.mix);
        control.curve = get_or<std::string>(control_node, "curve", control.curve);
        control.default_value = get_or<std::string>(control_node, "default", control.default_value);
        control.deadzone = get_or<double>(control_node, "deadzone", control.deadzone);
        control.min_value = get_or<double>(control_node, "min", control.min_value);
        control.max_value = get_or<double>(control_node, "max", control.max_value);
        control.alpha = get_or<double>(control_node, "alpha", control.alpha);
        control.threshold = get_or<double>(control_node, "threshold", control.threshold);
        control.hysteresis = get_or<double>(control_node, "hysteresis", control.hysteresis);
        control.expo = get_or<double>(control_node, "expo", control.expo);
        control.invert = get_or<bool>(control_node, "invert", control.invert);

        if (control_node["expr"]) {
            control.expression_groups = load_condition_groups(control_node["expr"]);
        }
        if (control_node["sources"]) {
            control.sources = load_signal_sources(control_node["sources"]);
        } else if (control_node["source"]) {
            control.sources = load_signal_sources(control_node["source"]);
        }
        if (!control.expression_groups.empty() && !control.sources.empty()) {
            throw std::runtime_error("controls." + control.name + " cannot contain both expr and source/sources");
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

        if (control.sources.empty() && control.expression_groups.empty()) {
            throw std::runtime_error("controls." + control.name + " must contain source, sources, or expr");
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
        binding.condition_groups = load_condition_groups(item["when"]);
        if (binding.output.empty() || binding.condition_groups.empty()) {
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

void collect_condition_controls(
    const std::vector<std::vector<BindingCondition>> &groups,
    std::set<std::string> &controls)
{
    for (const auto &group : groups) {
        for (const auto &condition : group) {
            controls.insert(condition.control);
        }
    }
}

void visit_expression_control(
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
                visit_expression_control(dependency, dependencies, visiting, visited);
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
    std::map<std::string, std::set<std::string>> expression_dependencies;
    for (const auto &control : config.controls) {
        if (!is_valid_name(control.name)) {
            add_diagnostic(config, "warning", "control name contains unusual characters: " + control.name);
        }
        if (control_names.count(control.name) > 0) {
            throw std::runtime_error("duplicate control name: " + control.name);
        }
        control_names.insert(control.name);

        if (control.type != "analog" && control.type != "bool" && control.type != "enum") {
            throw std::runtime_error("controls." + control.name + ".type must be analog, bool, or enum");
        }
        if (!control.expression_groups.empty() && control.type != "bool") {
            throw std::runtime_error("controls." + control.name + ".expr is only supported for bool controls");
        }
        if (control.mix != "max_abs" && control.mix != "sum" && control.mix != "first_active") {
            throw std::runtime_error("controls." + control.name + ".mix must be max_abs, sum, or first_active");
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
        for (const auto &source : control.sources) {
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
                    "controls." + control.name + " source " + source.source + " expo must be between 0 and 1");
            }
        }
        if (!control.expression_groups.empty()) {
            collect_condition_controls(control.expression_groups, expression_dependencies[control.name]);
        }

        if (control.type == "enum") {
            if (control.positions.empty()) {
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
                if (index > 0 && positions[index].min > positions[index - 1].max &&
                    control.default_value.empty()) {
                    add_diagnostic(
                        config,
                        "warning",
                        "enum control " + control.name + " has an uncovered range and no default value");
                }
            }
        }
    }

    for (const auto &item : expression_dependencies) {
        for (const auto &dependency : item.second) {
            if (control_names.count(dependency) == 0) {
                throw std::runtime_error(
                    "controls." + item.first + ".expr references unknown control: " + dependency);
            }
        }
    }
    std::set<std::string> visiting_expression_controls;
    std::set<std::string> visited_expression_controls;
    for (const auto &item : expression_dependencies) {
        visit_expression_control(
            item.first,
            expression_dependencies,
            visiting_expression_controls,
            visited_expression_controls);
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
        for (const auto &group : binding.condition_groups) {
            for (const auto &condition : group) {
                used_controls.insert(condition.control);
                if (control_names.count(condition.control) == 0) {
                    throw std::runtime_error(
                        "binding " + binding.output + " references unknown control: " + condition.control);
                }
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
    for (const auto &item : expression_dependencies) {
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

    for (const auto &source : semantic_sources) {
        const std::string raw = config.source_aliases[source];
        if (used_raw_sources.count(raw) == 0) {
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
        "remote config loaded: " + std::to_string(config.source_aliases.size()) +
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
