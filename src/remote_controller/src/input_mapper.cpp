#include "remote_controller/input_mapper.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "remote_controller/motion_commands_adapter.hpp"

namespace remote_controller {
namespace {

std::string joystick_axis_source(int axis_index)
{
    return "js.axis." + std::to_string(axis_index);
}

std::string joystick_button_source(int button_index)
{
    return "js.button." + std::to_string(button_index);
}

bool is_axis_source(const std::string &source)
{
    return source.find(".axis.") != std::string::npos ||
           starts_with(source, "keyboard.axis.");
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

}  // namespace

InputMapper::InputMapper(RemoteConfig config)
    : config_(std::move(config)),
      binding_active_(config_.bindings.size(), false)
{
    refresh_controls();
    refresh_bindings();
}

const RemoteConfig &InputMapper::config() const
{
    return config_;
}

std::vector<std::string> InputMapper::set_axis(int axis_index, double value)
{
    if (axis_index < 0 || axis_index >= kAxisCount) {
        return {};
    }
    const double normalized = value / static_cast<double>(kAxisValueMax);
    return set_signal(joystick_axis_source(axis_index), normalized);
}

std::vector<std::string> InputMapper::set_signal(const std::string &source, double value)
{
    signals_[source] = value;
    signal_expiry_.erase(source);
    signal_update_time_[source] = std::chrono::steady_clock::now();
    timed_out_sources_.erase(source);
    return refresh_bindings();
}

std::vector<std::string> InputMapper::set_signals(
    const std::vector<std::pair<std::string, double>> &signals)
{
    if (signals.empty()) {
        return {};
    }

    const auto now = std::chrono::steady_clock::now();
    for (const auto &signal : signals) {
        signals_[signal.first] = signal.second;
        signal_expiry_.erase(signal.first);
        signal_update_time_[signal.first] = now;
        timed_out_sources_.erase(signal.first);
    }
    return refresh_bindings();
}

void InputMapper::set_input_edges_enabled(bool enabled)
{
    input_edges_enabled_ = enabled;
    if (!enabled) {
        std::fill(edge_pulse_slots_, edge_pulse_slots_ + kButtonSlotCount + 1, 0);
    }
}

void InputMapper::set_debug_enabled(bool enabled)
{
    debug_enabled_ = enabled;
    debug_messages_.clear();
    if (!debug_enabled_) {
        return;
    }
    for (auto &control : controls_) {
        control.second.debug_trace.clear();
    }
    refresh_controls();
}

std::vector<std::string> InputMapper::take_debug_messages()
{
    std::vector<std::string> messages;
    messages.swap(debug_messages_);
    return messages;
}

void InputMapper::touch_runtime_sources_with_prefix(const std::string &prefix)
{
    const auto now = std::chrono::steady_clock::now();
    for (const auto &item : config_.source_runtime) {
        const std::string &source = item.second.source;
        if (!starts_with(source, prefix)) {
            continue;
        }
        signal_update_time_[source] = now;
        timed_out_sources_.erase(source);
    }
}

void InputMapper::touch_runtime_sources(const std::set<std::string> &sources)
{
    const auto now = std::chrono::steady_clock::now();
    for (const auto &item : config_.source_runtime) {
        const std::string &source = item.second.source;
        if (sources.count(source) > 0) {
            signal_update_time_[source] = now;
            timed_out_sources_.erase(source);
        }
    }
}

void InputMapper::clear_signals_with_prefix(const std::string &prefix)
{
    if (prefix.empty()) {
        return;
    }

    std::set<std::string> cleared_sources;
    for (auto &item : signals_) {
        if (starts_with(item.first, prefix)) {
            item.second = 0.0;
            cleared_sources.insert(item.first);
        }
    }

    for (const auto &item : config_.source_runtime) {
        const std::string &source = item.second.source;
        if (starts_with(source, prefix)) {
            signals_[source] = 0.0;
            cleared_sources.insert(source);
        }
    }

    for (auto it = signal_expiry_.begin(); it != signal_expiry_.end();) {
        if (starts_with(it->first, prefix)) {
            it = signal_expiry_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = signal_update_time_.begin(); it != signal_update_time_.end();) {
        if (starts_with(it->first, prefix)) {
            it = signal_update_time_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = timed_out_sources_.begin(); it != timed_out_sources_.end();) {
        if (starts_with(*it, prefix)) {
            it = timed_out_sources_.erase(it);
        } else {
            ++it;
        }
    }

    if (cleared_sources.empty()) {
        return;
    }

    for (auto &item : controls_) {
        const ControlConfig *control = find_control_config(item.first);
        if (control == nullptr) {
            continue;
        }
        for (const auto &input : control->inputs) {
            if (input.kind == ControlInputKind::kSource &&
                (cleared_sources.count(input.source.source) > 0 ||
                 starts_with(input.source.source, prefix))) {
                item.second = ControlValue();
                break;
            }
        }
    }

    refresh_bindings(false);
}

void InputMapper::clear_signals(const std::set<std::string> &sources)
{
    if (sources.empty()) {
        return;
    }

    for (const auto &source : sources) {
        signals_[source] = 0.0;
        signal_expiry_.erase(source);
        signal_update_time_.erase(source);
        timed_out_sources_.erase(source);
    }

    for (auto &item : controls_) {
        const ControlConfig *control = find_control_config(item.first);
        if (control == nullptr) {
            continue;
        }
        for (const auto &input : control->inputs) {
            if (input.kind == ControlInputKind::kSource &&
                sources.count(input.source.source) > 0) {
                item.second = ControlValue();
                break;
            }
        }
    }

    refresh_bindings(false);
}

void InputMapper::zero_motion_axes()
{
    signals_[config_.keyboard.vx_source] = 0.0;
    signals_[config_.keyboard.vy_source] = 0.0;
    signals_[config_.keyboard.yaw_source] = 0.0;
    signal_expiry_.erase(config_.keyboard.vx_source);
    signal_expiry_.erase(config_.keyboard.vy_source);
    signal_expiry_.erase(config_.keyboard.yaw_source);
    refresh_bindings();
}

void InputMapper::reset_motion()
{
    std::set<std::string> motion_sources;
    for (const auto &output : config_.analog_outputs) {
        for (const auto &output_control : output.controls) {
            for (const auto &control : config_.controls) {
                if (control.name != output_control) {
                    continue;
                }
                for (const auto &input : control.inputs) {
                    if (input.kind == ControlInputKind::kSource) {
                        motion_sources.insert(input.source.source);
                    }
                }
            }
        }
    }

    for (auto &item : signals_) {
        if (is_axis_source(item.first) || motion_sources.count(item.first) > 0) {
            item.second = 0.0;
        }
    }
    for (const auto &source : motion_sources) {
        signals_[source] = 0.0;
    }
    signals_[config_.keyboard.vx_source] = 0.0;
    signals_[config_.keyboard.vy_source] = 0.0;
    signals_[config_.keyboard.yaw_source] = 0.0;
    for (auto it = signal_expiry_.begin(); it != signal_expiry_.end();) {
        if (is_axis_source(it->first) || motion_sources.count(it->first) > 0) {
            it = signal_expiry_.erase(it);
        } else {
            ++it;
        }
    }
    signal_expiry_.erase(config_.keyboard.vx_source);
    signal_expiry_.erase(config_.keyboard.vy_source);
    signal_expiry_.erase(config_.keyboard.yaw_source);
    for (auto &control : controls_) {
        control.second.analog = 0.0;
    }
    std::fill(output_slots_, output_slots_ + kButtonSlotCount + 1, 0);
    std::fill(edge_pulse_slots_, edge_pulse_slots_ + kButtonSlotCount + 1, 0);
    height_filtered_ = kStandHeight;
    refresh_bindings();
}

std::vector<std::string> InputMapper::tick()
{
    const auto now = std::chrono::steady_clock::now();
    bool changed = false;
    for (auto it = signal_expiry_.begin(); it != signal_expiry_.end();) {
        if (it->second <= now) {
            signals_[it->first] = 0.0;
            it = signal_expiry_.erase(it);
            changed = true;
        } else {
            ++it;
        }
    }
    for (const auto &item : config_.source_runtime) {
        const SourceRuntimeConfig &runtime = item.second;
        if (runtime.timeout_ms <= 0 || timed_out_sources_.count(runtime.source) > 0) {
            continue;
        }
        const auto update_it = signal_update_time_.find(runtime.source);
        if (update_it == signal_update_time_.end()) {
            continue;
        }
        const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - update_it->second);
        if (age.count() >= runtime.timeout_ms) {
            signals_[runtime.source] = runtime.failsafe;
            timed_out_sources_.insert(runtime.source);
            changed = true;
        }
    }

    if (!changed) {
        return {};
    }
    return refresh_bindings();
}

std::vector<std::string> InputMapper::handle_button(int button_index, bool pressed)
{
    return set_signal(joystick_button_source(button_index), pressed ? 1.0 : 0.0);
}

std::vector<std::string> InputMapper::handle_keyboard_key(char key)
{
    const auto now = std::chrono::steady_clock::now();
    const auto expiry_for_signal = [this, now](const std::string &signal) {
        const auto hold_it = config_.keyboard.hold_ms_by_signal.find(signal);
        const int hold_ms = hold_it == config_.keyboard.hold_ms_by_signal.end() ?
            config_.keyboard.hold_ms :
            hold_it->second;
        return now + std::chrono::milliseconds(hold_ms);
    };

    if (key == config_.keyboard.forward) {
        signals_[config_.keyboard.vx_source] = -1.0;
        signal_update_time_[config_.keyboard.vx_source] = now;
        signal_expiry_[config_.keyboard.vx_source] = expiry_for_signal(config_.keyboard.vx_source);
        return refresh_bindings();
    }
    if (key == config_.keyboard.backward) {
        signals_[config_.keyboard.vx_source] = 1.0;
        signal_update_time_[config_.keyboard.vx_source] = now;
        signal_expiry_[config_.keyboard.vx_source] = expiry_for_signal(config_.keyboard.vx_source);
        return refresh_bindings();
    }
    if (key == config_.keyboard.yaw_left) {
        signals_[config_.keyboard.yaw_source] = -1.0;
        signal_update_time_[config_.keyboard.yaw_source] = now;
        signal_expiry_[config_.keyboard.yaw_source] = expiry_for_signal(config_.keyboard.yaw_source);
        return refresh_bindings();
    }
    if (key == config_.keyboard.yaw_right) {
        signals_[config_.keyboard.yaw_source] = 1.0;
        signal_update_time_[config_.keyboard.yaw_source] = now;
        signal_expiry_[config_.keyboard.yaw_source] = expiry_for_signal(config_.keyboard.yaw_source);
        return refresh_bindings();
    }
    if (key == config_.keyboard.strafe_left) {
        signals_[config_.keyboard.vy_source] = -1.0;
        signal_update_time_[config_.keyboard.vy_source] = now;
        signal_expiry_[config_.keyboard.vy_source] = expiry_for_signal(config_.keyboard.vy_source);
        return refresh_bindings();
    }
    if (key == config_.keyboard.strafe_right) {
        signals_[config_.keyboard.vy_source] = 1.0;
        signal_update_time_[config_.keyboard.vy_source] = now;
        signal_expiry_[config_.keyboard.vy_source] = expiry_for_signal(config_.keyboard.vy_source);
        return refresh_bindings();
    }
    if (key == config_.keyboard.stop) {
        zero_motion_axes();
        return {};
    }

    const auto signal_it = config_.keyboard.signals_by_key.find(key);
    if (signal_it == config_.keyboard.signals_by_key.end()) {
        return {};
    }

    for (const auto &signal : signal_it->second) {
        signals_[signal] = 1.0;
        signal_update_time_[signal] = now;
        timed_out_sources_.erase(signal);
        signal_expiry_[signal] = expiry_for_signal(signal);
    }
    return refresh_bindings();
}

void InputMapper::fill_message(communication::msg::MotionCommands &message)
{
    refresh_controls();

    bool height_written = false;
    for (const auto &output : config_.analog_outputs) {
        const double value = read_analog_output(output);
        if (set_motion_command_field(message, output.field, value) && output.field == "height_des") {
            height_written = true;
        }
    }

    for (int slot = 1; slot <= kButtonSlotCount; ++slot) {
        const int value = edge_pulse_slots_[slot] != 0 ? edge_pulse_slots_[slot] : output_slots_[slot];
        set_motion_command_button_slot(message, slot, value);
        edge_pulse_slots_[slot] = 0;
    }
    log_button_output_changes();

    if (!height_written) {
        height_filtered_ = height_filtered_ * 0.9 + kStandHeight * 0.1;
        message.height_des = static_cast<float>(height_filtered_);
    }
}

std::vector<std::string> InputMapper::refresh_bindings(bool emit_edges)
{
    refresh_controls();
    std::fill(output_slots_, output_slots_ + kButtonSlotCount + 1, 0);

    std::vector<std::string> edge_outputs;
    for (std::size_t index = 0; index < config_.bindings.size(); ++index) {
        const Binding &binding = config_.bindings[index];
        const bool active = conditions_match(binding);

        if (binding.mode == "edge") {
            if (emit_edges && input_edges_enabled_ && active && !binding_active_[index]) {
                if (!apply_edge_pulse_output(binding.output)) {
                    edge_outputs.push_back(binding.output);
                }
            }
        } else if (active) {
            apply_level_output(binding.output);
        }

        binding_active_[index] = active;
    }
    log_button_output_changes();

    return edge_outputs;
}

void InputMapper::log_button_output_changes()
{
    std::string line = "remote btn changed:";
    bool changed = false;
    for (int slot = 1; slot <= kButtonSlotCount; ++slot) {
        const int value = edge_pulse_slots_[slot] != 0 ? edge_pulse_slots_[slot] : output_slots_[slot];
        if (value == last_logged_output_slots_[slot]) {
            continue;
        }

        changed = true;
        line += " btn_" + std::to_string(slot) +
                ": " + std::to_string(last_logged_output_slots_[slot]) +
                "->" + std::to_string(value);
        last_logged_output_slots_[slot] = value;
    }

    if (changed) {
        std::cout << line << std::endl;
    }
}

void InputMapper::refresh_controls()
{
    std::set<std::string> visiting;
    std::set<std::string> evaluated;
    for (const auto &control : config_.controls) {
        evaluate_control_recursive(control.name, visiting, evaluated);
    }
}

void InputMapper::evaluate_control_recursive(
    const std::string &control,
    std::set<std::string> &visiting,
    std::set<std::string> &evaluated)
{
    if (evaluated.count(control) > 0) {
        return;
    }
    if (visiting.count(control) > 0) {
        throw std::runtime_error("control condition cycle detected at: " + control);
    }

    const ControlConfig *config = find_control_config(control);
    if (config == nullptr) {
        return;
    }

    visiting.insert(control);
    for (const auto &input : config->inputs) {
        if (input.kind == ControlInputKind::kConditionalValue) {
            evaluate_condition_dependencies(input.when, visiting, evaluated);
        }
    }
    controls_[control] = evaluate_control(*config);
    visiting.erase(control);
    evaluated.insert(control);
}

const ControlConfig *InputMapper::find_control_config(const std::string &control) const
{
    for (const auto &item : config_.controls) {
        if (item.name == control) {
            return &item;
        }
    }
    return nullptr;
}

void InputMapper::evaluate_condition_dependencies(
    const ConditionConfig &condition,
    std::set<std::string> &visiting,
    std::set<std::string> &evaluated)
{
    if (condition.kind == "pressed" || condition.kind == "released" ||
        condition.kind == "equals" || condition.kind == "range") {
        evaluate_control_recursive(condition.control, visiting, evaluated);
    }
    for (const auto &child : condition.children) {
        evaluate_condition_dependencies(child, visiting, evaluated);
    }
}

InputMapper::ControlValue InputMapper::evaluate_control(const ControlConfig &control)
{
    const ControlValue previous = controls_[control.name];
    ControlValue value;
    std::vector<InputCandidate> all_candidates;
    std::vector<InputCandidate> candidates;
    int selected_priority = std::numeric_limits<int>::min();
    for (const auto &input : control.inputs) {
        InputCandidate candidate = evaluate_input(control, input, previous);
        all_candidates.push_back(candidate);
        if (!candidate.active) {
            continue;
        }
        if (input.priority > selected_priority) {
            selected_priority = input.priority;
        }
    }
    if (selected_priority != std::numeric_limits<int>::min()) {
        for (const auto &candidate : all_candidates) {
            if (candidate.config->priority != selected_priority) {
                continue;
            }
            if (candidate.active ||
                (control.type == "bool" && control.mix == "all" &&
                 candidate.config->kind == ControlInputKind::kSource)) {
                candidates.push_back(candidate);
            }
        }
    }

    std::ostringstream trace;
    trace << control.name << " priority=";
    if (candidates.empty()) {
        trace << "default";
    } else {
        trace << selected_priority;
    }
    trace << " active=[";
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (index > 0) {
            trace << ", ";
        }
        trace << candidates[index].config->name << "=";
        if (control.type == "analog") {
            trace << candidates[index].analog;
        } else if (control.type == "bool") {
            trace << (candidates[index].boolean ? "true" : "false");
        } else {
            trace << candidates[index].enumeration;
        }
    }
    trace << "] ";

    if (control.type == "analog") {
        double analog = control.default_analog;
        if (!candidates.empty()) {
            analog = 0.0;
            if (control.mix == "sum") {
                for (const auto &candidate : candidates) {
                    analog += candidate.analog;
                }
            } else if (control.mix == "first_active") {
                analog = candidates.front().analog;
            } else {
                for (const auto &candidate : candidates) {
                    if (std::fabs(candidate.analog) > std::fabs(analog)) {
                        analog = candidate.analog;
                    }
                }
            }
        }
        if (control.invert) {
            analog = -analog;
        }
        analog = apply_curve(analog, control.curve);
        analog = apply_expo(analog, control.expo);
        if (std::fabs(analog) <= control.deadzone) {
            analog = 0.0;
        }
        analog = std::max(control.min_value, std::min(control.max_value, analog));
        value.analog = analog * control.alpha + previous.analog * (1.0 - control.alpha);
        // The input deadzone above turns a centered axis into an exact zero
        // target, but the low-pass filter otherwise only approaches zero
        // asymptotically.  Snap the decaying filter state to zero once it is
        // inside the configured deadzone.  Only do this for a zero target so
        // a small valid input can still build up through a low alpha filter.
        if (analog == 0.0 && std::fabs(value.analog) <= control.deadzone) {
            value.analog = 0.0;
        }
        value.pressed = std::fabs(value.analog) > 0.0;
        trace << "value=" << value.analog;
        value.debug_trace = trace.str();
        record_debug_trace(control.name, value);
        return value;
    }

    if (control.type == "enum") {
        value.value = control.default_enum;
        if (!candidates.empty()) {
            value.value = candidates.front().enumeration;
            value.analog = candidates.front().analog;
        }
        value.pressed = !candidates.empty() && value.value != control.default_enum;
        trace << "value=" << value.value;
        value.debug_trace = trace.str();
        record_debug_trace(control.name, value);
        return value;
    }

    bool pressed = control.default_bool;
    if (!candidates.empty()) {
        if (control.mix == "all") {
            pressed = true;
            for (const auto &candidate : candidates) {
                pressed = pressed && candidate.boolean;
            }
        } else if (control.mix == "first_active") {
            pressed = candidates.front().boolean;
        } else {
            pressed = false;
            for (const auto &candidate : candidates) {
                pressed = pressed || candidate.boolean;
            }
        }
    }
    value.pressed = pressed;
    value.analog = pressed ? 1.0 : 0.0;
    value.value = pressed ? "true" : "false";
    trace << "value=" << value.value;
    value.debug_trace = trace.str();
    record_debug_trace(control.name, value);
    return value;
}

InputMapper::InputCandidate InputMapper::evaluate_input(
    const ControlConfig &control,
    const ControlInputConfig &input,
    const ControlValue &previous) const
{
    InputCandidate candidate;
    candidate.config = &input;
    if (input.kind == ControlInputKind::kSource) {
        candidate.analog = read_source(input.source);
        candidate.active = std::fabs(candidate.analog) > 0.0;
        if (control.type == "bool") {
            const double threshold = previous.pressed ?
                control.threshold - control.hysteresis :
                control.threshold + control.hysteresis;
            candidate.boolean = candidate.analog >= threshold;
        } else if (control.type == "enum") {
            candidate.enumeration = enum_from_raw(control, candidate.analog, previous.value);
        }
        return candidate;
    }

    candidate.active = input.kind == ControlInputKind::kConstantValue ||
        condition_matches(input.when);
    if (control.type == "analog") {
        candidate.analog = input.analog_value;
    } else if (control.type == "bool") {
        candidate.boolean = input.bool_value;
    } else {
        candidate.enumeration = input.enum_value;
    }
    return candidate;
}

double InputMapper::read_source(const SignalSourceConfig &source) const
{
    const auto signal_it = signals_.find(source.source);
    double value = signal_it == signals_.end() ? 0.0 : signal_it->second;
    value = value * source.direction * source.scale + source.offset;
    value = apply_curve(value, source.curve);
    if (std::fabs(value) <= source.deadzone) {
        value = 0.0;
    }
    return apply_expo(value, source.expo);
}

std::string InputMapper::enum_from_raw(
    const ControlConfig &control,
    double raw,
    const std::string &previous) const
{
    for (const auto &position : control.positions) {
        if (position.value == previous &&
            raw >= position.min - control.hysteresis &&
            raw <= position.max + control.hysteresis) {
            return position.value;
        }
    }
    for (const auto &position : control.positions) {
        if (raw >= position.min && raw <= position.max) {
            return position.value;
        }
    }
    return control.default_enum;
}

void InputMapper::record_debug_trace(const std::string &control, ControlValue &value)
{
    const ControlConfig *config = find_control_config(control);
    bool has_value_rule = false;
    if (config != nullptr) {
        for (const auto &input : config->inputs) {
            if (input.kind != ControlInputKind::kSource) {
                has_value_rule = true;
                break;
            }
        }
    }
    if (!has_value_rule) {
        return;
    }
    const auto previous = controls_.find(control);
    if (debug_enabled_ &&
        (previous == controls_.end() || previous->second.debug_trace != value.debug_trace)) {
        debug_messages_.push_back("control debug: " + value.debug_trace);
    }
}

double InputMapper::apply_expo(double value, double expo) const
{
    if (expo <= 0.0) {
        return value;
    }
    if (expo > 1.0) {
        expo = 1.0;
    }
    const double curved = value * value * value;
    return value * (1.0 - expo) + curved * expo;
}

double InputMapper::apply_calibration(double value, const CalibrationConfig &calibration) const
{
    if (!calibration.enabled) {
        return value;
    }

    if (calibration.clamp) {
        value = std::max(calibration.input_min, std::min(calibration.input_max, value));
    }

    if (value <= calibration.input_center) {
        const double span = calibration.input_center - calibration.input_min;
        if (span <= 0.0) {
            return calibration.output_center;
        }
        const double t = (value - calibration.input_min) / span;
        return calibration.output_min + (calibration.output_center - calibration.output_min) * t;
    }

    const double span = calibration.input_max - calibration.input_center;
    if (span <= 0.0) {
        return calibration.output_center;
    }
    const double t = (value - calibration.input_center) / span;
    return calibration.output_center + (calibration.output_max - calibration.output_center) * t;
}

double InputMapper::apply_piecewise(double value, const std::vector<CurvePoint> &points) const
{
    if (points.empty()) {
        return value;
    }
    if (value <= points.front().input) {
        return points.front().output;
    }
    if (value >= points.back().input) {
        return points.back().output;
    }

    for (std::size_t index = 1; index < points.size(); ++index) {
        const CurvePoint &right = points[index];
        if (value > right.input) {
            continue;
        }
        const CurvePoint &left = points[index - 1];
        const double span = right.input - left.input;
        if (span <= 0.0) {
            return right.output;
        }
        const double t = (value - left.input) / span;
        return left.output + (right.output - left.output) * t;
    }
    return value;
}

double InputMapper::apply_curve(double value, const std::string &curve_name) const
{
    if (curve_name.empty()) {
        return value;
    }
    const auto curve_it = config_.curves.find(curve_name);
    if (curve_it == config_.curves.end()) {
        return value;
    }
    const CurveConfig &curve = curve_it->second;
    value = apply_calibration(value, curve.calibration);
    if (std::fabs(value) <= curve.deadzone) {
        value = 0.0;
    }
    if (curve.type == "piecewise") {
        value = apply_piecewise(value, curve.points);
    } else {
        value = apply_expo(value, curve.expo);
    }
    return std::max(curve.min_value, std::min(curve.max_value, value));
}

bool InputMapper::conditions_match(const Binding &binding) const
{
    return condition_matches(binding.when);
}

bool InputMapper::condition_matches(const ConditionConfig &condition) const
{
    if (condition.kind == "all") {
        for (const auto &child : condition.children) {
            if (!condition_matches(child)) {
                return false;
            }
        }
        return true;
    }
    if (condition.kind == "any") {
        for (const auto &child : condition.children) {
            if (condition_matches(child)) {
                return true;
            }
        }
        return false;
    }
    if (condition.kind == "not") {
        return condition.children.size() == 1 && !condition_matches(condition.children.front());
    }

    if (condition.kind == "raw_range") {
        const auto signal_it = signals_.find(condition.source);
        const double value = signal_it == signals_.end() ? 0.0 : signal_it->second;
        return value >= condition.min && value <= condition.max;
    }
    const auto control_it = controls_.find(condition.control);
    const ControlValue value = control_it == controls_.end() ? ControlValue() : control_it->second;

    if (condition.kind == "pressed") {
        return value.pressed;
    }
    if (condition.kind == "released") {
        return !value.pressed;
    }
    if (condition.kind == "equals") {
        return value.value == condition.value;
    }
    if (condition.kind == "range") {
        return value.analog >= condition.min && value.analog <= condition.max;
    }
    return false;
}

bool InputMapper::apply_level_output(const std::string &output)
{
    int slot = 0;
    int value = 0;
    if (!parse_button_output(output, slot, value)) {
        return false;
    }
    if (output_slots_[slot] != 0 && output_slots_[slot] != value) {
        if (config_.output_conflict_policy == "first_wins") {
            return true;
        }
        if (config_.output_conflict_policy == "error") {
            throw std::runtime_error("conflicting level outputs for btn_" + std::to_string(slot));
        }
    }
    output_slots_[slot] = value;
    return true;
}

bool InputMapper::apply_edge_pulse_output(const std::string &output)
{
    int slot = 0;
    int value = 0;
    if (!parse_button_output(output, slot, value)) {
        return false;
    }
    if (edge_pulse_slots_[slot] != 0 && edge_pulse_slots_[slot] != value) {
        if (config_.output_conflict_policy == "first_wins") {
            return true;
        }
        if (config_.output_conflict_policy == "error") {
            throw std::runtime_error("conflicting edge pulse outputs for btn_" + std::to_string(slot));
        }
    }
    edge_pulse_slots_[slot] = value;
    return true;
}

bool InputMapper::parse_button_output(const std::string &output, int &slot, int &value) const
{
    if (!starts_with(output, "btn_")) {
        return false;
    }

    const auto eq_pos = output.find('=');
    const std::string slot_text = output.substr(4, eq_pos == std::string::npos ? std::string::npos : eq_pos - 4);
    if (!parse_int_text(slot_text, slot)) {
        return false;
    }
    if (slot < 1 || slot > kButtonSlotCount) {
        return false;
    }

    if (eq_pos == std::string::npos) {
        value = 1;
    } else if (!parse_int_text(output.substr(eq_pos + 1), value)) {
        return false;
    }
    return true;
}

double InputMapper::read_analog_output(const AnalogOutputConfig &output) const
{
    double value = 0.0;
    if (output.mix == "sum") {
        for (const auto &control : output.controls) {
            value += read_analog_control(control);
        }
    } else if (output.mix == "first_active") {
        for (const auto &control : output.controls) {
            value = read_analog_control(control);
            if (std::fabs(value) > 0.0) {
                break;
            }
        }
    } else {
        for (const auto &control : output.controls) {
            const double candidate = read_analog_control(control);
            if (std::fabs(candidate) > std::fabs(value)) {
                value = candidate;
            }
        }
    }

    value = value * output.scale + output.offset;
    return std::max(output.min_value, std::min(output.max_value, value));
}

double InputMapper::read_analog_control(const std::string &control) const
{
    const auto control_it = controls_.find(control);
    if (control_it == controls_.end()) {
        return 0.0;
    }
    return control_it->second.analog;
}

}  // namespace remote_controller
