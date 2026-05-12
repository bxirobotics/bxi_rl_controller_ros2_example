#include "remote_controller/input_mapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

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
    return refresh_bindings();
}

void InputMapper::zero_motion_axes()
{
    signals_["keyboard.axis.vx"] = 0.0;
    signals_["keyboard.axis.vy"] = 0.0;
    signals_["keyboard.axis.yaw"] = 0.0;
    signal_expiry_.erase("keyboard.axis.vx");
    signal_expiry_.erase("keyboard.axis.vy");
    signal_expiry_.erase("keyboard.axis.yaw");
    refresh_bindings();
}

void InputMapper::reset_motion()
{
    for (auto &item : signals_) {
        if (is_axis_source(item.first)) {
            item.second = 0.0;
        }
    }
    for (auto &item : signal_expiry_) {
        if (is_axis_source(item.first)) {
            item.second = std::chrono::steady_clock::time_point();
        }
    }
    for (auto &control : controls_) {
        control.second.analog = 0.0;
    }
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
    const auto expiry = now + std::chrono::milliseconds(config_.keyboard.hold_ms);

    if (key == config_.keyboard.forward) {
        signals_["keyboard.axis.vx"] = -1.0;
        signal_expiry_["keyboard.axis.vx"] = expiry;
        return refresh_bindings();
    }
    if (key == config_.keyboard.backward) {
        signals_["keyboard.axis.vx"] = 1.0;
        signal_expiry_["keyboard.axis.vx"] = expiry;
        return refresh_bindings();
    }
    if (key == config_.keyboard.yaw_left) {
        signals_["keyboard.axis.yaw"] = -1.0;
        signal_expiry_["keyboard.axis.yaw"] = expiry;
        return refresh_bindings();
    }
    if (key == config_.keyboard.yaw_right) {
        signals_["keyboard.axis.yaw"] = 1.0;
        signal_expiry_["keyboard.axis.yaw"] = expiry;
        return refresh_bindings();
    }
    if (key == config_.keyboard.strafe_left) {
        signals_["keyboard.axis.vy"] = -1.0;
        signal_expiry_["keyboard.axis.vy"] = expiry;
        return refresh_bindings();
    }
    if (key == config_.keyboard.strafe_right) {
        signals_["keyboard.axis.vy"] = 1.0;
        signal_expiry_["keyboard.axis.vy"] = expiry;
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
        signal_expiry_[signal] = expiry;
    }
    return refresh_bindings();
}

void InputMapper::fill_message(communication::msg::MotionCommands &message)
{
    refresh_controls();

    for (const auto &output : config_.analog_outputs) {
        double value = 0.0;
        for (const auto &control : output.controls) {
            const double candidate = read_analog_control(control);
            if (std::fabs(candidate) > std::fabs(value)) {
                value = candidate;
            }
        }
        value += output.offset;

        if (output.field == "vx") {
            message.vel_des.x = static_cast<float>(value);
        } else if (output.field == "vy") {
            message.vel_des.y = static_cast<float>(value);
        } else if (output.field == "yaw") {
            message.yawdot_des = static_cast<float>(value);
        }
    }

    for (int slot = 1; slot <= kButtonSlotCount; ++slot) {
        set_button_slot(message, slot, output_slots_[slot]);
    }

    height_filtered_ = height_filtered_ * 0.9 + kStandHeight * 0.1;
    message.height_des = static_cast<float>(height_filtered_);
}

std::vector<std::string> InputMapper::refresh_bindings()
{
    refresh_controls();
    std::fill(output_slots_, output_slots_ + kButtonSlotCount + 1, 0);

    std::vector<std::string> edge_outputs;
    for (std::size_t index = 0; index < config_.bindings.size(); ++index) {
        const Binding &binding = config_.bindings[index];
        const bool active = conditions_match(binding);

        if (binding.mode == "edge") {
            if (active && !binding_active_[index]) {
                edge_outputs.push_back(binding.output);
            }
        } else if (active) {
            apply_level_output(binding.output);
        }

        binding_active_[index] = active;
    }

    return edge_outputs;
}

void InputMapper::refresh_controls()
{
    for (const auto &control : config_.controls) {
        controls_[control.name] = evaluate_control(control);
    }
}

InputMapper::ControlValue InputMapper::evaluate_control(const ControlConfig &control)
{
    const ControlValue previous = controls_[control.name];
    ControlValue value = previous;

    double raw = 0.0;
    for (const auto &source : control.sources) {
        const double candidate = read_source(source);
        if (std::fabs(candidate) > std::fabs(raw)) {
            raw = candidate;
        }
    }

    if (control.type == "analog") {
        double analog = raw;
        if (std::fabs(analog) <= control.deadzone) {
            analog = 0.0;
        }
        if (analog > 0.0) {
            analog *= control.max_value;
        } else if (analog < 0.0) {
            analog *= -control.min_value;
        }
        value.analog = analog * control.alpha + previous.analog * (1.0 - control.alpha);
        value.pressed = std::fabs(value.analog) > 0.0;
        value.value.clear();
        return value;
    }

    if (control.type == "enum") {
        const double sticky_margin = control.hysteresis;
        for (const auto &position : control.positions) {
            if (position.value == previous.value &&
                raw >= position.min - sticky_margin &&
                raw <= position.max + sticky_margin) {
                value.value = position.value;
                value.pressed = true;
                value.analog = raw;
                return value;
            }
        }
        value.value.clear();
        for (const auto &position : control.positions) {
            if (raw >= position.min && raw <= position.max) {
                value.value = position.value;
                break;
            }
        }
        value.pressed = !value.value.empty();
        value.analog = raw;
        return value;
    }

    const double press_threshold = previous.pressed ?
        control.threshold - control.hysteresis :
        control.threshold + control.hysteresis;
    value.pressed = raw >= press_threshold;
    value.analog = raw;
    value.value = value.pressed ? "true" : "false";
    return value;
}

double InputMapper::read_source(const SignalSourceConfig &source) const
{
    const auto signal_it = signals_.find(source.source);
    const double value = signal_it == signals_.end() ? 0.0 : signal_it->second;
    return value * source.direction * source.scale;
}

bool InputMapper::conditions_match(const Binding &binding) const
{
    for (const auto &group : binding.condition_groups) {
        bool group_matches = true;
        for (const auto &condition : group) {
            if (!condition_matches(condition)) {
                group_matches = false;
                break;
            }
        }
        if (group_matches) {
            return true;
        }
    }
    return false;
}

bool InputMapper::condition_matches(const BindingCondition &condition) const
{
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
    output_slots_[slot] = value;
    return true;
}

bool InputMapper::parse_button_output(const std::string &output, int &slot, int &value) const
{
    if (!starts_with(output, "btn_")) {
        return false;
    }

    const auto eq_pos = output.find('=');
    const std::string slot_text = output.substr(4, eq_pos == std::string::npos ? std::string::npos : eq_pos - 4);
    slot = std::atoi(slot_text.c_str());
    if (slot < 1 || slot > kButtonSlotCount) {
        return false;
    }

    value = eq_pos == std::string::npos ? 1 : std::atoi(output.substr(eq_pos + 1).c_str());
    return true;
}

double InputMapper::read_analog_control(const std::string &control) const
{
    const auto control_it = controls_.find(control);
    if (control_it == controls_.end()) {
        return 0.0;
    }
    return control_it->second.analog;
}

void InputMapper::set_button_slot(communication::msg::MotionCommands &message, int slot, int value)
{
    switch (slot) {
    case 1: message.btn_1 = value; break;
    case 2: message.btn_2 = value; break;
    case 3: message.btn_3 = value; break;
    case 4: message.btn_4 = value; break;
    case 5: message.btn_5 = value; break;
    case 6: message.btn_6 = value; break;
    case 7: message.btn_7 = value; break;
    case 8: message.btn_8 = value; break;
    case 9: message.btn_9 = value; break;
    case 10: message.btn_10 = value; break;
    default: break;
    }
}

}  // namespace remote_controller
