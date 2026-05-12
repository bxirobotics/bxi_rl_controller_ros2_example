#include "remote_controller/input_mapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace remote_controller {

InputMapper::InputMapper(RemoteConfig config)
    : config_(std::move(config))
{
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
    raw_axes_[axis_index] = value;

    std::vector<std::string> outputs;
    for (const auto &axis_button : config_.axis_buttons) {
        if (axis_button.index != axis_index) {
            continue;
        }

        const double normalized = value * axis_button.direction / static_cast<double>(kAxisValueMax);
        const bool pressed = normalized >= axis_button.threshold;
        const bool was_pressed = config_.control_down[axis_button.control];
        config_.control_down[axis_button.control] = pressed;

        if (was_pressed && !pressed) {
            outputs.insert(
                outputs.end(),
                axis_button.release_outputs.begin(),
                axis_button.release_outputs.end());
        }
    }

    return outputs;
}

void InputMapper::zero_motion_axes()
{
    set_axis(config_.vx.index, 0.0);
    set_axis(config_.vy.index, 0.0);
    set_axis(config_.yaw.index, 0.0);
}

void InputMapper::reset_motion()
{
    std::fill(raw_axes_, raw_axes_ + kAxisCount, 0.0);
    config_.vx.filtered = 0.0;
    config_.vy.filtered = 0.0;
    config_.yaw.filtered = 0.0;
    height_filtered_ = kStandHeight;
}

std::vector<std::string> InputMapper::handle_button(int button_index, bool pressed)
{
    const auto control_it = config_.control_by_button_index.find(button_index);
    if (control_it == config_.control_by_button_index.end()) {
        return {};
    }

    const std::string control = control_it->second;
    config_.control_down[control] = pressed;
    if (!pressed) {
        return {};
    }

    const Binding *binding = best_binding_for(control);
    if (binding == nullptr) {
        return {};
    }
    return {binding->output};
}

std::vector<std::string> InputMapper::handle_keyboard_key(char key)
{
    if (key == config_.keyboard.forward) {
        set_axis(config_.vx.index, -kAxisValueMax);
        return {};
    }
    if (key == config_.keyboard.backward) {
        set_axis(config_.vx.index, kAxisValueMax);
        return {};
    }
    if (key == config_.keyboard.yaw_left) {
        set_axis(config_.yaw.index, -kAxisValueMax);
        return {};
    }
    if (key == config_.keyboard.yaw_right) {
        set_axis(config_.yaw.index, kAxisValueMax);
        return {};
    }
    if (key == config_.keyboard.strafe_left) {
        set_axis(config_.vy.index, -kAxisValueMax);
        return {};
    }
    if (key == config_.keyboard.strafe_right) {
        set_axis(config_.vy.index, kAxisValueMax);
        return {};
    }
    if (key == config_.keyboard.stop) {
        zero_motion_axes();
        return {};
    }

    const auto output_it = config_.keyboard.bindings.find(key);
    if (output_it == config_.keyboard.bindings.end()) {
        return {};
    }
    return {output_it->second};
}

bool InputMapper::apply_button_output(const std::string &output)
{
    if (!starts_with(output, "btn_")) {
        return false;
    }

    const auto eq_pos = output.find('=');
    const std::string slot_text = output.substr(4, eq_pos == std::string::npos ? std::string::npos : eq_pos - 4);
    const int slot = std::atoi(slot_text.c_str());
    if (slot < 1 || slot > kButtonSlotCount) {
        return false;
    }

    if (eq_pos == std::string::npos) {
        output_slots_[slot] = output_slots_[slot] == 0 ? 1 : 0;
    } else {
        output_slots_[slot] = std::atoi(output.substr(eq_pos + 1).c_str());
    }

    printf("btn_%d: %d\n", slot, output_slots_[slot]);
    return true;
}

void InputMapper::fill_message(communication::msg::MotionCommands &message)
{
    message.vel_des.x = filter_axis(config_.vx) + config_.vel_offset;
    message.vel_des.y = filter_axis(config_.vy);
    message.yawdot_des = static_cast<float>(filter_axis(config_.yaw));

    for (int slot = 1; slot <= kButtonSlotCount; ++slot) {
        set_button_slot(message, slot, output_slots_[slot]);
    }

    height_filtered_ = height_filtered_ * 0.9 + kStandHeight * 0.1;
    message.height_des = static_cast<float>(height_filtered_);
}

double InputMapper::filter_axis(AxisConfig &axis)
{
    if (axis.index < 0 || axis.index >= kAxisCount) {
        return 0.0;
    }

    double value = raw_axes_[axis.index] * axis.direction / static_cast<double>(kAxisValueMax);
    const double deadzone = axis.deadzone / static_cast<double>(kAxisValueMax);
    value = std::fabs(value) > deadzone ? value : 0.0;

    if (value > 0.0) {
        value *= axis.max_value;
    } else if (value < 0.0) {
        value *= -axis.min_value;
    }

    axis.filtered = value * axis.alpha + axis.filtered * (1.0 - axis.alpha);
    return axis.filtered;
}

const Binding *InputMapper::best_binding_for(const std::string &control) const
{
    const Binding *best = nullptr;
    for (const auto &binding : config_.bindings) {
        if (!binding_matches(binding, control)) {
            continue;
        }
        if (best == nullptr || binding.controls.size() > best->controls.size()) {
            best = &binding;
        }
    }
    return best;
}

bool InputMapper::binding_matches(const Binding &binding, const std::string &trigger_control) const
{
    if (binding.controls.empty() || binding.controls.back() != trigger_control) {
        return false;
    }

    bool binding_uses_modifier = false;
    for (const auto &control : binding.controls) {
        const auto down_it = config_.control_down.find(control);
        if (down_it == config_.control_down.end() || !down_it->second) {
            return false;
        }
        if (config_.modifiers.count(control) > 0) {
            binding_uses_modifier = true;
        }
    }

    if (!binding_uses_modifier) {
        for (const auto &modifier : config_.modifiers) {
            const auto down_it = config_.control_down.find(modifier);
            if (down_it != config_.control_down.end() && down_it->second) {
                return false;
            }
        }
    }

    return true;
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
