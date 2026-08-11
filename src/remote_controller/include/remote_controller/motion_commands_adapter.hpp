#pragma once

#include <string>

#include <communication/msg/motion_commands.hpp>

namespace remote_controller {

bool is_motion_command_field_supported(const std::string &field);

bool set_motion_command_field(
    communication::msg::MotionCommands &message,
    const std::string &field,
    double value);

void set_motion_command_button_slot(
    communication::msg::MotionCommands &message,
    int slot,
    int value);

}  // namespace remote_controller
