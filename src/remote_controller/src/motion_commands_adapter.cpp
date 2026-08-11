#include "remote_controller/motion_commands_adapter.hpp"

namespace remote_controller {

bool is_motion_command_field_supported(const std::string &field)
{
    return field == "vel_des.x" ||
           field == "vel_des.y" ||
           field == "vel_des.z" ||
           field == "yawdot_des" ||
           field == "height_des";
}

bool set_motion_command_field(
    communication::msg::MotionCommands &message,
    const std::string &field,
    double value)
{
    const float float_value = static_cast<float>(value);
    if (field == "vel_des.x") {
        message.vel_des.x = float_value;
        return true;
    }
    if (field == "vel_des.y") {
        message.vel_des.y = float_value;
        return true;
    }
    if (field == "vel_des.z") {
        message.vel_des.z = float_value;
        return true;
    }
    if (field == "yawdot_des") {
        message.yawdot_des = float_value;
        return true;
    }
    if (field == "height_des") {
        message.height_des = float_value;
        return true;
    }
    return false;
}

void set_motion_command_button_slot(
    communication::msg::MotionCommands &message,
    int slot,
    int value)
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
