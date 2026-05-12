#pragma once

#include <string>
#include <vector>

#include <communication/msg/motion_commands.hpp>

#include "remote_controller/config.hpp"

namespace remote_controller {

class InputMapper {
public:
    explicit InputMapper(RemoteConfig config);

    const RemoteConfig &config() const;

    std::vector<std::string> set_axis(int axis_index, double value);
    void zero_motion_axes();
    void reset_motion();

    std::vector<std::string> handle_button(int button_index, bool pressed);
    std::vector<std::string> handle_keyboard_key(char key);
    bool apply_button_output(const std::string &output);

    void fill_message(communication::msg::MotionCommands &message);

private:
    RemoteConfig config_;
    double raw_axes_[kAxisCount] = {0.0};
    int output_slots_[kButtonSlotCount + 1] = {0};
    double height_filtered_ = kStandHeight;

    double filter_axis(AxisConfig &axis);
    const Binding *best_binding_for(const std::string &control) const;
    bool binding_matches(const Binding &binding, const std::string &trigger_control) const;
    void set_button_slot(communication::msg::MotionCommands &message, int slot, int value);
};

}  // namespace remote_controller
