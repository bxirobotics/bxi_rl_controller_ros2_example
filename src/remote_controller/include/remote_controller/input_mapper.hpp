#pragma once

#include <chrono>
#include <map>
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
    std::vector<std::string> set_signal(const std::string &source, double value);
    void zero_motion_axes();
    void reset_motion();
    std::vector<std::string> tick();

    std::vector<std::string> handle_button(int button_index, bool pressed);
    std::vector<std::string> handle_keyboard_key(char key);

    void fill_message(communication::msg::MotionCommands &message);

private:
    struct ControlValue {
        double analog = 0.0;
        bool pressed = false;
        std::string value;
    };

    RemoteConfig config_;
    std::map<std::string, double> signals_;
    std::map<std::string, ControlValue> controls_;
    std::map<std::string, std::chrono::steady_clock::time_point> signal_expiry_;
    std::vector<bool> binding_active_;
    int output_slots_[kButtonSlotCount + 1] = {0};
    double height_filtered_ = kStandHeight;

    std::vector<std::string> refresh_bindings();
    void refresh_controls();
    ControlValue evaluate_control(const ControlConfig &control);
    double read_source(const SignalSourceConfig &source) const;
    bool conditions_match(const Binding &binding) const;
    bool condition_matches(const BindingCondition &condition) const;
    bool apply_level_output(const std::string &output);
    bool parse_button_output(const std::string &output, int &slot, int &value) const;
    double read_analog_control(const std::string &control) const;
    void set_button_slot(communication::msg::MotionCommands &message, int slot, int value);
};

}  // namespace remote_controller
