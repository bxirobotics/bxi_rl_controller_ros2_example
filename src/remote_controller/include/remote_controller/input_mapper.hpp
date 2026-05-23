#pragma once

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <communication/msg/motion_commands.hpp>

#include "remote_controller/config.hpp"

namespace remote_controller {

class InputMapper {
public:
    explicit InputMapper(RemoteConfig config);

    const RemoteConfig &config() const;
    void reload_config(RemoteConfig config);

    std::vector<std::string> set_axis(int axis_index, double value);
    std::vector<std::string> set_signal(const std::string &source, double value);
    void touch_runtime_sources_with_prefix(const std::string &prefix);
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
    std::map<std::string, std::chrono::steady_clock::time_point> signal_update_time_;
    std::set<std::string> timed_out_sources_;
    std::vector<bool> binding_active_;
    int output_slots_[kButtonSlotCount + 1] = {0};
    int edge_pulse_slots_[kButtonSlotCount + 1] = {0};
    double height_filtered_ = kStandHeight;

    std::vector<std::string> refresh_bindings(bool emit_edges = true);
    void refresh_controls();
    void evaluate_control_recursive(
        const std::string &control,
        std::set<std::string> &visiting,
        std::set<std::string> &evaluated);
    const ControlConfig *find_control_config(const std::string &control) const;
    ControlValue evaluate_control(const ControlConfig &control);
    double read_source(const SignalSourceConfig &source) const;
    double mix_sources(const ControlConfig &control) const;
    double apply_calibration(double value, const CalibrationConfig &calibration) const;
    double apply_piecewise(double value, const std::vector<CurvePoint> &points) const;
    double apply_curve(double value, const std::string &curve_name) const;
    double apply_expo(double value, double expo) const;
    bool conditions_match(const Binding &binding) const;
    bool condition_groups_match(const std::vector<std::vector<BindingCondition>> &groups) const;
    bool condition_matches(const BindingCondition &condition) const;
    bool apply_level_output(const std::string &output);
    bool apply_edge_pulse_output(const std::string &output);
    bool parse_button_output(const std::string &output, int &slot, int &value) const;
    double read_analog_output(const AnalogOutputConfig &output) const;
    double read_analog_control(const std::string &control) const;
};

}  // namespace remote_controller
