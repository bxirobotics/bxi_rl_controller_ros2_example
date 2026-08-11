#pragma once

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <utility>
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
    // Atomically applies one coherent device frame and evaluates bindings once.
    // Protocol drivers should use this instead of calling set_signal repeatedly.
    std::vector<std::string> set_signals(
        const std::vector<std::pair<std::string, double>> &signals);
    void touch_runtime_sources_with_prefix(const std::string &prefix);
    void touch_runtime_sources(const std::set<std::string> &sources);
    void clear_signals_with_prefix(const std::string &prefix);
    void clear_signals(const std::set<std::string> &sources);
    void set_input_edges_enabled(bool enabled);
    void set_debug_enabled(bool enabled);
    std::vector<std::string> take_debug_messages();
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
        std::string debug_trace;
    };

    struct InputCandidate {
        const ControlInputConfig *config = nullptr;
        bool active = false;
        double analog = 0.0;
        bool boolean = false;
        std::string enumeration;
    };

    RemoteConfig config_;
    std::map<std::string, double> signals_;
    std::map<std::string, ControlValue> controls_;
    std::map<std::string, std::chrono::steady_clock::time_point> signal_expiry_;
    std::map<std::string, std::chrono::steady_clock::time_point> signal_update_time_;
    std::set<std::string> timed_out_sources_;
    bool debug_enabled_ = false;
    std::vector<std::string> debug_messages_;
    bool input_edges_enabled_ = true;
    std::vector<bool> binding_active_;
    int output_slots_[kButtonSlotCount + 1] = {0};
    int edge_pulse_slots_[kButtonSlotCount + 1] = {0};
    int last_logged_output_slots_[kButtonSlotCount + 1] = {0};
    double height_filtered_ = kStandHeight;

    std::vector<std::string> refresh_bindings(bool emit_edges = true);
    void log_button_output_changes();
    void refresh_controls();
    void evaluate_control_recursive(
        const std::string &control,
        std::set<std::string> &visiting,
        std::set<std::string> &evaluated);
    const ControlConfig *find_control_config(const std::string &control) const;
    ControlValue evaluate_control(const ControlConfig &control);
    InputCandidate evaluate_input(
        const ControlConfig &control,
        const ControlInputConfig &input,
        const ControlValue &previous) const;
    double read_source(const SignalSourceConfig &source) const;
    std::string enum_from_raw(
        const ControlConfig &control,
        double raw,
        const std::string &previous) const;
    void evaluate_condition_dependencies(
        const ConditionConfig &condition,
        std::set<std::string> &visiting,
        std::set<std::string> &evaluated);
    double apply_calibration(double value, const CalibrationConfig &calibration) const;
    double apply_piecewise(double value, const std::vector<CurvePoint> &points) const;
    double apply_curve(double value, const std::string &curve_name) const;
    double apply_expo(double value, double expo) const;
    bool conditions_match(const Binding &binding) const;
    bool condition_matches(const ConditionConfig &condition) const;
    void record_debug_trace(const std::string &control, ControlValue &value);
    bool apply_level_output(const std::string &output);
    bool apply_edge_pulse_output(const std::string &output);
    bool parse_button_output(const std::string &output, int &slot, int &value) const;
    double read_analog_output(const AnalogOutputConfig &output) const;
    double read_analog_control(const std::string &control) const;
};

}  // namespace remote_controller
