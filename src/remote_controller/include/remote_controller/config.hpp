#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace remote_controller {

constexpr int kAxisValueMax = 32767;
constexpr int kAxisCount = 32;
constexpr int kButtonSlotCount = 10;
constexpr double kStandHeight = 1.0;

struct SignalSourceConfig {
    std::string source;
    std::string curve;
    double direction = 1.0;
    double scale = 1.0;
    double offset = 0.0;
    double deadzone = 0.0;
    double expo = 0.0;
};

struct EnumPositionConfig {
    std::string value;
    double min = 0.0;
    double max = 0.0;
};

struct BindingCondition {
    std::string kind = "pressed";
    std::string control;
    std::string value;
    double min = 0.0;
    double max = 0.0;
};

struct ControlConfig {
    std::string name;
    std::string type = "bool";
    std::string mix = "max_abs";
    std::string curve;
    std::string default_value;
    std::vector<std::vector<BindingCondition>> expression_groups;
    std::vector<SignalSourceConfig> sources;
    double deadzone = 0.0;
    double min_value = -1.0;
    double max_value = 1.0;
    double alpha = 0.03;
    double threshold = 0.5;
    double hysteresis = 0.0;
    double expo = 0.0;
    bool invert = false;
    std::vector<EnumPositionConfig> positions;
};

struct AnalogOutputConfig {
    std::string field;
    std::string mix = "max_abs";
    std::vector<std::string> controls;
    double scale = 1.0;
    double offset = 0.0;
    double min_value = -1000000000.0;
    double max_value = 1000000000.0;
};

struct Binding {
    std::string output;
    std::string mode;
    std::vector<std::vector<BindingCondition>> condition_groups;
};

struct CurvePoint {
    double input = 0.0;
    double output = 0.0;
};

struct CalibrationConfig {
    bool enabled = false;
    bool clamp = true;
    double input_min = -1.0;
    double input_center = 0.0;
    double input_max = 1.0;
    double output_min = -1.0;
    double output_center = 0.0;
    double output_max = 1.0;
};

struct CurveConfig {
    std::string name;
    std::string type = "expo";
    CalibrationConfig calibration;
    std::vector<CurvePoint> points;
    double deadzone = 0.0;
    double expo = 0.0;
    double min_value = -1.0;
    double max_value = 1.0;
};

struct SourceRuntimeConfig {
    std::string source;
    int timeout_ms = 0;
    double failsafe = 0.0;
};

struct ConfigDiagnostic {
    std::string severity;
    std::string message;
};

struct KeyboardConfig {
    std::string vx_source = "keyboard.vx";
    std::string vy_source = "keyboard.vy";
    std::string yaw_source = "keyboard.yaw";
    char forward = 'w';
    char backward = 's';
    char yaw_left = 'a';
    char yaw_right = 'd';
    char strafe_left = 'q';
    char strafe_right = 'e';
    char stop = ' ';
    int poll_timeout_us = 20000;
    int hold_ms = 200;
    std::map<char, std::vector<std::string>> signals_by_key;
    std::map<std::string, int> hold_ms_by_signal;
};

struct SystemMutexConfig {
    std::string name;
    std::string acquire;
    std::string release;
};

struct RemoteConfig {
    std::string js_device = "/dev/input/js0";
    std::string output_conflict_policy = "last_wins";
    bool publish_on_change = true;
    std::map<std::string, std::string> source_aliases;
    std::map<std::string, CurveConfig> curves;
    std::map<std::string, SourceRuntimeConfig> source_runtime;
    std::vector<ControlConfig> controls;
    std::vector<AnalogOutputConfig> analog_outputs;
    std::vector<Binding> bindings;
    KeyboardConfig keyboard;
    std::map<std::string, std::vector<std::string>> system_commands;
    std::vector<SystemMutexConfig> system_mutexes;
    std::set<std::string> reset_motion_after_system;
    std::vector<ConfigDiagnostic> diagnostics;
};

RemoteConfig load_remote_config(const std::string &path);
char key_from_name(const std::string &name);
bool starts_with(const std::string &value, const std::string &prefix);

}  // namespace remote_controller
