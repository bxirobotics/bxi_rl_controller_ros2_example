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
    double direction = 1.0;
    double scale = 1.0;
};

struct EnumPositionConfig {
    std::string value;
    double min = 0.0;
    double max = 0.0;
};

struct ControlConfig {
    std::string name;
    std::string type = "bool";
    std::vector<SignalSourceConfig> sources;
    double deadzone = 0.0;
    double min_value = -1.0;
    double max_value = 1.0;
    double alpha = 0.03;
    double threshold = 0.5;
    double hysteresis = 0.0;
    std::vector<EnumPositionConfig> positions;
};

struct AnalogOutputConfig {
    std::string field;
    std::vector<std::string> controls;
    double offset = 0.0;
};

struct BindingCondition {
    std::string kind = "pressed";
    std::string control;
    std::string value;
    double min = 0.0;
    double max = 0.0;
};

struct Binding {
    std::string output;
    std::string mode;
    std::vector<std::vector<BindingCondition>> condition_groups;
};

struct KeyboardConfig {
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
};

struct SystemMutexConfig {
    std::string name;
    std::string acquire;
    std::string release;
};

struct RemoteConfig {
    std::string js_device = "/dev/input/js0";
    std::map<std::string, std::string> source_aliases;
    std::vector<ControlConfig> controls;
    std::vector<AnalogOutputConfig> analog_outputs;
    std::vector<Binding> bindings;
    KeyboardConfig keyboard;
    std::map<std::string, std::vector<std::string>> system_commands;
    std::vector<SystemMutexConfig> system_mutexes;
    std::set<std::string> reset_motion_after_system;
};

RemoteConfig load_remote_config(const std::string &path);
char key_from_name(const std::string &name);
bool starts_with(const std::string &value, const std::string &prefix);

}  // namespace remote_controller
