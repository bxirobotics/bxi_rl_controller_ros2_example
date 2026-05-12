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

struct AxisConfig {
    int index = 0;
    int direction = 1;
    double deadzone = 1000.0;
    double min_value = -1.0;
    double max_value = 1.0;
    double alpha = 0.03;
    double filtered = 0.0;
};

struct Binding {
    std::string output;
    std::vector<std::string> controls;
};

struct AxisButtonConfig {
    std::string control;
    int index = 0;
    int direction = 1;
    double threshold = 0.85;
    std::vector<std::string> release_outputs;
};

struct KeyboardConfig {
    char forward = 'w';
    char backward = 's';
    char yaw_left = 'a';
    char yaw_right = 'd';
    char strafe_left = 'q';
    char strafe_right = 'e';
    char stop = ' ';
    int timeout_us = 150000;
    std::map<char, std::string> bindings;
};

struct SystemMutexConfig {
    std::string name;
    std::string acquire;
    std::string release;
};

struct RemoteConfig {
    std::string js_device = "/dev/input/js0";
    double vel_offset = 0.0;
    AxisConfig vx;
    AxisConfig vy;
    AxisConfig yaw;
    std::map<int, std::string> control_by_button_index;
    std::map<std::string, bool> control_down;
    std::set<std::string> modifiers;
    std::vector<AxisButtonConfig> axis_buttons;
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
