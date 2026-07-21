#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "remote_controller/input_mapper.hpp"

namespace {

using remote_controller::ConditionConfig;
using remote_controller::ControlConfig;
using remote_controller::ControlInputConfig;
using remote_controller::ControlInputKind;
using remote_controller::EnumPositionConfig;
using remote_controller::InputMapper;
using remote_controller::RemoteConfig;

ControlInputConfig source_input(const std::string &source, int priority = 0)
{
    ControlInputConfig input;
    input.name = source;
    input.priority = priority;
    input.kind = ControlInputKind::kSource;
    input.source.source = source;
    return input;
}

ControlInputConfig conditional_input(
    const ConditionConfig &condition,
    double value,
    int priority = 0)
{
    ControlInputConfig input;
    input.name = "conditional";
    input.priority = priority;
    input.kind = ControlInputKind::kConditionalValue;
    input.when = condition;
    input.analog_value = value;
    return input;
}

ConditionConfig equals(const std::string &control, const std::string &value)
{
    ConditionConfig condition;
    condition.kind = "equals";
    condition.control = control;
    condition.value = value;
    return condition;
}

ConditionConfig raw_range(const std::string &source, double min, double max)
{
    ConditionConfig condition;
    condition.kind = "raw_range";
    condition.source = source;
    condition.min = min;
    condition.max = max;
    return condition;
}

EnumPositionConfig position(const std::string &value, double min, double max)
{
    EnumPositionConfig result;
    result.value = value;
    result.min = min;
    result.max = max;
    return result;
}

double yaw(InputMapper &mapper)
{
    communication::msg::MotionCommands message;
    mapper.fill_message(message);
    return message.yawdot_des;
}

void expect_close(double actual, double expected)
{
    if (std::fabs(actual - expected) > 1e-6) {
        std::abort();
    }
}

void expect(bool value)
{
    if (!value) {
        std::abort();
    }
}

void test_enum_conditions_are_independent_of_raw_gaps()
{
    RemoteConfig config;

    ControlConfig group;
    group.name = "button_group";
    group.type = "enum";
    group.mix = "first_active";
    group.default_enum = "idle";
    group.inputs.push_back(source_input("raw.group"));
    group.positions = {
        position("idle", -0.1, 0.1),
        position("dpad_up", 0.2, 0.3),
        position("dpad_down", 0.4, 0.5),
        position("dpad_left", 0.6, 0.7),
        position("dpad_right", 0.9, 1.0),
    };
    config.controls.push_back(group);

    ControlConfig yaw_control;
    yaw_control.name = "yaw";
    yaw_control.type = "analog";
    yaw_control.mix = "sum";
    yaw_control.inputs.push_back(conditional_input(equals("button_group", "dpad_left"), 1.0));
    yaw_control.inputs.push_back(conditional_input(equals("button_group", "dpad_right"), -1.0));
    config.controls.push_back(yaw_control);

    remote_controller::AnalogOutputConfig output;
    output.field = "yawdot_des";
    output.controls = {"yaw"};
    config.analog_outputs.push_back(output);

    InputMapper mapper(config);
    mapper.set_signal("raw.group", 0.65);
    expect_close(yaw(mapper), 1.0);

    mapper.set_signal("raw.group", 0.95);
    expect_close(yaw(mapper), -1.0);

    mapper.set_signal("raw.group", 0.25);
    expect_close(yaw(mapper), 0.0);
}

void test_priority_preempts_lower_active_inputs()
{
    RemoteConfig config;
    ControlConfig control;
    control.name = "priority_axis";
    control.type = "analog";
    control.mix = "sum";
    control.inputs.push_back(source_input("raw.axis", 0));
    control.inputs.push_back(conditional_input(raw_range("raw.override", 0.5, 1.0), -1.0, 100));
    config.controls.push_back(control);

    remote_controller::AnalogOutputConfig output;
    output.field = "yawdot_des";
    output.controls = {"priority_axis"};
    config.analog_outputs.push_back(output);

    InputMapper mapper(config);
    mapper.set_signals({{"raw.axis", 0.8}, {"raw.override", 0.0}});
    expect_close(yaw(mapper), 0.8);

    mapper.set_signal("raw.override", 0.7);
    expect_close(yaw(mapper), -1.0);
}

void test_bool_all_keeps_inactive_raw_inputs_in_the_selected_group()
{
    RemoteConfig config;

    ControlConfig both;
    both.name = "both";
    both.type = "bool";
    both.mix = "all";
    both.inputs.push_back(source_input("raw.left"));
    both.inputs.push_back(source_input("raw.right"));
    config.controls.push_back(both);

    ControlConfig gate;
    gate.name = "gate";
    gate.type = "analog";
    gate.inputs.push_back(conditional_input(
        []() {
            ConditionConfig condition;
            condition.kind = "pressed";
            condition.control = "both";
            return condition;
        }(),
        1.0));
    config.controls.push_back(gate);

    remote_controller::AnalogOutputConfig output;
    output.field = "yawdot_des";
    output.controls = {"gate"};
    config.analog_outputs.push_back(output);

    InputMapper mapper(config);
    mapper.set_signals({{"raw.left", 1.0}, {"raw.right", 0.0}});
    expect_close(yaw(mapper), 0.0);

    mapper.set_signal("raw.right", 1.0);
    expect_close(yaw(mapper), 1.0);
}

void test_debug_reports_changed_rule_selection()
{
    RemoteConfig config;
    ControlConfig control;
    control.name = "debug_axis";
    control.type = "analog";
    control.inputs.push_back(conditional_input(raw_range("raw.axis", 0.1, 1.0), 0.5));
    config.controls.push_back(control);

    InputMapper mapper(config);
    mapper.set_debug_enabled(true);
    mapper.take_debug_messages();
    mapper.set_signal("raw.axis", 0.5);
    const std::vector<std::string> messages = mapper.take_debug_messages();
    expect(!messages.empty());
    expect(messages.front().find("debug_axis") != std::string::npos);
}

}  // namespace

int main()
{
    test_enum_conditions_are_independent_of_raw_gaps();
    test_priority_preempts_lower_active_inputs();
    test_bool_all_keeps_inactive_raw_inputs_in_the_selected_group();
    test_debug_reports_changed_rule_selection();
    return 0;
}
