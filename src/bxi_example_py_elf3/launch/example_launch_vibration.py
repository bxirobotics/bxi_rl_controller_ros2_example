import os
from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description(
    controller_executable_default="bxi_example_py_elf3_vibration",
    controller_name_default="bxi_example_py_elf3_vibration",
    joint_test_required_default="true",
    allow_hardware_without_joint_test_default="false",
):
    workspace_config = Path(
        "src/bxi_example_py_elf3/config/suspended_tests.yaml"
    )
    installed_config = (
        get_package_share_path("bxi_example_py_elf3")
        / "config"
        / "suspended_tests.yaml"
    )
    default_config_file = str(
        workspace_config if workspace_config.is_file() else installed_config
    )
    model_file = os.path.join(
        get_package_share_path("bxi_example_py_elf3"),
        "data/elf3.xml",
    )

    joint_name = LaunchConfiguration("joint_name")
    amplitude_rad = LaunchConfiguration("amplitude_rad")
    start_frequency_hz = LaunchConfiguration("start_frequency_hz")
    end_frequency_hz = LaunchConfiguration("end_frequency_hz")
    duration_sec = LaunchConfiguration("duration_sec")
    control_rate_hz = LaunchConfiguration("control_rate_hz")
    stop_ramp_sec = LaunchConfiguration("stop_ramp_sec")
    motion_button_mode = LaunchConfiguration("motion_button_mode")
    joint_test_required = LaunchConfiguration("joint_test_required")
    allow_hardware_without_joint_test = LaunchConfiguration(
        "allow_hardware_without_joint_test"
    )
    joint_test_amplitude_rad = LaunchConfiguration("joint_test_amplitude_rad")
    joint_test_move_sec = LaunchConfiguration("joint_test_move_sec")
    joint_test_hold_sec = LaunchConfiguration("joint_test_hold_sec")
    joint_test_min_motion_rad = LaunchConfiguration(
        "joint_test_min_motion_rad"
    )
    log_rate_hz = LaunchConfiguration("log_rate_hz")
    release_suspension = LaunchConfiguration("release_suspension")
    auto_start = LaunchConfiguration("auto_start")
    log_csv_path = LaunchConfiguration("log_csv_path")
    start_remote_controller = LaunchConfiguration("start_remote_controller")
    controller_executable = LaunchConfiguration("controller_executable")
    controller_name = LaunchConfiguration("controller_name")
    controller_config_file = LaunchConfiguration("controller_config_file")

    simulation_node = Node(
        package="mujoco",
        executable="simulation",
        name="simulation_mujoco",
        output="screen",
        parameters=[{"simulation/model_file": model_file}],
        emulate_tty=True,
    )

    vibration_node = Node(
        package="bxi_example_py_elf3",
        executable=controller_executable,
        name=controller_name,
        output="screen",
        parameters=[
            controller_config_file,
            {
                "topic_prefix": "simulation/",
                "joint_name": joint_name,
                "amplitude_rad": ParameterValue(
                    amplitude_rad, value_type=float
                ),
                "start_frequency_hz": ParameterValue(
                    start_frequency_hz, value_type=float
                ),
                "end_frequency_hz": ParameterValue(
                    end_frequency_hz, value_type=float
                ),
                "duration_sec": ParameterValue(duration_sec, value_type=float),
                "control_rate_hz": ParameterValue(
                    control_rate_hz, value_type=float
                ),
                "stop_ramp_sec": ParameterValue(stop_ramp_sec, value_type=float),
                "motion_button_mode": motion_button_mode,
                "joint_test_required": ParameterValue(
                    joint_test_required, value_type=bool
                ),
                "allow_hardware_without_joint_test": ParameterValue(
                    allow_hardware_without_joint_test,
                    value_type=bool,
                ),
                "joint_test_amplitude_rad": ParameterValue(
                    joint_test_amplitude_rad, value_type=float
                ),
                "joint_test_move_sec": ParameterValue(
                    joint_test_move_sec, value_type=float
                ),
                "joint_test_hold_sec": ParameterValue(
                    joint_test_hold_sec, value_type=float
                ),
                "joint_test_min_motion_rad": ParameterValue(
                    joint_test_min_motion_rad, value_type=float
                ),
                "joint_test_verify_feedback": True,
                "log_rate_hz": ParameterValue(log_rate_hz, value_type=float),
                "release_suspension": ParameterValue(
                    release_suspension, value_type=bool
                ),
                "auto_start": ParameterValue(auto_start, value_type=bool),
                "log_csv_path": log_csv_path,
            }
        ],
        emulate_tty=True,
    )

    remote_node = Node(
        package="remote_controller",
        executable="remote_controller",
        name="remote_controller",
        output="screen",
        condition=IfCondition(start_remote_controller),
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "controller_config_file",
                default_value=default_config_file,
                description=(
                    "YAML parameters for the combined A-key full-range test"
                ),
            ),
            DeclareLaunchArgument(
                "joint_name",
                default_value="all",
                description="Elf3 joint to excite; use 'all' for all 29 joints",
            ),
            DeclareLaunchArgument(
                "amplitude_rad",
                default_value="0.23",
                description="Peak sine position amplitude in radians",
            ),
            DeclareLaunchArgument(
                "start_frequency_hz",
                default_value="10.0",
                description="Sweep start frequency",
            ),
            DeclareLaunchArgument(
                "end_frequency_hz",
                default_value="20.0",
                description="Sweep end frequency",
            ),
            DeclareLaunchArgument(
                "duration_sec",
                default_value="60.0",
                description="Test duration; 0 means continuous fixed frequency",
            ),
            DeclareLaunchArgument(
                "control_rate_hz",
                default_value="200.0",
                description="Actuator command publishing rate",
            ),
            DeclareLaunchArgument(
                "stop_ramp_sec",
                default_value="0.5",
                description="Smooth return-to-center time after a normal stop",
            ),
            DeclareLaunchArgument(
                "motion_button_mode",
                default_value="toggle",
                description=(
                    "btn_9 source mode: toggle for the C++ gamepad, "
                    "momentary for keyboard-style press/release sources"
                ),
            ),
            DeclareLaunchArgument(
                "joint_test_required",
                default_value=joint_test_required_default,
                description=(
                    "Run the 29-joint rotation precheck before vibration"
                ),
            ),
            DeclareLaunchArgument(
                "allow_hardware_without_joint_test",
                default_value=allow_hardware_without_joint_test_default,
                description="Explicit opt-out used by the combined test mode",
            ),
            DeclareLaunchArgument(
                "joint_test_amplitude_rad",
                default_value="0.03",
                description="Per-joint positive/negative precheck amplitude",
            ),
            DeclareLaunchArgument(
                "joint_test_move_sec",
                default_value="0.4",
                description="Smooth travel time for each precheck waypoint",
            ),
            DeclareLaunchArgument(
                "joint_test_hold_sec",
                default_value="0.1",
                description="Feedback verification hold at each waypoint",
            ),
            DeclareLaunchArgument(
                "joint_test_min_motion_rad",
                default_value="0.015",
                description="Minimum measured motion in each direction",
            ),
            DeclareLaunchArgument(
                "log_rate_hz",
                default_value="100.0",
                description="Asynchronous CSV sampling rate",
            ),
            DeclareLaunchArgument(
                "release_suspension",
                default_value="false",
                description="Release the MuJoCo virtual suspension after reset",
            ),
            DeclareLaunchArgument(
                "auto_start",
                default_value="false",
                description=(
                    "Automatic vibration start; must remain false while the "
                    "joint precheck is required"
                ),
            ),
            DeclareLaunchArgument(
                "log_csv_path",
                default_value="/tmp/elf3_vibration_test.csv",
                description="Command and measured-position CSV output",
            ),
            DeclareLaunchArgument(
                "start_remote_controller",
                default_value="true",
                description=(
                    "Start the C++ gamepad publisher. Set false when an "
                    "external remote_controller is already running"
                ),
            ),
            DeclareLaunchArgument(
                "controller_executable",
                default_value=controller_executable_default,
                description="Internal controller executable selected by wrappers",
            ),
            DeclareLaunchArgument(
                "controller_name",
                default_value=controller_name_default,
                description="Internal controller node name selected by wrappers",
            ),
            RegisterEventHandler(
                OnProcessStart(
                    target_action=simulation_node,
                    on_start=[
                        TimerAction(period=1.0, actions=[vibration_node])
                    ],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=simulation_node,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="MuJoCo simulation exited"))
                    ],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=vibration_node,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(reason="vibration controller exited")
                        )
                    ],
                )
            ),
            simulation_node,
            remote_node,
        ]
    )
