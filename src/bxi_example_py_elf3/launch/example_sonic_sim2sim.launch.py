"""Launch ELF3 sim2sim through the BXI Python controller with SONIC enabled."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_path("bxi_example_py_elf3")
    default_mujoco_model_file = os.path.join(
        share, "data/mujoco_simulation/elf3.xml"
    )
    state_machine_config = os.path.join(share, "config/elf3_state_machine.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mujoco_model_file",
                default_value=default_mujoco_model_file,
                description=(
                    "MuJoCo XML used by this simulation-only launch. This "
                    "allows no-gripper/gripper A/B without replacing elf3.xml."
                ),
            ),
            DeclareLaunchArgument(
                "startup_release_delay",
                default_value="1.0",
                description=(
                    "Seconds after reset step1 before the demo may release the "
                    "MuJoCo suspension once the state is allowed."
                ),
            ),
            DeclareLaunchArgument(
                "startup_release_allowed_states",
                default_value="normal",
                description=(
                    "States allowed for startup release. Keep this at normal to "
                    "mirror the real-robot guide: pd_brake -> normal -> sonic_teleop."
                ),
            ),
            DeclareLaunchArgument(
                "state_machine_config",
                default_value=state_machine_config,
                description="BXI state machine config with sonic_teleop state.",
            ),
            DeclareLaunchArgument(
                "sonic_pico_auto_start",
                default_value="true",
                description=(
                    "Start manager+bridge when state_machine_info enters sonic_teleop."
                ),
            ),
            Node(
                package="mujoco",
                executable="simulation",
                name="simulation_mujoco",
                output="screen",
                # The installed simulator treats argv[1] as its XML path.
                # Passing only a ROS parameter makes argv[1] become
                # ``--ros-args`` and the model parse fails.
                arguments=[LaunchConfiguration("mujoco_model_file")],
                emulate_tty=True,
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_demo",
                name="bxi_example_py_elf3_demo",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                    {"/state_machine_config": LaunchConfiguration("state_machine_config")},
                    {"/state_machine_info_topic": "simulation/state_machine_info"},
                    {"/hot_reload": True},
                    {"/startup_release_delay": LaunchConfiguration("startup_release_delay")},
                    {
                        "/startup_release_allowed_states": LaunchConfiguration(
                            "startup_release_allowed_states"
                        )
                    },
                ],
                emulate_tty=True,
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="sonic_pico_runtime_supervisor",
                name="sonic_pico_runtime_supervisor",
                output="screen",
                parameters=[
                    {"state_machine_info_topic": "simulation/state_machine_info"},
                    {"target_state": "sonic_teleop"},
                    {"target_states": ["sonic_teleop", "sonic_teleop_gripper"]},
                    {
                        "enabled": ParameterValue(
                            LaunchConfiguration("sonic_pico_auto_start"),
                            value_type=bool,
                        )
                    },
                ],
                emulate_tty=True,
            ),
        ]
    )
