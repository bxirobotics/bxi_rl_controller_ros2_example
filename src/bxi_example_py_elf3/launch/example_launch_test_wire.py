"""Launch MuJoCo, suspended running trajectory control and optional gamepad."""

import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_file = os.path.join(
        get_package_share_path("bxi_example_py_elf3"),
        "data/elf3.xml",
    )
    start_remote_controller = LaunchConfiguration("start_remote_controller")

    simulation_node = Node(
        package="mujoco",
        executable="simulation",
        name="simulation_mujoco",
        output="screen",
        parameters=[{"simulation/model_file": model_file}],
        emulate_tty=True,
    )
    run_test_node = Node(
        package="bxi_example_py_elf3",
        executable="bxi_example_py_elf3_test_wire",
        name="bxi_example_py_elf3_test_wire",
        output="screen",
        parameters=[{"/topic_prefix": "simulation/"}],
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
                "start_remote_controller",
                default_value="true",
                description=(
                    "Start the C++ gamepad publisher. Set false when an "
                    "external remote_controller is already running"
                ),
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=simulation_node,
                    on_exit=[EmitEvent(event=Shutdown(reason="MuJoCo exited"))],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=run_test_node,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="run test controller exited"))
                    ],
                )
            ),
            simulation_node,
            run_test_node,
            remote_node,
        ]
    )
