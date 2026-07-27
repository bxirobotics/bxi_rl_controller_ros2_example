"""Launch Elf3 hardware, suspended running test and optional gamepad."""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _require_root(_context):
    if os.geteuid() != 0:
        raise RuntimeError("Elf3 hardware run-test launch must be run as root")
    return []


def generate_launch_description():
    start_remote_controller = LaunchConfiguration("start_remote_controller")

    hardware_node = Node(
        package="hardware_elf3",
        executable="hardware_elf3",
        name="hardware_elf3",
        output="screen",
        parameters=[],
        emulate_tty=True,
    )
    run_test_node = Node(
        package="bxi_example_py_elf3",
        executable="bxi_example_py_elf3_test_wire",
        name="bxi_example_py_elf3_test_wire",
        output="screen",
        parameters=[{"/topic_prefix": "hardware/"}],
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
            OpaqueFunction(function=_require_root),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=hardware_node,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="hardware_elf3 exited"))
                    ],
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
            hardware_node,
            run_test_node,
            remote_node,
        ]
    )
