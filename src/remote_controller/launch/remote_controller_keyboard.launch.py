import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    remote_config = os.path.join(
        get_package_share_path("remote_controller"),
        "config/xbox_default.yaml",
    )
    arguments = [
        "--keyboard",
        "--config", remote_config,
        "__log_level:=debug",
    ]
    debug_value = LaunchConfiguration("DEBUG").perform(context).lower()
    if debug_value in ("true", "1", "yes", "on"):
        arguments.append("--DEBUG")
    return [
        Node(
            package="remote_controller",
            executable="remote_controller",
            name="remote_controller",
            output="screen",
            emulate_tty=True,
            arguments=arguments,
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "DEBUG",
                default_value="false",
                description="Enable periodic input-driver diagnostics.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
