import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription(
        [
            Node(
                package="hardware_arm",
                executable="hardware_arm",
                name="hardware_arm",
                output="screen",
                parameters=[
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
            Node(
                package="bxi_example_arm",
                executable="bxi_example_arm",
                name="bxi_example_arm",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
