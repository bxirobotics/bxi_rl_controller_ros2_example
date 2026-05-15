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
                package="bxi_bms",
                executable="bxi_bms",
                name="bxi_bms",
                output="screen",
                # parameters=[
                #     #
                # ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_bms",
                executable="bxi_example_bms",
                name="bxi_example_bms",
                output="screen",
                # parameters=[
                #     #
                # ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
