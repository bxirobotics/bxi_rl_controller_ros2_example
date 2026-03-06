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
                package="remote_py_test",
                executable="remote_py_test",
                name="remote_py_test",
                output="screen",
                # parameters=[
                #     {"/topic_prefix": "simulation/"},
                # ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
