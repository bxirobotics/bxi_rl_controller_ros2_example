import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    policy_file_name = "policy/policy.jit"
    policy_file = os.path.join(get_package_share_path("bxi_example_py"), policy_file_name)

    return LaunchDescription(
        [
            Node(
                package="hardware",
                executable="hardware",
                name="hardware",
                output="screen",
                parameters=[
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py",
                executable="bxi_example_py",
                name="bxi_example_py",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/policy_file": policy_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
