import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    policy_file_name = "policy/policy.jit"
    policy_file = os.path.join(get_package_share_path("bxi_example_py_armrc"), policy_file_name)

    onnx_file_name = "policy/model.onnx"
    onnx_file = os.path.join(get_package_share_path("bxi_example_py_armrc"), onnx_file_name)

    return LaunchDescription(
        [
            Node(
                package="hardware_hand",
                executable="hardware_hand",
                name="hardware_hand",
                output="screen",
                parameters=[
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_armrc",
                executable="bxi_example_py_armrc",
                name="bxi_example_py_armrc",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/policy_file": policy_file},
                    {"/onnx_file": onnx_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
