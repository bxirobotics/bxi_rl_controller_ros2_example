import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    xml_file_name = "model/xml/elf2-footv4/elf2_footv4_dof25.xml"
    xml_file = os.path.join(get_package_share_path("description"), xml_file_name)

    policy_file_name = "policy/policy.jit"
    policy_file = os.path.join(get_package_share_path("bxi_example_py_foot"), policy_file_name)

    onnx_file_name = "policy/model_walk.onnx"
    onnx_file = os.path.join(get_package_share_path("bxi_example_py_foot"), onnx_file_name)

    return LaunchDescription(
        [
            Node(
                package="mujoco",
                executable="simulation",
                name="simulation_mujoco",
                output="screen",
                parameters=[
                    {"simulation/model_file": xml_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_foot",
                executable="bxi_example_py_foot_walk",
                name="bxi_example_py_foot_walk",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                    {"/policy_file": policy_file},
                    {"/onnx_file": onnx_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
