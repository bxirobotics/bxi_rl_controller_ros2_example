import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    xml_file_name = "model/xml/elf2-top-arm/elf2_top_arm.xml"
    xml_file = os.path.join(get_package_share_path("description"), xml_file_name)

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
                package="bxi_example_py_arm",
                executable="bxi_example_py_arm",
                name="bxi_example_py_arm",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
