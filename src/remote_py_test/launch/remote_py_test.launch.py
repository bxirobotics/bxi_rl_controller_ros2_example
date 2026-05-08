from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="remote_py_test",
                executable="remote_py_test",
                name="remote_py_test",
                output="screen",
                emulate_tty=True,
                arguments=["__log_level:=debug"],
            ),
        ]
    )
