from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="remote_controller",
                executable="remote_controller",
                name="remote_controller",
                output="screen",
                emulate_tty=True,
                arguments=["__log_level:=debug"],
            ),
        ]
    )
