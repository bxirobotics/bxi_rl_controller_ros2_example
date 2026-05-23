import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    remote_config = os.path.join(
        get_package_share_path("remote_controller"),
        "config/xbox_default.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="remote_controller",
                executable="remote_controller",
                name="remote_controller",
                output="screen",
                emulate_tty=True,
                arguments=[
                    "--config", remote_config,
                    "--hot-reload", "true",
                    "__log_level:=debug",
                ],
            ),
        ]
    )
