import os
import sys
import fcntl
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

LOCK_FILE = "/tmp/bxi_example_hw.lock"
_lock_fd = None

def _acquire_lock():
    global _lock_fd
    _lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("\n[ERROR] bxi_example_hw is already running! "
              "Please stop the existing instance before starting a new one.\n",
              file=sys.stderr)
        _lock_fd.close()
        _lock_fd = None
        sys.exit(1)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()

def generate_launch_description():
    _acquire_lock()

    xml_file_name = "data/elf3.xml"
    xml_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), xml_file_name)
    state_machine_config = os.path.join(
        get_package_share_path("bxi_example_py_elf3"),
        "config/elf3_state_machine.yaml",
    )

    npz_file_dict = {
        "recover": "data/recover.npz",
        "dance": "data/dance.npz",
    }  
    onnx_file_dict = {
        "normal": "data/amp_terrain.onnx",
        "recover": "data/recover.onnx",
        "dance": "data/dance.onnx",
        "amp_run": "data/amp_run.onnx",
        "normal_run": "data/model_normal.onnx",
        "noarm": "data/arm8.onnx",
    }

    for key, value in npz_file_dict.items():
        npz_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_elf3"), value)
    for key, value in onnx_file_dict.items():
        onnx_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_elf3"), value)

    return LaunchDescription(
        [
            Node(
                package="hardware_elf3",
                executable="hardware_elf3",
                name="hardware_elf3",
                output="screen",
                parameters=[
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_demo",
                name="bxi_example_py_elf3_demo",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/npz_file_dict": json.dumps(npz_file_dict)},
                    {"/onnx_file_dict": json.dumps(onnx_file_dict)},
                    {"/state_machine_config": state_machine_config},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
