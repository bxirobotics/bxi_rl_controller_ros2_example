import os
import sys
import fcntl
import atexit
import stat
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

LOCK_FILE = "/tmp/bxi_example_hw.lock"
_lock_fd = None


def _release_lock():
    global _lock_fd
    if _lock_fd is None:
        return
    fd = _lock_fd
    _lock_fd = None
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _acquire_lock():
    global _lock_fd
    if _lock_fd is not None:
        return

    # Create the lock file world-writable so a different user (or root)
    # can reuse the same system-wide lock. Force umask to 0 so the 0o666
    # mode is honored on creation regardless of the caller's umask.
    old_umask = os.umask(0)
    try:
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(LOCK_FILE, flags, 0o666)
        except PermissionError as e:
            print(
                f"\n[ERROR] Cannot open lock file {LOCK_FILE}: {e}\n"
                f"        A stale lock file with restrictive permissions exists.\n"
                f"        Remove it and retry:  sudo rm {LOCK_FILE}\n",
                file=sys.stderr,
            )
            sys.exit(1)
        except OSError as e:
            print(
                f"\n[ERROR] Cannot safely open lock file {LOCK_FILE}: {e}\n"
                "        The path must be a regular file, not a symlink.\n",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        os.umask(old_umask)

    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        print(
            f"\n[ERROR] Lock path {LOCK_FILE} is not a regular file.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Repair perms through the already validated descriptor.  Using fchmod
    # avoids following a replacement pathname between validation and chmod.
    try:
        os.fchmod(fd, 0o666)
    except OSError:
        pass

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            holder = os.read(fd, 64).decode("utf-8", errors="replace").strip() or "unknown"
        except OSError:
            holder = "unknown"
        os.close(fd)
        print(
            f"\n[ERROR] bxi_example_hw is already running (pid={holder})! "
            f"Please stop the existing instance before starting a new one.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    _lock_fd = fd
    atexit.register(_release_lock)


def _default_sonic_pico_python():
    env_python = os.environ.get("SONIC_PICO_PYTHON")
    if env_python:
        return env_python
    for candidate in (
        "/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python",
        "/home/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python",
        "/home/bxi/bxi_ws/bxi_rl_controller_ros2_example/.venv_teleop/bin/python",
        "/opt/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python",
    ):
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return sys.executable


def generate_launch_description():
    _acquire_lock()

    state_machine_config = os.path.join(
        get_package_share_path("bxi_example_py_elf3"),
        "config/elf3_state_machine.yaml",
    )
    state_machine_info_topic = "hardware/state_machine_info"

    hardware_node = Node(
        package="hardware_elf3",
        executable="hardware_elf3",
        name="hardware_elf3",
        output="screen",
        parameters=[
            {"hardware_config/imu": True},      #start imu
            {"hardware_config/motor_pwr": True}, #motor poweron
            {"hardware_config/motor_disable": 0x60000000}, #motor disable head
        ],
        emulate_tty=True,
        arguments=[("__log_level:=debug")],
    )
    controller_node = Node(
        package="bxi_example_py_elf3",
        executable="bxi_example_py_elf3_demo",
        name="bxi_example_py_elf3_demo",
        output="screen",
        parameters=[
            {"/topic_prefix": "hardware/"},
            {"/state_machine_config": state_machine_config},
            {"/state_machine_info_topic": state_machine_info_topic},
            {"/hot_reload": False},
        ],
        emulate_tty=True,
        arguments=[("__log_level:=debug")],
    )
    pico_supervisor_node = Node(
        package="bxi_example_py_elf3",
        executable="sonic_pico_runtime_supervisor",
        name="sonic_pico_runtime_supervisor",
        output="screen",
        # PicoPipeline.close() deliberately gives the manager/bridge process
        # groups up to 7.5 s to stop.  The launch default would escalate from
        # SIGINT to SIGTERM after about 5 s and could interrupt that cleanup,
        # leaving a worker behind.  Keep the launch process alive long enough
        # for the supervisor's bounded shutdown path to finish.
        sigterm_timeout="10",
        sigkill_timeout="5",
        parameters=[
            {"state_machine_info_topic": state_machine_info_topic},
            {"target_state": "sonic_teleop"},
            {"target_states": ["sonic_teleop", "sonic_teleop_gripper"]},
            {
                "enabled": ParameterValue(
                    LaunchConfiguration("sonic_pico_auto_start"),
                    value_type=bool,
                )
            },
            {"python_executable": LaunchConfiguration("sonic_pico_python")},
        ],
        # Hardware publishes the PICO trigger topics for the explicit gripper
        # state.  The body-only state has no CAN publisher, so these topics are
        # inert unless the operator selects SONIC遥操（夹爪）.
        additional_env={"PICO_ENABLE_ROS_BUTTONS": "1"},
        emulate_tty=True,
    )

    # The App stop command terminates the hardware/controller processes.  The
    # PICO supervisor is a third long-lived node, so without an explicit
    # launch shutdown it keeps ros2 launch (and the single-instance lock)
    # alive.  Treat either critical control-node exit as the end of the whole
    # hardware session so all auxiliary processes are cleaned up together.
    shutdown_on_hardware_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=hardware_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="hardware_elf3 exited")
                )
            ],
        )
    )
    shutdown_on_controller_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=controller_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="bxi_example_py_elf3_demo exited")
                )
            ],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sonic_pico_auto_start",
                default_value="true",
                description=(
                    "Start manager+bridge automatically when the state machine "
                    "enters either SONIC mode."
                ),
            ),
            DeclareLaunchArgument(
                "sonic_pico_python",
                default_value=_default_sonic_pico_python(),
                description=(
                    "Python interpreter that contains XRoboToolkit/torch/zmq "
                    "for the robot-side PICO runtime."
                ),
            ),
            shutdown_on_hardware_exit,
            shutdown_on_controller_exit,
            hardware_node,
            controller_node,
            pico_supervisor_node,
        ]
    )
