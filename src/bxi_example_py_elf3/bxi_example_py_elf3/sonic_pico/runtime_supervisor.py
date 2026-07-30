"""Start the PICO runtime only while the ELF3 state machine needs SONIC."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


START_FAILURE_SIGINT_GRACE = 3.0
START_FAILURE_SIGTERM_GRACE = 2.0
START_FAILURE_SIGKILL_GRACE = 1.0
STOP_POLL_INTERVAL = 0.05


def default_pico_python_executable() -> str:
    """Pick the Python interpreter that has the robot-side PICO dependencies."""
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


def env_flag_enabled(env: dict[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def prepend_existing_ld_paths(env: dict[str, str], paths: list[str]) -> list[str]:
    """Prepend existing native-library paths to LD_LIBRARY_PATH.

    The xrobotoolkit_sdk Python extension is installed in the PICO venv, but it
    depends on libPXREARobotSDK.so from the RoboticsService installation.  That
    dependency must be discoverable before importing the Python extension, so
    the child process environment has to be fixed by the supervisor.
    """
    old_paths = [path for path in env.get("LD_LIBRARY_PATH", "").split(":") if path]
    added: list[str] = []
    for path in paths:
        if path and os.path.isdir(path) and path not in old_paths and path not in added:
            added.append(path)
    if added:
        env["LD_LIBRARY_PATH"] = ":".join(added + old_paths)
    return added


def _target_names(target: str | tuple[str, ...] | list[str]) -> set[str]:
    if isinstance(target, str):
        return {target} if target else set()
    return {str(name) for name in target if str(name)}


def state_requests_sonic(
    info: Any,
    target: str | tuple[str, ...] | list[str] = "sonic_teleop",
) -> bool:
    """Return the desired runtime state from a state-machine snapshot."""
    if not isinstance(info, dict):
        return False
    targets = _target_names(target)
    transition = info.get("transition")
    if transition is not None:
        if not isinstance(transition, dict):
            return False
        destination = transition.get("to")
        return isinstance(destination, dict) and destination.get("name") in targets
    current = info.get("current")
    return isinstance(current, dict) and current.get("name") in targets


@dataclass
class StateSnapshotMonitor:
    """Parse state snapshots and fail closed when their heartbeat expires."""

    target: str | tuple[str, ...] | list[str] = "sonic_teleop"
    enabled: bool = True
    heartbeat_timeout: float = 1.0
    desired: bool = False
    received_at: float | None = None

    def update(self, payload: str, received_at: float) -> str | None:
        """Consume one JSON snapshot, returning an error message when invalid."""
        self.received_at = received_at
        try:
            info = json.loads(payload)
            if not isinstance(info, dict):
                raise ValueError("state-machine snapshot must be a JSON object")
        except (TypeError, ValueError) as exc:
            self.desired = False
            return str(exc)

        self.desired = self.enabled and state_requests_sonic(info, self.target)
        return None

    def active(self, now: float) -> bool:
        if not self.desired or self.received_at is None:
            return False
        return now - self.received_at <= self.heartbeat_timeout


@dataclass
class ChildProcess:
    name: str
    process: subprocess.Popen
    # Every child is created with start_new_session=True, so its PID is also
    # the process-group ID.  Keep that ID after the leader exits: the manager
    # and bridge may leave worker processes alive in their respective groups.
    pgid: int | None = None
    group_gone: bool = False

    def __post_init__(self) -> None:
        if self.pgid is None:
            self.pgid = int(self.process.pid)


class PicoPipeline:
    """Non-blocking owner for the manager and bridge process groups."""

    def __init__(self, logger, python_executable: str):
        self.logger = logger
        self.python_executable = python_executable
        self.children: list[ChildProcess] = []
        self.stop_started_at: float | None = None
        self.stop_stage = 0
        self.supervisor_pgid = os.getpgrp()

    @property
    def running(self) -> bool:
        return bool(self.children) and self.stop_started_at is None

    @property
    def stopping(self) -> bool:
        return self.stop_started_at is not None

    def start(self) -> None:
        if self.children:
            return
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        xrt_service_dir = env.get("SONIC_XRT_SERVICE_DIR", "/opt/apps/roboticsservice")
        xrt_ld_paths = prepend_existing_ld_paths(
            env,
            [
                os.path.join(xrt_service_dir, "SDK", "x64"),
                xrt_service_dir,
                os.path.join(xrt_service_dir, "lib"),
            ],
        )
        pico_port = env.get("PICO_PORT", "5556")
        out_host = env.get(
            "BXI_SONIC_SMPL_REF_ZMQ_HOST",
            env.get("SMPL_REF_ZMQ_HOST", "127.0.0.1"),
        )
        out_port = env.get(
            "BXI_SONIC_SMPL_REF_ZMQ_PORT",
            env.get("SMPL_REF_ZMQ_PORT", "5557"),
        )
        out_topic = env.get(
            "BXI_SONIC_SMPL_REF_ZMQ_TOPIC",
            env.get("SMPL_REF_ZMQ_TOPIC", "smpl_ref"),
        )
        manager = [
            self.python_executable,
            "-m",
            "bxi_example_py_elf3.sonic_pico.pico_manager_legacy",
            "--manager",
            "--num_frames_to_send",
            "10",
            "--target_fps",
            "50",
            "--port",
            pico_port,
        ]
        if env_flag_enabled(env, "SONIC_PICO_USE_CUDA", default=False):
            manager.append("--cuda")
        bridge = [
            self.python_executable,
            "-m",
            "bxi_example_py_elf3.sonic_pico.pico_pose_to_smpl_ref_bridge",
            "--pico-host",
            env.get("PICO_HOST", "127.0.0.1"),
            "--pico-port",
            pico_port,
            "--pico-topic",
            "pose",
            "--out-host",
            out_host,
            "--out-port",
            out_port,
            "--out-topic",
            out_topic,
            "--stale-warning-seconds",
            env.get("SONIC_PICO_STALE_SECONDS", "0.2"),
        ]
        if env.get("PICO_ENABLE_ROS_BUTTONS", "0").lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            bridge.append("--disable-ros-pico-topics")

        self.logger.info("starting SONIC PICO manager and bridge")
        if xrt_ld_paths:
            self.logger.info(
                "SONIC PICO native library path includes: " + ", ".join(xrt_ld_paths)
            )
        self.stop_started_at = None
        self.stop_stage = 0
        try:
            manager_proc = subprocess.Popen(manager, env=env, start_new_session=True)
            self.children = [ChildProcess("manager", manager_proc)]
            bridge_proc = subprocess.Popen(bridge, env=env, start_new_session=True)
            self.children.append(ChildProcess("bridge", bridge_proc))
        except BaseException as exc:
            if self.children:
                try:
                    self._cleanup_failed_start(
                        f"pipeline start failed ({type(exc).__name__})"
                    )
                except BaseException as cleanup_exc:
                    self.logger.error(
                        "failed while cleaning up a partially started SONIC PICO "
                        f"runtime: {cleanup_exc}"
                    )
            raise

    def request_stop(self, reason: str) -> None:
        if not self.children or self.stopping:
            return
        self.logger.info(f"stopping SONIC PICO runtime: {reason}")
        self.stop_started_at = time.monotonic()
        self.stop_stage = 1
        self._signal(signal.SIGINT, process_group=True)

    def _safe_pgid(self, child: ChildProcess) -> int | None:
        pgid = child.pgid
        if pgid is None or pgid <= 1 or pgid == self.supervisor_pgid:
            # Never send a group signal when a bad PGID could include this
            # supervisor (and, on ROS launch, unrelated controller nodes).
            self.logger.error(
                f"refusing unsafe process-group signal for {child.name}: pgid={pgid}"
            )
            return None
        return pgid

    def _group_alive(self, child: ChildProcess) -> bool:
        if child.group_gone:
            return False
        pgid = self._safe_pgid(child)
        if pgid is None:
            # We cannot prove that descendants are gone.  Retain ownership and
            # prevent a second pipeline from starting under this unsafe state.
            return True
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            # Once absence is observed, never signal this numeric PGID again:
            # it could eventually be reused by an unrelated process group.
            child.group_gone = True
            return False
        except PermissionError:
            # The group exists even if the supervisor unexpectedly cannot
            # signal it.  Keep ownership instead of incorrectly declaring it
            # stopped and launching a duplicate pipeline.
            return True

    def _child_stopped(self, child: ChildProcess) -> bool:
        leader_exited = child.process.poll() is not None
        return leader_exited and not self._group_alive(child)

    def _all_children_stopped(self) -> bool:
        return bool(self.children) and all(
            self._child_stopped(child) for child in self.children
        )

    def _signal(self, signum: int, process_group: bool) -> None:
        for child in self.children:
            try:
                if process_group:
                    if child.group_gone:
                        continue
                    pgid = self._safe_pgid(child)
                    if pgid is not None:
                        # Deliberately signal the saved PGID even when the
                        # Popen leader has exited; workers can still be alive.
                        os.killpg(pgid, signum)
                    elif child.process.poll() is None:
                        child.process.send_signal(signum)
                elif child.process.poll() is None:
                    child.process.send_signal(signum)
            except ProcessLookupError:
                if process_group:
                    child.group_gone = True
            except PermissionError:
                self.logger.error(
                    f"permission denied signalling {child.name} process group"
                )

    def _wait_for_all_stopped(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self._all_children_stopped():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(STOP_POLL_INTERVAL, remaining))

    def _reap_leaders(self, timeout: float = 0.0) -> None:
        for child in self.children:
            try:
                child.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    def _finish_stop_if_complete(self) -> str | None:
        if not self._all_children_stopped():
            return None
        self._reap_leaders()
        result = ", ".join(
            f"{child.name}={child.process.returncode}" for child in self.children
        )
        self.logger.info(f"SONIC PICO runtime stopped ({result})")
        self.children = []
        self.stop_started_at = None
        self.stop_stage = 0
        return result

    def _cleanup_failed_start(self, reason: str) -> None:
        """Synchronously reclaim a manager when the bridge failed to spawn."""
        self.request_stop(reason)
        if not self._wait_for_all_stopped(START_FAILURE_SIGINT_GRACE):
            self.stop_stage = 2
            self._signal(signal.SIGTERM, process_group=True)
        if not self._wait_for_all_stopped(START_FAILURE_SIGTERM_GRACE):
            self.stop_stage = 3
            self._signal(signal.SIGKILL, process_group=True)
        self._wait_for_all_stopped(START_FAILURE_SIGKILL_GRACE)
        # Explicit wait reaps the direct child.  Group liveness is checked
        # separately above because waiting for the leader alone is insufficient.
        self._reap_leaders(timeout=0.0)
        if self._finish_stop_if_complete() is None:
            self.logger.error(
                "partially started SONIC PICO runtime still has live process groups "
                "after SIGKILL; retaining ownership for subsequent cleanup"
            )

    def poll(self) -> str | None:
        if not self.children:
            return None
        exited = [
            child for child in self.children
            if child.process.poll() is not None
        ]
        if exited and not self.stopping:
            detail = ", ".join(f"{c.name}={c.process.returncode}" for c in exited)
            self.request_stop(f"child exited ({detail})")

        if self.stopping:
            elapsed = time.monotonic() - float(self.stop_started_at)
            if elapsed >= 3.0 and self.stop_stage == 1:
                self.stop_stage = 2
                self._signal(signal.SIGTERM, process_group=True)
            if elapsed >= 5.0 and self.stop_stage == 2:
                self.stop_stage = 3
                self._signal(signal.SIGKILL, process_group=True)

        return self._finish_stop_if_complete()

    def close(self) -> None:
        self.request_stop("supervisor shutdown")
        deadline = time.monotonic() + 6.5
        while self.children and time.monotonic() < deadline:
            self.poll()
            time.sleep(0.05)
        if self.children:
            self._signal(signal.SIGKILL, process_group=True)
            self._wait_for_all_stopped(1.0)
            self._reap_leaders(timeout=0.0)
            if self._finish_stop_if_complete() is None:
                self.logger.error(
                    "SONIC PICO runtime still has live process groups during shutdown"
                )


class SonicPicoRuntimeSupervisor(Node):
    def __init__(self):
        super().__init__("sonic_pico_runtime_supervisor")
        self.declare_parameter("state_machine_info_topic", "hardware/state_machine_info")
        self.declare_parameter("target_state", "sonic_teleop")
        self.declare_parameter("target_states", ["sonic_teleop"])
        self.declare_parameter("enabled", True)
        self.declare_parameter("heartbeat_timeout", 1.0)
        self.declare_parameter("restart_delay", 3.0)
        self.declare_parameter(
            "python_executable",
            default_pico_python_executable(),
        )
        self.topic = str(self.get_parameter("state_machine_info_topic").value)
        self.target_state = str(self.get_parameter("target_state").value)
        configured_targets = [
            str(name) for name in self.get_parameter("target_states").value
        ]
        if self.target_state and self.target_state not in configured_targets:
            configured_targets.append(self.target_state)
        self.target_states = tuple(configured_targets)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.heartbeat_timeout = float(self.get_parameter("heartbeat_timeout").value)
        self.restart_delay = float(self.get_parameter("restart_delay").value)
        python_executable = str(self.get_parameter("python_executable").value)
        self.pipeline = PicoPipeline(self.get_logger(), python_executable)
        self.state_snapshot = StateSnapshotMonitor(
            target=self.target_states,
            enabled=self.enabled,
            heartbeat_timeout=self.heartbeat_timeout,
        )
        self.desired = False
        self.last_state_message: float | None = None
        self.last_stop_time = 0.0
        self.subscription = self.create_subscription(String, self.topic, self._on_state, 10)
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"watching {self.topic} for states={','.join(self.target_states)}, "
            f"enabled={self.enabled}"
        )

    def _on_state(self, msg: String) -> None:
        now = time.monotonic()
        error = self.state_snapshot.update(msg.data, now)
        self.last_state_message = self.state_snapshot.received_at
        self.desired = self.state_snapshot.active(now)
        if error is not None:
            self.get_logger().warning(f"invalid state-machine snapshot: {error}")

    def _tick(self) -> None:
        now = time.monotonic()
        self.desired = self.state_snapshot.active(now)
        was_stopping = self.pipeline.stopping
        stopped = self.pipeline.poll()
        if stopped is not None or (was_stopping and not self.pipeline.children):
            self.last_stop_time = now
        if self.desired:
            if (
                not self.pipeline.children
                and now - self.last_stop_time >= self.restart_delay
            ):
                try:
                    self.pipeline.start()
                except Exception as exc:
                    self.last_stop_time = now
                    self.get_logger().error(f"failed to start SONIC PICO runtime: {exc}")
        else:
            self.pipeline.request_stop("SONIC state inactive or heartbeat lost")

    def destroy_node(self):
        self.pipeline.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SonicPicoRuntimeSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
