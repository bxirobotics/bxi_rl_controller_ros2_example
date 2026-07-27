"""Suspended Elf3 running/loom test using a recorded 29-joint trajectory."""

import math
import time
from pathlib import Path
from threading import RLock

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from std_srvs.srv import SetBool

import communication.msg as bxiMsg
import communication.srv as bxiSrv
import sensor_msgs.msg

from .control.elf3 import (
    DOF_NUM,
    JOINT_KD,
    JOINT_NAMES,
    ROBOT_NAME,
    SUSPENDED_RUN_NOMINAL_POS,
    position_limit_violations,
    validate_joint_vector,
)
from .control.remote import RemoteButtonEdge
from .control.trajectory import load_joint_trajectory


# Deliberately preserved from the current debugging setup. This is far above
# the normal Elf3 gains and is reported loudly at startup, but is not silently
# changed by this structural refactor.
TEST_WIRE_KP = np.full(DOF_NUM, 10000.0, dtype=np.float64)


class SuspendedRunTestNode(Node):
    """Initialize Elf3, hold suspension and play a validated joint trajectory."""

    def __init__(self):
        super().__init__("bxi_example_py_elf3_test_wire")

        # Keep the legacy parameter name so existing commands and launch files
        # remain compatible.
        self.topic_prefix = str(
            self.declare_parameter("/topic_prefix", "simulation/").value
        )
        self.hardware_mode = self.topic_prefix.startswith("hardware/")
        self.control_rate_hz = float(
            self.declare_parameter("control_rate_hz", 50.0).value
        )
        self.initialization_sec = float(
            self.declare_parameter("initialization_sec", 2.0).value
        )
        self.release_suspension = bool(
            self.declare_parameter("release_suspension", False).value
        )
        self.motion_button_mode = str(
            self.declare_parameter("motion_button_mode", "toggle").value
        ).strip().lower()
        self.motion_command_resync_sec = float(
            self.declare_parameter("motion_command_resync_sec", 0.5).value
        )
        self.joint_state_timeout_sec = float(
            self.declare_parameter("joint_state_timeout_sec", 0.2).value
        )
        self.max_command_gap_sec = float(
            self.declare_parameter("max_command_gap_sec", 0.05).value
        )
        self.require_joint_state = bool(
            self.declare_parameter(
                "require_joint_state", self.hardware_mode
            ).value
        )
        requested_path = str(
            self.declare_parameter("trajectory_path", "").value
        ).strip()
        self._validate_parameters()

        self.remote_button = RemoteButtonEdge(
            self.motion_button_mode,
            self.motion_command_resync_sec,
        )
        trajectory_path = self._resolve_trajectory_path(requested_path)
        self.trajectory = load_joint_trajectory(
            trajectory_path,
            self.control_rate_hz,
        )

        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self.actuator_topic = self.topic_prefix + "actuators_cmds"
        self.motion_topic = "motion_commands"
        self.actuator_pub = self.create_publisher(
            bxiMsg.ActuatorCmds,
            self.actuator_topic,
            qos,
        )
        self.joint_sub = self.create_subscription(
            sensor_msgs.msg.JointState,
            self.topic_prefix + "joint_states",
            self._joint_callback,
            qos,
        )
        self.motion_sub = self.create_subscription(
            bxiMsg.MotionCommands,
            self.motion_topic,
            self._motion_callback,
            qos,
        )
        self.reset_client = self.create_client(
            bxiSrv.RobotReset,
            self.topic_prefix + "robot_reset",
        )
        self.enable_service = self.create_service(
            SetBool,
            "run_trajectory_enable",
            self._enable_service_callback,
        )

        self.state_lock = RLock()
        self.measured_positions = SUSPENDED_RUN_NOMINAL_POS.copy()
        self.joint_state_last_seen_at = np.zeros(DOF_NUM, dtype=np.float64)
        self.reset_stage = 0
        self.reset_future = None
        self.reset_pending_step = 0
        self.reset_request_sent_at = 0.0
        self.reset_retry_after_at = 0.0
        self.reset_response_timeout_sec = 5.0
        self.initialization_started_at = 0.0
        self.playing = False
        self.safety_fault = False
        self.last_command_publish_at = 0.0
        self.last_reset_wait_log_at = 0.0
        self.last_remote_conflict_log_at = 0.0

        self.control_group = MutuallyExclusiveCallbackGroup()
        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self._timer_callback,
            callback_group=self.control_group,
        )
        self.watchdog_group = MutuallyExclusiveCallbackGroup()
        self.publisher_watchdog = self.create_timer(
            0.5,
            self._publisher_watchdog_callback,
            callback_group=self.watchdog_group,
        )

        diagnostics = self.trajectory.diagnostics()
        self.get_logger().info(
            "loaded %d trajectory frames from %s (%.3f seconds at %.1f Hz)"
            % (
                diagnostics.frame_count,
                self.trajectory.source_path,
                diagnostics.frame_count / self.control_rate_hz,
                self.control_rate_hz,
            )
        )
        self.get_logger().warning(
            "debug gains preserved: every joint Kp is %.1f; verify the robot "
            "is securely suspended before enabling trajectory playback"
            % TEST_WIRE_KP[0]
        )
        self.get_logger().warning(
            "trajectory dynamics: max adjacent change %.6f rad on %s "
            "(%.6f rad/s at %.1f Hz); loop change %.6f rad on %s "
            "(%.6f rad/s)"
            % (
                diagnostics.max_step_delta_rad,
                diagnostics.max_step_joint,
                diagnostics.max_step_velocity_rad_s,
                self.control_rate_hz,
                diagnostics.loop_delta_rad,
                diagnostics.loop_delta_joint,
                diagnostics.loop_velocity_rad_s,
            )
        )
        self.get_logger().info(
            "remote ready: press X once to play/pause after initialization; "
            "service /run_trajectory_enable provides the same control"
        )

    def _validate_parameters(self):
        positive = (
            ("control_rate_hz", self.control_rate_hz),
            ("initialization_sec", self.initialization_sec),
            ("motion_command_resync_sec", self.motion_command_resync_sec),
            ("joint_state_timeout_sec", self.joint_state_timeout_sec),
            ("max_command_gap_sec", self.max_command_gap_sec),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and > 0" % name)
        if self.motion_button_mode not in RemoteButtonEdge.MODES:
            raise ValueError("motion_button_mode must be 'toggle' or 'momentary'")
        if self.max_command_gap_sec < 1.0 / self.control_rate_hz:
            raise ValueError(
                "max_command_gap_sec must be at least one control period"
            )

    @staticmethod
    def _resolve_trajectory_path(requested_path):
        if requested_path:
            return Path(requested_path).expanduser()
        workspace_path = Path(
            "src/bxi_example_py_elf3/data/data.txt"
        )
        if workspace_path.is_file():
            return workspace_path
        return get_package_share_path("bxi_example_py_elf3") / "data/data.txt"

    def _timer_callback(self):
        now = time.monotonic()
        with self.state_lock:
            if self.safety_fault or not self._command_gap_is_safe(now):
                return

            if self.reset_stage == 0:
                if self.reset_pending_step == 1:
                    if self._reset_request_completed(1, now):
                        self.initialization_started_at = time.monotonic()
                        self.reset_stage = 1
                        self.get_logger().info(
                            "robot reset 1 acknowledged; soft initialization started"
                        )
                elif self._call_robot_reset(1, False, now):
                    self.get_logger().info("robot reset 1 sent")
                return

            if self.reset_stage == 1:
                elapsed = now - self.initialization_started_at
                ramp = min(max(elapsed / self.initialization_sec, 0.0), 1.0)
                if not self._publish_command(
                    SUSPENDED_RUN_NOMINAL_POS,
                    TEST_WIRE_KP * ramp,
                    JOINT_KD,
                ):
                    return
                if elapsed >= self.initialization_sec:
                    if self.reset_pending_step == 2:
                        if self._reset_request_completed(2, now):
                            self.reset_stage = 2
                            self.get_logger().info(
                                "robot reset 2 acknowledged; initialization complete"
                            )
                    elif self._call_robot_reset(
                        2,
                        self.release_suspension,
                        now,
                    ):
                        self.get_logger().info(
                            "robot reset 2 sent (release_suspension=%s)"
                            % self.release_suspension
                        )
                return

            if (
                self.playing
                and self.require_joint_state
                and not self._joint_feedback_ready(now)
            ):
                self._latch_safety_fault(
                    "joint feedback became incomplete or stale during playback"
                )
                return

            command = (
                self.trajectory.next()
                if self.playing
                else SUSPENDED_RUN_NOMINAL_POS
            )
            self._publish_command(command, TEST_WIRE_KP, JOINT_KD)

    def _joint_callback(self, msg):
        if not msg.position:
            return
        updated = self.measured_positions.copy()
        seen = np.zeros(DOF_NUM, dtype=bool)
        if msg.name:
            positions_by_name = dict(zip(msg.name, msg.position))
            for index, name in enumerate(JOINT_NAMES):
                if name in positions_by_name:
                    value = float(positions_by_name[name])
                    if not math.isfinite(value):
                        self._latch_safety_fault(
                            "non-finite joint feedback received for %s" % name
                        )
                        return
                    updated[index] = value
                    seen[index] = True
        elif len(msg.position) >= DOF_NUM:
            received = np.asarray(msg.position[:DOF_NUM], dtype=np.float64)
            if not np.all(np.isfinite(received)):
                self._latch_safety_fault("non-finite joint feedback received")
                return
            updated[:] = received
            seen[:] = True
        else:
            return
        if not np.any(seen):
            return
        now = time.monotonic()
        with self.state_lock:
            self.measured_positions[seen] = updated[seen]
            self.joint_state_last_seen_at[seen] = now

    def _motion_callback(self, msg):
        now = time.monotonic()
        if self.count_publishers(self.motion_topic) > 1:
            if now - self.last_remote_conflict_log_at >= 1.0:
                self.get_logger().error(
                    "multiple motion_commands publishers detected; remote "
                    "input is ignored. Stop duplicate remote controllers and "
                    "use /run_trajectory_enable if needed"
                )
                self.last_remote_conflict_log_at = now
            return
        with self.state_lock:
            activated = self.remote_button.update(msg.btn_9 != 0, now)
            target_enabled = not self.playing
        if activated:
            success, message = self._set_playback(
                target_enabled,
                "remote X button",
            )
            if not success:
                self.get_logger().warning("remote command rejected: %s" % message)

    def _enable_service_callback(self, request, response):
        response.success, response.message = self._set_playback(
            bool(request.data),
            "enable service",
        )
        return response

    def _set_playback(self, enabled, source):
        now = time.monotonic()
        with self.state_lock:
            if enabled:
                if self.safety_fault:
                    return False, "latched safety fault; restart required"
                if self.reset_stage != 2:
                    return False, "robot initialization is not complete"
                if self.require_joint_state and not self._joint_feedback_ready(now):
                    return False, "complete fresh joint feedback is required"
                if self._actuator_publisher_count() > 1:
                    self._latch_safety_fault(
                        "multiple actuator command publishers detected"
                    )
                    return False, "multiple actuator command publishers detected"
                if self.playing:
                    return True, "trajectory playback is already active"
                self.playing = True
                self.get_logger().info(
                    "trajectory playback started by %s at frame %d"
                    % (source, self.trajectory.index + 1)
                )
                return True, "trajectory playback started"

            if not self.playing:
                return True, "trajectory playback is already paused"
            self.playing = False
            self.get_logger().info(
                "trajectory playback paused by %s; holding nominal position"
                % source
            )
            return True, "trajectory playback paused"

    def _joint_feedback_ready(self, now):
        return bool(
            np.all(self.joint_state_last_seen_at > 0.0)
            and np.all(
                now - self.joint_state_last_seen_at
                <= self.joint_state_timeout_sec
            )
        )

    def _publish_command(self, positions, kp, kd):
        if self.safety_fault:
            return False
        try:
            positions = validate_joint_vector("positions", positions)
            kp = validate_joint_vector("kp", kp)
            kd = validate_joint_vector("kd", kd)
        except ValueError as exc:
            self._latch_safety_fault(str(exc))
            return False
        violations = position_limit_violations(positions)
        if violations:
            self._latch_safety_fault(
                "refusing command outside software position limits: "
                + "; ".join(violations)
            )
            return False
        now = time.monotonic()
        if not self._command_gap_is_safe(now):
            return False
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = ROBOT_NAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = list(JOINT_NAMES)
        msg.pos = positions.tolist()
        msg.vel = [0.0] * DOF_NUM
        msg.torque = [0.0] * DOF_NUM
        msg.kp = kp.tolist()
        msg.kd = kd.tolist()
        self.actuator_pub.publish(msg)
        self.last_command_publish_at = time.monotonic()
        return True

    def _command_gap_is_safe(self, now):
        if not self.hardware_mode or self.last_command_publish_at <= 0.0:
            return True
        gap = now - self.last_command_publish_at
        if gap <= self.max_command_gap_sec:
            return True
        self._latch_safety_fault(
            "command publish gap %.6f seconds exceeded %.6f seconds"
            % (gap, self.max_command_gap_sec)
        )
        return False

    def _actuator_publisher_count(self):
        try:
            return self.count_publishers(self.actuator_topic)
        except Exception as exc:
            self.get_logger().warning(
                "cannot inspect actuator publisher count: %s" % exc
            )
            return 1

    def _publisher_watchdog_callback(self):
        with self.state_lock:
            if self.safety_fault:
                return
        count = self._actuator_publisher_count()
        if count > 1:
            self._latch_safety_fault(
                "multiple publishers detected on %s (count=%d)"
                % (self.actuator_topic, count)
            )

    def _call_robot_reset(self, reset_step, release, now):
        if self.reset_pending_step or now < self.reset_retry_after_at:
            return False
        if not self.reset_client.service_is_ready():
            if now - self.last_reset_wait_log_at >= 1.0:
                self.get_logger().info("robot_reset service unavailable; waiting")
                self.last_reset_wait_log_at = now
            return False
        request = bxiSrv.RobotReset.Request()
        request.header.frame_id = ROBOT_NAME
        request.reset_step = reset_step
        request.release = release
        try:
            self.reset_future = self.reset_client.call_async(request)
        except Exception as exc:
            self.get_logger().error(
                "failed to send robot reset %d: %s" % (reset_step, exc)
            )
            self.reset_retry_after_at = now + 1.0
            return False
        self.reset_pending_step = reset_step
        self.reset_request_sent_at = now
        return True

    def _reset_request_completed(self, expected_step, now):
        if self.reset_pending_step != expected_step or self.reset_future is None:
            return False
        if not self.reset_future.done():
            if now - self.reset_request_sent_at <= self.reset_response_timeout_sec:
                return False
            self.reset_future.cancel()
            self.get_logger().error(
                "robot reset %d response timed out; retrying" % expected_step
            )
            self._clear_reset_request(now + 1.0)
            return False
        try:
            response = self.reset_future.result()
        except Exception as exc:
            self.get_logger().error(
                "robot reset %d failed: %s; retrying" % (expected_step, exc)
            )
            self._clear_reset_request(now + 1.0)
            return False
        if response is None or not response.is_success:
            self.get_logger().error(
                "robot reset %d was rejected; retrying" % expected_step
            )
            self._clear_reset_request(now + 1.0)
            return False
        self._clear_reset_request(0.0)
        return True

    def _clear_reset_request(self, retry_after):
        self.reset_future = None
        self.reset_pending_step = 0
        self.reset_retry_after_at = retry_after

    def _latch_safety_fault(self, reason):
        with self.state_lock:
            if self.safety_fault:
                return
            self.safety_fault = True
            self.playing = False
        self.get_logger().fatal(
            "SAFETY FAULT: %s; actuator command publishing stopped, restart required"
            % reason
        )
        if self.hardware_mode:
            rclpy.try_shutdown(context=self.context)

    def destroy_node(self):
        try:
            self.control_timer.cancel()
            self.publisher_watchdog.cancel()
        finally:
            super().destroy_node()


# Backward-compatible class name for code importing the original example.
BxiExample = SuspendedRunTestNode


def main(args=None):
    # Preserve the existing startup delay used by the original test-wire node.
    time.sleep(5)
    node = None
    executor = None
    rclpy.init(args=args)
    try:
        node = SuspendedRunTestNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
