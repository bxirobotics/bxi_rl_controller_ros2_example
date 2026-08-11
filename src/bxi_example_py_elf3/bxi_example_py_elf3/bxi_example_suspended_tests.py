"""Combined suspended motion tests with one actuator-command publisher."""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_path
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from std_srvs.srv import SetBool

from .bxi_example_vibration import VibrationTestNode
from .control.elf3 import (
    DOF_NUM,
    JOINT_KD,
    JOINT_KP,
    JOINT_NAMES,
    JOINT_NOMINAL_POS,
    SUSPENDED_RUN_NOMINAL_POS,
)
from .control.remote import RemoteButtonEdge
from .control.limb_sequence import (
    WHOLE_BODY_TEST_GROUPS,
    build_safe_ranges,
    full_range_waypoints,
    velocity_limited_duration,
)
from .control.trajectory import load_joint_trajectory, minimum_jerk_progress


class SuspendedTestNode(VibrationTestNode):
    """Own the actuator topic while offering mutually exclusive test modes.

    X controls the recorded running trajectory, Y controls vibration directly,
    and A runs the collision-margined whole-body full-range sequence. Modes
    are mutually exclusive and share one actuator command publisher.
    """

    def __init__(self):
        super().__init__()

        self.run_control_rate_hz = float(
            self.declare_parameter("run_control_rate_hz", 50.0).value
        )
        self.run_gain_ramp_sec = float(
            self.declare_parameter("run_gain_ramp_sec", 2.0).value
        )
        self.limb_test_move_sec = float(
            self.declare_parameter("limb_test_move_sec", 1.5).value
        )
        self.limb_test_hold_sec = float(
            self.declare_parameter("limb_test_hold_sec", 0.5).value
        )
        self.limb_test_tracking_tolerance_rad = math.radians(
            float(
                self.declare_parameter(
                    "limb_test_tracking_tolerance_deg", 2.0
                ).value
            )
        )
        self.limb_test_start_tolerance_rad = math.radians(
            float(
                self.declare_parameter(
                    "limb_test_start_tolerance_deg", 5.0
                ).value
            )
        )
        self.vibration_start_envelope_slack_rad = float(
            self.declare_parameter(
                "vibration_start_envelope_slack_rad", 0.001
            ).value
        )
        self.vibration_start_center_tolerance_rad = float(
            self.declare_parameter(
                "vibration_start_center_tolerance_rad", 0.1
            ).value
        )
        self.limb_test_collision_margin_deg = float(
            self.declare_parameter(
                "limb_test_collision_margin_deg", 5.0
            ).value
        )
        self.limb_test_mechanical_margin_deg = float(
            self.declare_parameter(
                "limb_test_mechanical_margin_deg", 2.0
            ).value
        )
        self.limb_test_range_speed_deg_s = float(
            self.declare_parameter(
                "limb_test_range_speed_deg_s", 20.0
            ).value
        )
        self.limb_test_visual_check_period_sec = float(
            self.declare_parameter(
                "limb_test_visual_check_period_sec", 0.05
            ).value
        )
        requested_path = str(
            self.declare_parameter("run_trajectory_path", "").value
        ).strip()
        if (
            not math.isfinite(self.run_control_rate_hz)
            or self.run_control_rate_hz <= 0.0
        ):
            raise ValueError("run_control_rate_hz must be finite and > 0")
        if self.run_control_rate_hz > self.control_rate_hz:
            raise ValueError(
                "run_control_rate_hz cannot exceed control_rate_hz"
            )
        if (
            not math.isfinite(self.run_gain_ramp_sec)
            or self.run_gain_ramp_sec <= 0.0
        ):
            raise ValueError("run_gain_ramp_sec must be finite and > 0")
        for name, value in (
            ("limb_test_move_sec", self.limb_test_move_sec),
            (
                "limb_test_tracking_tolerance_rad",
                self.limb_test_tracking_tolerance_rad,
            ),
            (
                "limb_test_start_tolerance_rad",
                self.limb_test_start_tolerance_rad,
            ),
            (
                "limb_test_range_speed_deg_s",
                self.limb_test_range_speed_deg_s,
            ),
            (
                "limb_test_visual_check_period_sec",
                self.limb_test_visual_check_period_sec,
            ),
            (
                "vibration_start_center_tolerance_rad",
                self.vibration_start_center_tolerance_rad,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and > 0" % name)
        if self.vibration_start_center_tolerance_rad > 0.1:
            raise ValueError(
                "vibration_start_center_tolerance_rad must be <= 0.1"
            )
        if (
            not math.isfinite(self.vibration_start_envelope_slack_rad)
            or self.vibration_start_envelope_slack_rad < 0.0
            or self.vibration_start_envelope_slack_rad > 0.005
        ):
            raise ValueError(
                "vibration_start_envelope_slack_rad must be finite and in "
                "[0.0, 0.005]"
            )
        limb_test_speed_limit = 180.0 if self.hardware_mode else 1000.0
        if self.limb_test_range_speed_deg_s > limb_test_speed_limit:
            raise ValueError(
                "limb_test_range_speed_deg_s must be <= %.1f in %s mode"
                % (
                    limb_test_speed_limit,
                    "hardware" if self.hardware_mode else "simulation",
                )
            )
        if (
            not math.isfinite(self.limb_test_hold_sec)
            or self.limb_test_hold_sec < 0.0
        ):
            raise ValueError("limb_test_hold_sec must be finite and >= 0")
        for name, value in (
            (
                "limb_test_collision_margin_deg",
                self.limb_test_collision_margin_deg,
            ),
            (
                "limb_test_mechanical_margin_deg",
                self.limb_test_mechanical_margin_deg,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and >= 0" % name)

        self.run_trajectory = load_joint_trajectory(
            self._resolve_run_trajectory_path(requested_path),
            self.run_control_rate_hz,
        )
        self.run_button = RemoteButtonEdge(
            self.motion_button_mode,
            self.motion_command_resync_sec,
        )
        self.vibration_button = RemoteButtonEdge(
            self.motion_button_mode,
            self.motion_command_resync_sec,
        )
        self.limb_test_button = RemoteButtonEdge(
            self.motion_button_mode,
            self.motion_command_resync_sec,
        )
        self.run_enabled = False
        self.run_phase = "idle"
        self.run_phase_started_at = 0.0
        self.pending_mode = ""
        self.run_command_positions = SUSPENDED_RUN_NOMINAL_POS.copy()
        self.run_blend_start_positions = SUSPENDED_RUN_NOMINAL_POS.copy()
        self.run_blend_target_positions = SUSPENDED_RUN_NOMINAL_POS.copy()
        self.next_run_frame_at = 0.0
        self.last_run_settle_log_at = 0.0
        self.limb_test_running = False
        self.limb_test_phase = "idle"
        self.limb_test_group_index = 0
        self.limb_test_segment_index = 0
        self.limb_test_segment_started_at = 0.0
        self.limb_test_segment_start = JOINT_NOMINAL_POS.copy()
        self.limb_test_segment_target = JOINT_NOMINAL_POS.copy()
        self.limb_test_waypoints = tuple()
        self.limb_test_motion_names = tuple()
        self.limb_test_segment_duration_sec = self.limb_test_move_sec
        self.limb_test_center_positions = np.zeros(DOF_NUM, dtype=np.float64)
        self.limb_test_target_ranges = build_safe_ranges(
            collision_margin_deg=self.limb_test_collision_margin_deg,
            mechanical_margin_deg=self.limb_test_mechanical_margin_deg,
        )
        self.limb_test_failures = 0
        self.limb_test_last_visual_check_at = 0.0
        self.limb_test_visual_check_measured_next = False
        self.limb_test_last_settle_log_at = 0.0
        self.vibration_last_settle_log_at = 0.0
        self.limb_collision_guard = None
        self.limb_collision_guard_error = "disabled by configuration"
        self.run_enable_service = self.create_service(
            SetBool,
            "run_trajectory_enable",
            self._run_enable_service_callback,
        )
        self.limb_test_enable_service = self.create_service(
            SetBool,
            "whole_body_joint_test_enable",
            self._limb_test_enable_service_callback,
        )

        diagnostics = self.run_trajectory.diagnostics()
        self.get_logger().info(
            "combined suspended tests ready: X=running, Y=vibration, "
            "A=arms/torso/legs joint test; "
            "run frames=%d run_rate=%.1f Hz max_step=%.6f rad "
            "(%.6f rad/s), loop_step=%.6f rad (%.6f rad/s), "
            "Kp=JOINT_KP[%.3f, %.3f]"
            % (
                diagnostics.frame_count,
                self.run_control_rate_hz,
                diagnostics.max_step_delta_rad,
                diagnostics.max_step_velocity_rad_s,
                diagnostics.loop_delta_rad,
                diagnostics.loop_velocity_rad_s,
                float(np.min(JOINT_KP)),
                float(np.max(JOINT_KP)),
            )
        )
        self.get_logger().info(
            "combined start tolerances: running/A center=%.3f deg, "
            "vibration center=%.6f rad, vibration envelope slack=%.6f rad; "
            "vibration commands remain clipped to the configured software "
            "position limits"
            % (
                math.degrees(self.limb_test_start_tolerance_rad),
                self.vibration_start_center_tolerance_rad,
                self.vibration_start_envelope_slack_rad,
            )
        )
        self.get_logger().warning(
            "A-key MuJoCo collision checks are disabled; joint tests will "
            "use configured joint limits, feedback freshness, start "
            "tolerance and tracking checks only"
        )

    def _remote_help_message(self):
        return (
            "combined remote: X starts/pauses running; Y starts/stops "
            "vibration directly; A tests arms, torso and legs in sequence; "
            "modes are mutually exclusive (motion_button_mode=%s)"
            % self.motion_button_mode
        )

    @staticmethod
    def _resolve_run_trajectory_path(requested_path):
        if requested_path:
            return Path(requested_path).expanduser()
        workspace_path = Path("src/bxi_example_py_elf3/data/data.txt")
        if workspace_path.is_file():
            return workspace_path
        return get_package_share_path("bxi_example_py_elf3") / "data/data.txt"

    def _motion_callback(self, msg):
        now = time.monotonic()
        if self.count_publishers(self.motion_topic) > 1:
            if now - self.last_remote_conflict_log_at >= 1.0:
                self.get_logger().error(
                    "multiple motion_commands publishers detected; X/Y/A "
                    "input is ignored until only one remote controller remains"
                )
                self.last_remote_conflict_log_at = now
            return

        with self.state_lock:
            run_activated = self.run_button.update(msg.btn_9 != 0, now)
            vibration_activated = self.vibration_button.update(
                msg.btn_10 != 0,
                now,
            )
            limb_test_activated = self.limb_test_button.update(
                msg.btn_7 != 0,
                now,
            )

        if sum((run_activated, vibration_activated, limb_test_activated)) > 1:
            self.get_logger().error(
                "multiple X/Y/A buttons changed in the same remote sample; "
                "all commands were ignored"
            )
            return
        if run_activated:
            self._handle_run_button()
        elif vibration_activated:
            self._handle_vibration_button(now)
        elif limb_test_activated:
            self._handle_limb_test_button()

    def _handle_run_button(self):
        with self.state_lock:
            if self.safety_fault:
                self.get_logger().error(
                    "X rejected after a latched safety fault; restart required"
                )
                return
            if self.reset_stage != 2:
                self.get_logger().warning(
                    "X ignored because robot initialization is incomplete"
                )
                return
            if self.run_enabled:
                self._stop_running_locked("remote X button")
                return
            if (
                self.test_enabled
                or self.joint_test_running
                or self.limb_test_running
            ):
                self.get_logger().warning(
                    "X rejected while vibration or joint test is active; "
                    "stop the active mode first"
                )
                return
            if self.returning_to_center or self.pending_mode:
                self.get_logger().warning(
                    "X ignored while a mode transition is in progress"
                )
                return
            self._prepare_running_locked("remote X button")

    def _handle_vibration_button(self, request_received_at):
        with self.state_lock:
            if self.safety_fault:
                self.get_logger().error(
                    "Y rejected after a latched safety fault; restart required"
                )
                return
            if self.reset_stage != 2:
                self.get_logger().warning(
                    "Y ignored because robot initialization is incomplete"
                )
                return
            if self.test_enabled:
                self._stop_test("remote Y button")
                return
            if self.limb_test_running:
                self.get_logger().warning(
                    "Y rejected while the A-key joint test is active; press A "
                    "to stop it first"
                )
                return
            if self.joint_test_running:
                self._cancel_joint_rotation_test("remote Y button")
                return
            if self.pending_mode in ("vibration", "vibration_start"):
                self.pending_mode = ""
                self._queue_diagnostic_log(
                    "info",
                    "pending vibration start cancelled by remote Y button",
                )
                return
            if (
                self.returning_to_center
                and self.return_owner == "limb_test_stop"
            ):
                self.pending_mode = "vibration"
                self._queue_diagnostic_log(
                    "info",
                    "vibration requested during A-key safe return; vibration "
                    "will start automatically after the zero pose is reached",
                )
                return
            if self.returning_to_center or self.pending_mode:
                self.get_logger().warning(
                    "Y ignored while a mode transition is in progress"
                )
                return
        self._request_vibration_start(
            "remote Y button", request_received_at
        )

    def _handle_limb_test_button(self):
        with self.state_lock:
            if self.safety_fault:
                self.get_logger().error(
                    "A rejected after a latched safety fault; restart required"
                )
                return
            if self.reset_stage != 2:
                self.get_logger().warning(
                    "A ignored because robot initialization is incomplete"
                )
                return
            if self.limb_test_running:
                self._stop_limb_test_locked("remote A button")
                return
            if self.test_enabled or self.joint_test_running:
                self.get_logger().warning(
                    "A rejected while vibration is active; stop it with Y "
                    "first"
                )
                return
            if self.returning_to_center or self.pending_mode:
                self.get_logger().warning(
                    "A ignored while a mode transition is in progress"
                )
                return
            self._prepare_limb_test_locked("remote A button")

    def _preflight_limb_plan(self):
        """Verify every full-range segment against model collision geoms."""
        current = self.limb_test_center_positions.copy()
        reason = self._limb_pose_collision_reason(
            current, JOINT_NAMES, visual=True
        )
        if reason:
            raise ValueError(
                "zero reference pose is not collision-free: " + reason
            )
        for group in WHOLE_BODY_TEST_GROUPS:
            motion_names, waypoints = full_range_waypoints(
                self.limb_test_center_positions,
                group,
                self.limb_test_target_ranges,
            )
            for target in waypoints:
                max_travel_deg = float(
                    np.max(np.abs(np.rad2deg(target - current)))
                )
                sample_count = max(2, int(math.ceil(max_travel_deg / 2.0)) + 1)
                for progress in np.linspace(0.0, 1.0, sample_count):
                    pose = current + progress * (target - current)
                    reason = self._limb_pose_collision_reason(
                        pose, motion_names, visual=False
                    )
                    if reason:
                        raise ValueError(
                            "%s planned path is not collision-free: %s"
                            % (group.label, reason)
                        )
                # Visual STL checks are much heavier. Check every endpoint at
                # startup and check interpolated commands online while moving.
                reason = self._limb_pose_collision_reason(
                    target, motion_names, visual=True
                )
                if reason:
                    raise ValueError(
                        "%s visual-mesh endpoint is unsafe: %s"
                        % (group.label, reason)
                    )
                current = target.copy()

    def _limb_pose_collision_reason(self, pose, motion_names, visual):
        if self.limb_collision_guard is None:
            return ""
        contacts = self.limb_collision_guard.collisions(pose)
        if contacts:
            first = contacts[0]
            return "collision geoms %s / %s (depth %.6f m)" % first
        if visual:
            contacts = self.limb_collision_guard.visual_mesh_collisions_any(
                pose
            )
            if contacts:
                first = contacts[0]
                return "visual meshes %s / %s (depth %.6f m)" % first
        return ""

    def _transition_is_collision_free(self, start, target):
        max_travel_deg = float(np.max(np.abs(np.rad2deg(target - start))))
        sample_count = max(2, int(math.ceil(max_travel_deg / 2.0)) + 1)
        for progress in np.linspace(0.0, 1.0, sample_count):
            pose = start + progress * (target - start)
            reason = self._limb_pose_collision_reason(
                pose, JOINT_NAMES, visual=False
            )
            if reason:
                return False, reason
        for pose in (start, target):
            reason = self._limb_pose_collision_reason(
                pose, JOINT_NAMES, visual=True
            )
            if reason:
                return False, reason
        return True, ""

    def _prepare_limb_test_locked(self, source):
        now = time.monotonic()
        if not self._joint_feedback_ready(now):
            self.get_logger().error(
                "A rejected because complete fresh joint feedback is required"
            )
            return False
        with self.feedback_lock:
            measured = self.measured_positions.copy()
        start_error = np.abs(measured - self.last_command_positions)
        worst_index = int(np.argmax(start_error))
        if start_error[worst_index] > self.limb_test_start_tolerance_rad:
            self.get_logger().error(
                "A rejected because %s feedback differs from the hold command "
                "by %.3f deg (limit %.3f deg)"
                % (
                    JOINT_NAMES[worst_index],
                    math.degrees(start_error[worst_index]),
                    math.degrees(self.limb_test_start_tolerance_rad),
                )
            )
            return False
        self.run_enabled = False
        self.run_phase = "idle"
        self.joint_test_passed = False
        self.joint_test_passed_at = 0.0
        self.center_positions[:] = self.limb_test_center_positions
        self.pending_mode = "limb_test"
        transition_duration = velocity_limited_duration(
            self.last_command_positions,
            self.limb_test_center_positions,
            JOINT_NAMES,
            self.limb_test_move_sec,
            self.limb_test_range_speed_deg_s,
        )
        self._begin_smooth_return_locked(
            "prepare_limb_test", duration_sec=transition_duration
        )
        self._queue_diagnostic_log(
            "info",
            "full-range joint test requested by %s; moving to the collision-"
            "scanned zero reference over %.3f seconds"
            % (source, transition_duration),
        )
        return True

    def _start_limb_test_locked(self):
        if self.safety_fault or self.reset_stage != 2:
            return False
        if (
            self.test_enabled
            or self.joint_test_running
            or self.limb_test_running
            or self.returning_to_center
        ):
            return False
        now = time.monotonic()
        if not self._joint_feedback_ready(now):
            self.pending_mode = "limb_test"
            return False
        with self.feedback_lock:
            measured = self.measured_positions.copy()
        center_error = np.abs(measured - self.limb_test_center_positions)
        worst_index = int(np.argmax(center_error))
        if center_error[worst_index] > self.limb_test_start_tolerance_rad:
            self.pending_mode = "limb_test"
            if now - self.limb_test_last_settle_log_at >= 1.0:
                self._queue_diagnostic_log(
                    "warning",
                    "waiting for zero reference to settle: %s error %.3f deg "
                    "exceeds %.3f deg"
                    % (
                        JOINT_NAMES[worst_index],
                        math.degrees(center_error[worst_index]),
                        math.degrees(self.limb_test_start_tolerance_rad),
                    ),
                )
                self.limb_test_last_settle_log_at = now
            return False
        self.limb_test_running = True
        self.limb_test_phase = "move"
        self.limb_test_group_index = 0
        self.limb_test_segment_index = 0
        self.limb_test_failures = 0
        self.limb_test_last_visual_check_at = 0.0
        self.limb_test_visual_check_measured_next = False
        self._load_limb_group_locked(now)
        self._queue_diagnostic_log(
            "info",
            "FULL-RANGE JOINT TEST STARTED: arms -> torso -> legs; A stops",
        )
        return True

    def _load_limb_group_locked(self, now):
        group = WHOLE_BODY_TEST_GROUPS[self.limb_test_group_index]
        (
            self.limb_test_motion_names,
            self.limb_test_waypoints,
        ) = full_range_waypoints(
            self.limb_test_center_positions,
            group,
            self.limb_test_target_ranges,
        )
        self.limb_test_segment_index = 0
        self.limb_test_segment_start[:] = self.last_command_positions
        self.limb_test_segment_target[:] = self.limb_test_waypoints[0]
        self.limb_test_segment_duration_sec = velocity_limited_duration(
            self.limb_test_segment_start,
            self.limb_test_segment_target,
            self.limb_test_motion_names,
            self.limb_test_move_sec,
            self.limb_test_range_speed_deg_s,
        )
        self.limb_test_segment_started_at = now
        self.limb_test_phase = "move"
        self._queue_diagnostic_log(
            "info",
            "joint test %d/%d %s started; segment 1/%d duration %.3f seconds"
            % (
                self.limb_test_group_index + 1,
                len(WHOLE_BODY_TEST_GROUPS),
                group.label,
                len(self.limb_test_waypoints),
                self.limb_test_segment_duration_sec,
            ),
        )

    def _start_next_limb_segment_locked(self, now):
        self.limb_test_segment_index += 1
        if self.limb_test_segment_index >= len(self.limb_test_waypoints):
            completed = WHOLE_BODY_TEST_GROUPS[self.limb_test_group_index]
            self._queue_diagnostic_log(
                "info", "joint test group completed: %s" % completed.label
            )
            self.limb_test_group_index += 1
            if self.limb_test_group_index >= len(WHOLE_BODY_TEST_GROUPS):
                self.limb_test_running = False
                self.limb_test_phase = "idle"
                self.center_positions[:] = self.limb_test_center_positions
                self._queue_diagnostic_log(
                    "info",
                    "FULL-RANGE JOINT TEST COMPLETE: failures=%d; holding the "
                    "zero reference pose" % self.limb_test_failures,
                )
                return
            self._load_limb_group_locked(now)
            return
        self.limb_test_segment_start[:] = self.limb_test_segment_target
        self.limb_test_segment_target[:] = self.limb_test_waypoints[
            self.limb_test_segment_index
        ]
        self.limb_test_segment_duration_sec = velocity_limited_duration(
            self.limb_test_segment_start,
            self.limb_test_segment_target,
            self.limb_test_motion_names,
            self.limb_test_move_sec,
            self.limb_test_range_speed_deg_s,
        )
        self.limb_test_segment_started_at = now
        self.limb_test_phase = "move"

    def _stop_limb_test_locked(self, source):
        self.limb_test_running = False
        self.limb_test_phase = "idle"
        self.pending_mode = ""
        self.center_positions[:] = self.limb_test_center_positions
        duration = velocity_limited_duration(
            self.last_command_positions,
            self.limb_test_center_positions,
            JOINT_NAMES,
            self.limb_test_move_sec,
            self.limb_test_range_speed_deg_s,
        )
        self._begin_smooth_return_locked(
            "limb_test_stop", duration_sec=duration
        )
        self._queue_diagnostic_log(
            "info",
            "full-range joint test stopped by %s; returning to zero over "
            "%.3f seconds" % (source, duration),
        )

    def _request_vibration_start(self, source, request_received_at):
        with self.state_lock:
            centered = np.allclose(
                self.last_command_positions,
                JOINT_NOMINAL_POS,
                rtol=0.0,
                atol=1.0e-9,
            )
            if not centered:
                return self._prepare_vibration_locked(source)
        return self._start_test(request_received_at=request_received_at)

    def _prepare_vibration_locked(self, source):
        if self.safety_fault or self.reset_stage != 2:
            return False
        target = JOINT_NOMINAL_POS
        self.run_enabled = False
        self.run_phase = "idle"
        self.center_positions[:] = target
        self.pending_mode = "vibration_start"
        duration = velocity_limited_duration(
            self.last_command_positions,
            target,
            JOINT_NAMES,
            self.limb_test_move_sec,
            self.limb_test_range_speed_deg_s,
        )
        self._begin_smooth_return_locked(
            "prepare_vibration", duration_sec=duration
        )
        self._queue_diagnostic_log(
            "info",
            "vibration requested by %s; moving from the current pose to the "
            "vibration center over %.3f seconds before excitation starts"
            % (source, duration),
        )
        return True

    def _start_centered_vibration(self, now):
        if not self._joint_feedback_ready(now):
            with self.state_lock:
                self.pending_mode = "vibration_start"
            return False
        with self.feedback_lock:
            measured = self.measured_positions.copy()
        target_center = self.center_positions.copy()
        center_error = np.abs(measured - target_center)
        worst_center_index = int(np.argmax(center_error))
        if (
            center_error[worst_center_index]
            > self.vibration_start_center_tolerance_rad
        ):
            with self.state_lock:
                self.pending_mode = "vibration_start"
                if now - self.vibration_last_settle_log_at >= 1.0:
                    self._queue_diagnostic_log(
                        "warning",
                        "waiting for measured joints to settle within the "
                        "vibration center tolerance: %s error %.6f rad "
                        "exceeds %.6f rad"
                        % (
                            JOINT_NAMES[worst_center_index],
                            center_error[worst_center_index],
                            self.vibration_start_center_tolerance_rad,
                        ),
                    )
                    self.vibration_last_settle_log_at = now
            return False
        violations = self._envelope_violation_details(
            target_center,
            self.amplitude_rad,
            self.active_joint_indices,
            slack_rad=self.vibration_start_envelope_slack_rad,
        )
        if violations:
            with self.state_lock:
                self.pending_mode = "vibration_start"
                if now - self.vibration_last_settle_log_at >= 1.0:
                    self._queue_diagnostic_log(
                        "warning",
                        "waiting for a vibration-safe measured center: "
                        + "; ".join(violations),
                    )
                    self.vibration_last_settle_log_at = now
            return False
        return self._start_test(
            request_received_at=now,
            center_override=target_center,
        )

    def _prepare_running_locked(self, source):
        self.joint_test_passed = False
        self.joint_test_passed_at = 0.0
        self.joint_test_message = (
            "joint rotation precheck invalidated by running mode"
        )
        self.center_positions[:] = SUSPENDED_RUN_NOMINAL_POS
        self.pending_mode = "running"
        self._begin_smooth_return_locked("prepare_running")
        self._queue_diagnostic_log(
            "info",
            "running requested by %s; returning to the running center over "
            "%.3f seconds" % (source, self.stop_ramp_sec),
        )

    def _start_running_locked(self):
        if self.safety_fault or self.reset_stage != 2:
            return False
        if (
            self.test_enabled
            or self.joint_test_running
            or self.limb_test_running
            or self.returning_to_center
        ):
            return False
        if self.require_joint_state and not self._joint_feedback_ready(
            time.monotonic()
        ):
            self._latch_safety_fault(
                "joint feedback is incomplete or stale before running"
            )
            return False
        self.run_enabled = True
        self.run_phase = "gain_ramp"
        self.run_phase_started_at = time.monotonic()
        self.run_trajectory.reset()
        self.run_command_positions[:] = SUSPENDED_RUN_NOMINAL_POS
        self.run_blend_start_positions[:] = SUSPENDED_RUN_NOMINAL_POS
        self.run_blend_target_positions[:] = SUSPENDED_RUN_NOMINAL_POS
        self.next_run_frame_at = 0.0
        self._queue_diagnostic_log(
            "info",
            "running center settle started: holding with the shared JOINT_KP "
            "profile for %.3f seconds" % self.run_gain_ramp_sec,
        )
        return True

    def _stop_running_locked(self, source):
        self.run_enabled = False
        self.run_phase = "idle"
        self.pending_mode = ""
        self.center_positions[:] = SUSPENDED_RUN_NOMINAL_POS
        self._begin_smooth_return_locked("running")
        self._queue_diagnostic_log(
            "info",
            "running stopped by %s; returning to the running center over "
            "%.3f seconds" % (source, self.stop_ramp_sec),
        )

    def _idle_command(self, now):
        pending = ""
        with self.state_lock:
            if self.pending_mode and not self.returning_to_center:
                pending = self.pending_mode
                self.pending_mode = ""

        if pending == "running":
            with self.state_lock:
                self._start_running_locked()
        elif pending == "limb_test":
            with self.state_lock:
                self._start_limb_test_locked()
        elif pending == "vibration":
            with self.state_lock:
                self._prepare_vibration_locked("queued remote Y button")
        elif pending == "vibration_start":
            self._start_centered_vibration(now)

        with self.state_lock:
            if self.limb_test_running:
                command = self._limb_test_command_locked(now)
                return command, JOINT_KP, JOINT_KD
            if not self.run_enabled:
                return super()._idle_command(now)

            if self.run_phase == "gain_ramp":
                progress = (
                    now - self.run_phase_started_at
                ) / self.run_gain_ramp_sec
                command_kp = JOINT_KP
                if progress >= 1.0:
                    with self.feedback_lock:
                        has_feedback = self.has_joint_state
                        measured = self.measured_positions.copy()
                    if has_feedback:
                        center_error = np.abs(
                            measured - SUSPENDED_RUN_NOMINAL_POS
                        )
                        worst_index = int(np.argmax(center_error))
                        if (
                            center_error[worst_index]
                            > self.limb_test_start_tolerance_rad
                        ):
                            if now - self.last_run_settle_log_at >= 1.0:
                                self._queue_diagnostic_log(
                                    "warning",
                                    "waiting for running center to settle: "
                                    "%s error %.6f rad exceeds "
                                    "%.6f rad"
                                    % (
                                        JOINT_NAMES[worst_index],
                                        center_error[worst_index],
                                        self.limb_test_start_tolerance_rad,
                                    ),
                                )
                                self.last_run_settle_log_at = now
                            return (
                                SUSPENDED_RUN_NOMINAL_POS.copy(),
                                JOINT_KP,
                                JOINT_KD,
                            )
                    self.run_phase = "trajectory_blend"
                    self.run_phase_started_at = now
                    self.run_blend_start_positions[:] = (
                        SUSPENDED_RUN_NOMINAL_POS
                    )
                    self.run_blend_target_positions[:] = (
                        self.run_trajectory.next()
                    )
                    self._queue_diagnostic_log(
                        "info",
                        "running center settle complete; blending smoothly "
                        "into trajectory frame 1 over %.3f seconds"
                        % self.stop_ramp_sec,
                    )
                return (
                    SUSPENDED_RUN_NOMINAL_POS.copy(),
                    command_kp,
                    JOINT_KD,
                )

            if self.run_phase == "trajectory_blend":
                progress = (
                    now - self.run_phase_started_at
                ) / self.stop_ramp_sec
                smooth = minimum_jerk_progress(progress)
                self.run_command_positions[:] = (
                    self.run_blend_start_positions
                    + smooth
                    * (
                        self.run_blend_target_positions
                        - self.run_blend_start_positions
                    )
                )
                if progress >= 1.0:
                    self.run_phase = "playback"
                    self.run_command_positions[:] = (
                        self.run_blend_target_positions
                    )
                    self.next_run_frame_at = (
                        now + 1.0 / self.run_control_rate_hz
                    )
                    self._queue_diagnostic_log(
                        "info",
                        "RUNNING TEST STARTED: trajectory starts from frame "
                        "1; X pauses running; Y switches safely to vibration",
                    )
                return self.run_command_positions.copy(), JOINT_KP, JOINT_KD

            if self.run_phase == "loop_blend":
                progress = (
                    now - self.run_phase_started_at
                ) / self.stop_ramp_sec
                smooth = minimum_jerk_progress(progress)
                self.run_command_positions[:] = (
                    self.run_blend_start_positions
                    + smooth
                    * (
                        self.run_blend_target_positions
                        - self.run_blend_start_positions
                    )
                )
                if progress >= 1.0:
                    self.run_phase = "playback"
                    self.run_command_positions[:] = (
                        self.run_blend_target_positions
                    )
                    self.next_run_frame_at = (
                        now + 1.0 / self.run_control_rate_hz
                    )
                    self._queue_diagnostic_log(
                        "info",
                        "running loop transition complete; entering the "
                        "trajectory stop segment",
                    )
                return self.run_command_positions.copy(), JOINT_KP, JOINT_KD

            if now >= self.next_run_frame_at:
                if self.run_trajectory.next_is_first_frame:
                    self.run_phase = "loop_blend"
                    self.run_phase_started_at = now
                    self.run_blend_start_positions[:] = (
                        self.run_command_positions
                    )
                    self.run_blend_target_positions[:] = (
                        self.run_trajectory.next()
                    )
                    self._queue_diagnostic_log(
                        "info",
                        "running cycle complete; blending the final frame "
                        "into frame 1 over %.3f seconds before stopping"
                        % self.stop_ramp_sec,
                    )
                    return (
                        self.run_command_positions.copy(),
                        JOINT_KP,
                        JOINT_KD,
                    )
                self.run_command_positions[:] = self.run_trajectory.next()
                self.next_run_frame_at += 1.0 / self.run_control_rate_hz
                if self.next_run_frame_at <= now:
                    self.next_run_frame_at = (
                        now + 1.0 / self.run_control_rate_hz
                    )
            return self.run_command_positions.copy(), JOINT_KP, JOINT_KD

    def _limb_test_command_locked(self, now):
        if not self._joint_feedback_ready(now):
            self._latch_safety_fault(
                "joint feedback became incomplete or stale during the "
                "full-range joint test"
            )
            return self.last_command_positions.copy()

        if self.limb_test_phase == "hold":
            command = self.limb_test_segment_target.copy()
            if (
                now - self.limb_test_segment_started_at
                >= self.limb_test_hold_sec
            ):
                self._evaluate_limb_segment_locked()
                self._start_next_limb_segment_locked(now)
                if not self.limb_test_running:
                    return self.limb_test_center_positions.copy()
                command = self.limb_test_segment_start.copy()
        else:
            progress = (
                (now - self.limb_test_segment_started_at)
                / self.limb_test_segment_duration_sec
            )
            smooth = minimum_jerk_progress(progress)
            command = (
                self.limb_test_segment_start
                + smooth
                * (
                    self.limb_test_segment_target
                    - self.limb_test_segment_start
                )
            )
            if progress >= 1.0:
                command = self.limb_test_segment_target.copy()
                self.limb_test_phase = "hold"
                self.limb_test_segment_started_at = now

        visual = (
            now - self.limb_test_last_visual_check_at
            >= self.limb_test_visual_check_period_sec
        )
        visual_measured = (
            visual and self.limb_test_visual_check_measured_next
        )
        reason = self._limb_pose_collision_reason(
            command,
            self.limb_test_motion_names,
            visual=visual and not visual_measured,
        )
        if reason:
            self._latch_safety_fault(
                "planned A-key joint-test command entered a collision: "
                + reason
            )
            return self.last_command_positions.copy()
        if visual:
            self.limb_test_last_visual_check_at = now
            self.limb_test_visual_check_measured_next = not visual_measured
            if visual_measured:
                with self.feedback_lock:
                    measured = self.measured_positions.copy()
                reason = self._limb_pose_collision_reason(
                    measured, self.limb_test_motion_names, visual=True
                )
                if reason:
                    self._latch_safety_fault(
                        "measured A-key joint-test pose entered a collision: "
                        + reason
                    )
                    return self.last_command_positions.copy()
        return command

    def _evaluate_limb_segment_locked(self):
        with self.feedback_lock:
            measured = self.measured_positions.copy()
        indices = [
            JOINT_NAMES.index(name)
            for name in self.limb_test_motion_names
        ]
        errors = np.abs(
            measured[indices] - self.limb_test_segment_target[indices]
        )
        worst_offset = int(np.argmax(errors))
        worst_error = float(errors[worst_offset])
        if worst_error <= self.limb_test_tracking_tolerance_rad:
            return
        self.limb_test_failures += 1
        joint_name = self.limb_test_motion_names[worst_offset]
        self._queue_diagnostic_log(
            "warning",
            "joint-test tracking warning: %s error %.3f deg exceeds %.3f deg"
            % (
                joint_name,
                math.degrees(worst_error),
                math.degrees(self.limb_test_tracking_tolerance_rad),
            ),
        )

    def _return_to_center_command(self, now):
        with self.state_lock:
            owner = self.return_owner
        command = super()._return_to_center_command(now)
        if owner not in (
            "prepare_limb_test",
            "limb_test_stop",
            "prepare_vibration",
        ):
            return command
        if self.limb_collision_guard is None:
            return command
        visual = (
            now - self.limb_test_last_visual_check_at
            >= self.limb_test_visual_check_period_sec
        )
        reason = self._limb_pose_collision_reason(
            command, JOINT_NAMES, visual=visual
        )
        if reason:
            self._latch_safety_fault(
                "%s transition entered a collision: %s" % (owner, reason)
            )
            return self.last_command_positions.copy()
        if visual:
            self.limb_test_last_visual_check_at = now
        return command

    def _latch_safety_fault(self, reason):
        with self.state_lock:
            if hasattr(self, "run_enabled"):
                self.run_enabled = False
                self.run_phase = "idle"
                self.pending_mode = ""
            if hasattr(self, "limb_test_running"):
                self.limb_test_running = False
                self.limb_test_phase = "idle"
        super()._latch_safety_fault(reason)

    def _start_joint_rotation_test(self):
        return self._reject_joint_test(
            "legacy vibration precheck is disabled in combined mode; Y "
            "starts vibration directly and A runs the full-range joint test"
        )

    def _start_test(self, request_received_at=None, center_override=None):
        with self.state_lock:
            if (
                self.run_enabled
                or self.limb_test_running
                or self.pending_mode in ("running", "limb_test")
            ):
                return self._reject_start(
                    "cannot start vibration while running or the A-key joint "
                    "test is active; stop the active mode first"
                )
        return super()._start_test(
            request_received_at=request_received_at,
            envelope_slack_rad=self.vibration_start_envelope_slack_rad,
            center_override=center_override,
        )

    def _run_enable_service_callback(self, request, response):
        with self.state_lock:
            if request.data:
                if self.run_enabled:
                    response.success = True
                    response.message = "running trajectory is already active"
                    return response
                if self.safety_fault:
                    response.success = False
                    response.message = "latched safety fault; restart required"
                    return response
                if self.reset_stage != 2:
                    response.success = False
                    response.message = "robot initialization is incomplete"
                    return response
                if (
                    self.test_enabled
                    or self.joint_test_running
                    or self.limb_test_running
                    or self.returning_to_center
                    or self.pending_mode
                ):
                    response.success = False
                    response.message = "another test or transition is active"
                    return response
                self._prepare_running_locked("run enable service")
                response.success = True
                response.message = (
                    "running accepted; motion starts after smooth centering"
                )
                return response

            if self.run_enabled:
                self._stop_running_locked("run enable service")
                response.message = (
                    "running stopped; returning smoothly to center"
                )
            elif self.pending_mode == "running":
                self.pending_mode = ""
                response.message = "pending running start cancelled"
            else:
                response.message = "running trajectory is already stopped"
            response.success = True
            return response

    def _limb_test_enable_service_callback(self, request, response):
        with self.state_lock:
            if request.data:
                if self.limb_test_running:
                    response.success = True
                    response.message = (
                        "full-range joint test is already active"
                    )
                    return response
                if self.safety_fault:
                    response.success = False
                    response.message = "latched safety fault; restart required"
                    return response
                if self.reset_stage != 2:
                    response.success = False
                    response.message = "robot initialization is incomplete"
                    return response
                if (
                    self.test_enabled
                    or self.joint_test_running
                    or self.returning_to_center
                    or self.pending_mode
                ):
                    response.success = False
                    response.message = "another test or transition is active"
                    return response
                accepted = self._prepare_limb_test_locked(
                    "whole-body joint-test service"
                )
                response.success = bool(accepted)
                response.message = (
                    "full-range joint test accepted; motion starts after "
                    "zero centering"
                    if accepted
                    else "full-range joint test rejected; see controller log"
                )
                return response

            if self.limb_test_running:
                self._stop_limb_test_locked("whole-body joint-test service")
                response.message = (
                    "full-range joint test stopped; returning safely to zero"
                )
            elif self.pending_mode == "limb_test":
                self.pending_mode = ""
                response.message = (
                    "pending full-range joint-test start cancelled; zero "
                    "centering will finish"
                )
            else:
                response.message = "full-range joint test is already stopped"
            response.success = True
            return response


def main(args=None):
    node = None
    executor = None
    rclpy.init(args=args)
    try:
        node = SuspendedTestNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except (Exception, KeyboardInterrupt):
                pass
        if node is not None:
            try:
                node.destroy_node()
            except (Exception, KeyboardInterrupt):
                pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
