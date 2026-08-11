import csv
import math
import time
from queue import Empty, Full, Queue, SimpleQueue
from threading import Event, Lock, RLock, Thread

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

import communication.msg as bxiMsg
import communication.srv as bxiSrv
import sensor_msgs.msg
from std_srvs.srv import SetBool, Trigger

from .control.elf3 import (
    DOF_NUM,
    JOINT_KD,
    JOINT_KP,
    JOINT_NAMES,
    JOINT_NOMINAL_POS,
    JOINT_POSITION_MAX,
    JOINT_POSITION_MIN,
    ROBOT_NAME,
)
from .control.remote import RemoteButtonEdge


MINIMUM_JERK_MAX_SLOPE = 1.875
MINIMUM_JERK_MAX_CURVATURE = 10.0 * math.sqrt(3.0) / 3.0


class VibrationTestNode(Node):
    """Apply a sine or linear swept-sine command to one or all Elf3 joints."""

    def __init__(self):
        super().__init__("bxi_example_py_elf3_vibration")

        self.topic_prefix = str(
            self.declare_parameter("topic_prefix", "simulation/").value
        )
        self.joint_name = str(
            self.declare_parameter("joint_name", "all").value
        )
        self.amplitude_rad = float(
            self.declare_parameter("amplitude_rad", 0.02).value
        )
        self.start_frequency_hz = float(
            self.declare_parameter("start_frequency_hz", 1.0).value
        )
        self.end_frequency_hz = float(
            self.declare_parameter("end_frequency_hz", 5.0).value
        )
        self.duration_sec = float(
            self.declare_parameter("duration_sec", 60.0).value
        )
        self.control_rate_hz = float(
            self.declare_parameter("control_rate_hz", 200.0).value
        )
        self.initialization_sec = float(
            self.declare_parameter("initialization_sec", 2.0).value
        )
        self.stop_ramp_sec = float(
            self.declare_parameter("stop_ramp_sec", 0.5).value
        )
        self.motion_command_resync_sec = float(
            self.declare_parameter("motion_command_resync_sec", 0.5).value
        )
        self.motion_button_mode = str(
            self.declare_parameter("motion_button_mode", "toggle").value
        ).strip().lower()
        self.joint_test_required = bool(
            self.declare_parameter("joint_test_required", True).value
        )
        self.allow_hardware_without_joint_test = bool(
            self.declare_parameter(
                "allow_hardware_without_joint_test", False
            ).value
        )
        self.joint_test_amplitude_rad = float(
            self.declare_parameter("joint_test_amplitude_rad", 0.03).value
        )
        self.joint_test_move_sec = float(
            self.declare_parameter("joint_test_move_sec", 0.4).value
        )
        self.joint_test_hold_sec = float(
            self.declare_parameter("joint_test_hold_sec", 0.1).value
        )
        self.joint_test_initial_settle_sec = float(
            self.declare_parameter("joint_test_initial_settle_sec", 0.2).value
        )
        self.joint_test_start_tolerance_rad = float(
            self.declare_parameter(
                "joint_test_start_tolerance_rad", 0.05
            ).value
        )
        self.joint_test_min_motion_rad = float(
            self.declare_parameter("joint_test_min_motion_rad", 0.015).value
        )
        self.joint_test_tracking_tolerance_rad = float(
            self.declare_parameter(
                "joint_test_tracking_tolerance_rad", 0.02
            ).value
        )
        self.joint_test_center_tolerance_rad = float(
            self.declare_parameter(
                "joint_test_center_tolerance_rad", 0.02
            ).value
        )
        self.joint_test_cross_axis_limit_rad = float(
            self.declare_parameter(
                "joint_test_cross_axis_limit_rad", 0.03
            ).value
        )
        self.joint_test_pass_valid_sec = float(
            self.declare_parameter("joint_test_pass_valid_sec", 300.0).value
        )
        self.joint_test_confirmation_delay_sec = float(
            self.declare_parameter(
                "joint_test_confirmation_delay_sec", 1.0
            ).value
        )
        self.joint_test_min_feedback_samples = int(
            self.declare_parameter("joint_test_min_feedback_samples", 3).value
        )
        self.joint_test_verify_feedback = bool(
            self.declare_parameter("joint_test_verify_feedback", True).value
        )
        self.joint_test_cycle_sec = 4.0 * (
            self.joint_test_move_sec + self.joint_test_hold_sec
        )
        self.release_suspension = bool(
            self.declare_parameter("release_suspension", False).value
        )
        self.auto_start = bool(self.declare_parameter("auto_start", False).value)
        self.kp_scale = float(self.declare_parameter("kp_scale", 1.0).value)
        self.kd_scale = float(self.declare_parameter("kd_scale", 1.0).value)
        self.log_csv_path = str(
            self.declare_parameter(
                "log_csv_path", "/tmp/elf3_vibration_test.csv"
            ).value
        )
        self.log_rate_hz = float(
            self.declare_parameter(
                "log_rate_hz", min(self.control_rate_hz, 100.0)
            ).value
        )
        self.log_queue_capacity = int(
            self.declare_parameter("log_queue_capacity", 2048).value
        )
        self.hardware_mode = bool(
            self.declare_parameter("hardware_mode", False).value
        )
        self.shutdown_on_safety_fault = bool(
            self.declare_parameter(
                "shutdown_on_safety_fault", self.hardware_mode
            ).value
        )
        self.require_joint_state = bool(
            self.declare_parameter("require_joint_state", False).value
        )
        self.allow_all_joints = bool(
            self.declare_parameter("allow_all_joints", True).value
        )
        self.joint_limit_margin_rad = float(
            self.declare_parameter("joint_limit_margin_rad", 0.02).value
        )
        self.joint_state_timeout_sec = float(
            self.declare_parameter("joint_state_timeout_sec", 0.2).value
        )
        self.max_command_gap_sec = float(
            self.declare_parameter("max_command_gap_sec", 0.05).value
        )
        self.publisher_check_period_sec = float(
            self.declare_parameter("publisher_check_period_sec", 0.5).value
        )
        self.reset_response_timeout_sec = float(
            self.declare_parameter("reset_response_timeout_sec", 5.0).value
        )
        self.hardware_max_amplitude_rad = float(
            self.declare_parameter("hardware_max_amplitude_rad", 0.01).value
        )
        self.hardware_max_frequency_hz = float(
            self.declare_parameter("hardware_max_frequency_hz", 2.0).value
        )
        self.hardware_max_velocity_rad_s = float(
            self.declare_parameter("hardware_max_velocity_rad_s", 0.2).value
        )
        self.hardware_max_acceleration_rad_s2 = float(
            self.declare_parameter("hardware_max_acceleration_rad_s2", 2.0).value
        )
        self.hardware_max_control_rate_hz = float(
            self.declare_parameter("hardware_max_control_rate_hz", 500.0).value
        )
        self.hardware_max_joint_test_amplitude_rad = float(
            self.declare_parameter(
                "hardware_max_joint_test_amplitude_rad", 0.1
            ).value
        )
        self.hardware_joint_test_min_move_sec = float(
            self.declare_parameter(
                "hardware_joint_test_min_move_sec", 0.2
            ).value
        )
        self.hardware_joint_test_min_response_ratio = float(
            self.declare_parameter(
                "hardware_joint_test_min_response_ratio", 0.25
            ).value
        )
        self.hardware_joint_test_max_tolerance_rad = float(
            self.declare_parameter(
                "hardware_joint_test_max_tolerance_rad", 0.05
            ).value
        )
        self.hardware_joint_test_max_velocity_rad_s = float(
            self.declare_parameter(
                "hardware_joint_test_max_velocity_rad_s", 0.5
            ).value
        )
        self.hardware_joint_test_max_acceleration_rad_s2 = float(
            self.declare_parameter(
                "hardware_joint_test_max_acceleration_rad_s2", 5.0
            ).value
        )

        self._validate_parameters()
        self.remote_button = RemoteButtonEdge(
            self.motion_button_mode,
            self.motion_command_resync_sec,
        )
        if self.joint_name == "all":
            self.active_joint_indices = np.arange(DOF_NUM, dtype=np.int64)
            self.active_joint_names = JOINT_NAMES
        else:
            self.active_joint_indices = np.array(
                [JOINT_NAMES.index(self.joint_name)], dtype=np.int64
            )
            self.active_joint_names = (self.joint_name,)

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
            "vibration_test_enable",
            self._enable_service_callback,
        )
        self.joint_test_service = self.create_service(
            SetBool,
            "joint_rotation_test_enable",
            self._joint_test_service_callback,
        )
        self.joint_test_status_service = self.create_service(
            Trigger,
            "joint_rotation_test_status",
            self._joint_test_status_callback,
        )

        self.feedback_lock = Lock()
        self.state_lock = RLock()
        self.measured_positions = JOINT_NOMINAL_POS.copy()
        self.has_joint_state = False
        self.joint_state_seen = np.zeros(DOF_NUM, dtype=bool)
        self.joint_state_last_seen_at = np.zeros(DOF_NUM, dtype=np.float64)
        self.joint_feedback_sequence = np.zeros(DOF_NUM, dtype=np.uint64)
        self.center_positions = JOINT_NOMINAL_POS.copy()

        self.reset_stage = 0
        self.reset_future = None
        self.reset_pending_step = 0
        self.reset_request_sent_at = 0.0
        self.reset_retry_after_at = 0.0
        self.initialization_started_at = 0.0
        self.test_started_at = 0.0
        self.test_enabled = False
        self.joint_test_running = False
        self.joint_test_passed = False
        self.joint_test_failed = False
        self.joint_test_failure_reason = ""
        self.joint_test_passed_at = 0.0
        self.joint_test_current_index = 0
        self.joint_test_joint_started_at = 0.0
        self.joint_test_segment_index = 0
        self.joint_test_center_positions = JOINT_NOMINAL_POS.copy()
        self.joint_test_feedback_baseline_positions = JOINT_NOMINAL_POS.copy()
        self.joint_test_baseline_finalized = False
        self.joint_test_observed_positive_rad = 0.0
        self.joint_test_observed_negative_rad = 0.0
        self.joint_test_hold_samples = []
        self.joint_test_last_feedback_sequence = -1
        self.joint_test_hold_active = False
        self.joint_test_hold_started_at = 0.0
        self.joint_test_cross_axis_peak_rad = 0.0
        self.joint_test_cross_axis_joint = ""
        self.joint_test_message = "joint rotation precheck has not run"
        self.returning_to_center = False
        self.return_owner = ""
        self.return_started_at = 0.0
        self.return_duration_sec = self.stop_ramp_sec
        self.return_start_positions = JOINT_NOMINAL_POS.copy()
        self.last_command_positions = JOINT_NOMINAL_POS.copy()
        self.safety_fault = False
        self.last_command_publish_at = 0.0
        self.last_feedback_wait_log_at = 0.0
        self.last_reset_wait_log_at = 0.0
        self.last_invalid_feedback_log_at = 0.0
        self.last_remote_conflict_log_at = 0.0
        self.last_start_message = "not started"

        self.csv_row_queue = Queue(maxsize=self.log_queue_capacity)
        self.csv_control_queue = SimpleQueue()
        self.csv_wakeup = Event()
        self.csv_force_shutdown = Event()
        self.csv_session_counter = 0
        self.csv_active_session = 0
        self.csv_last_enqueued_sequence = 0
        self.csv_next_log_elapsed = 0.0
        self.csv_dropped_since_warning = 0
        self.csv_last_drop_warning_at = 0.0
        self.csv_worker_stopped = False
        self.csv_thread = Thread(
            target=self._csv_writer_main,
            name="elf3-vibration-csv",
            daemon=True,
        )
        self.csv_thread.start()

        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self._timer_callback,
            callback_group=self.control_callback_group,
        )
        self.publisher_watchdog_group = MutuallyExclusiveCallbackGroup()
        self.publisher_watchdog_timer = self.create_timer(
            self.publisher_check_period_sec,
            self._publisher_watchdog_callback,
            callback_group=self.publisher_watchdog_group,
        )

        self.get_logger().info(
            "Vibration test configured: joint=%s amplitude=%.6f rad "
            "frequency=%.3f->%.3f Hz duration=%.3f s "
            "command_rate=%.1f Hz log_rate=%.1f Hz joint_precheck=%s"
            % (
                self.joint_name,
                self.amplitude_rad,
                self.start_frequency_hz,
                self.end_frequency_hz,
                self.duration_sec,
                self.control_rate_hz,
                self.log_rate_hz,
                self.joint_test_required,
            )
        )
        self.get_logger().info(self._remote_help_message())

    def _remote_help_message(self):
        return (
            "remote ready: X starts the joint precheck, X starts vibration "
            "after a pass, and X stops the active operation "
            "(motion_button_mode=%s)" % self.motion_button_mode
        )

    def _validate_parameters(self):
        if self.joint_name != "all" and self.joint_name not in JOINT_NAMES:
            raise ValueError(
                "unknown joint_name %r; use 'all' or one of: %s"
                % (self.joint_name, ", ".join(JOINT_NAMES))
            )

        positive_finite = (
            ("amplitude_rad", self.amplitude_rad),
            ("start_frequency_hz", self.start_frequency_hz),
            ("end_frequency_hz", self.end_frequency_hz),
            ("control_rate_hz", self.control_rate_hz),
            ("initialization_sec", self.initialization_sec),
            ("stop_ramp_sec", self.stop_ramp_sec),
            ("motion_command_resync_sec", self.motion_command_resync_sec),
            ("joint_test_amplitude_rad", self.joint_test_amplitude_rad),
            ("joint_test_move_sec", self.joint_test_move_sec),
            ("joint_test_hold_sec", self.joint_test_hold_sec),
            ("joint_test_min_motion_rad", self.joint_test_min_motion_rad),
            (
                "joint_test_start_tolerance_rad",
                self.joint_test_start_tolerance_rad,
            ),
            (
                "joint_test_tracking_tolerance_rad",
                self.joint_test_tracking_tolerance_rad,
            ),
            (
                "joint_test_center_tolerance_rad",
                self.joint_test_center_tolerance_rad,
            ),
            (
                "joint_test_cross_axis_limit_rad",
                self.joint_test_cross_axis_limit_rad,
            ),
            ("joint_test_pass_valid_sec", self.joint_test_pass_valid_sec),
            (
                "joint_test_confirmation_delay_sec",
                self.joint_test_confirmation_delay_sec,
            ),
            ("joint_state_timeout_sec", self.joint_state_timeout_sec),
            ("max_command_gap_sec", self.max_command_gap_sec),
            ("log_rate_hz", self.log_rate_hz),
            ("publisher_check_period_sec", self.publisher_check_period_sec),
            ("reset_response_timeout_sec", self.reset_response_timeout_sec),
        )
        for name, value in positive_finite:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and > 0" % name)

        nonnegative_finite = (
            ("duration_sec", self.duration_sec),
            ("kp_scale", self.kp_scale),
            ("kd_scale", self.kd_scale),
            ("joint_limit_margin_rad", self.joint_limit_margin_rad),
            (
                "joint_test_initial_settle_sec",
                self.joint_test_initial_settle_sec,
            ),
        )
        for name, value in nonnegative_finite:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and >= 0" % name)

        if self.log_queue_capacity <= 0:
            raise ValueError("log_queue_capacity must be > 0")
        if self.motion_button_mode not in {"toggle", "momentary"}:
            raise ValueError(
                "motion_button_mode must be 'toggle' or 'momentary'"
            )
        if self.joint_test_min_feedback_samples <= 0:
            raise ValueError("joint_test_min_feedback_samples must be > 0")
        if (
            self.joint_test_confirmation_delay_sec
            >= self.joint_test_pass_valid_sec
        ):
            raise ValueError(
                "joint_test_confirmation_delay_sec must be smaller than "
                "joint_test_pass_valid_sec"
            )
        if self.joint_test_min_motion_rad >= self.joint_test_amplitude_rad:
            raise ValueError(
                "joint_test_min_motion_rad must be smaller than "
                "joint_test_amplitude_rad"
            )
        if self.control_rate_hz * self.joint_test_move_sec < 20.0:
            raise ValueError(
                "joint_test_move_sec must contain at least 20 control samples"
            )
        if (
            self.control_rate_hz * self.joint_test_hold_sec
            < self.joint_test_min_feedback_samples + 1
        ):
            raise ValueError(
                "joint_test_hold_sec is too short for the requested number "
                "of distinct feedback samples"
            )
        if self.max_command_gap_sec < 1.0 / self.control_rate_hz:
            raise ValueError(
                "max_command_gap_sec must be at least one control period "
                "(%.6f seconds at %.1f Hz)"
                % (1.0 / self.control_rate_hz, self.control_rate_hz)
            )

        max_frequency = max(self.start_frequency_hz, self.end_frequency_hz)
        if self.control_rate_hz < 10.0 * max_frequency:
            raise ValueError(
                "control_rate_hz (%.3f) must be at least 10 times the "
                "maximum frequency (%.3f Hz); use at least %.3f Hz"
                % (
                    self.control_rate_hz,
                    max_frequency,
                    10.0 * max_frequency,
                )
            )
        if self.log_rate_hz > self.control_rate_hz:
            self.get_logger().warning(
                "log_rate_hz %.1f exceeds control_rate_hz %.1f; effective "
                "CSV rate is limited by the command rate"
                % (self.log_rate_hz, self.control_rate_hz)
            )
        if self.duration_sec == 0.0 and not math.isclose(
            self.start_frequency_hz, self.end_frequency_hz
        ):
            self.get_logger().warning(
                "duration_sec is 0, so the test is continuous and uses "
                "start_frequency_hz only"
            )
        if self.joint_test_required and self.auto_start:
            raise ValueError(
                "joint_test_required=true requires auto_start=false so the "
                "precheck and vibration each need an explicit confirmation"
            )

        if self.hardware_mode:
            if not self.require_joint_state:
                raise ValueError("hardware_mode requires require_joint_state=true")
            if self.auto_start:
                raise ValueError("hardware_mode requires auto_start=false")
            if (
                not self.joint_test_required
                and not self.allow_hardware_without_joint_test
            ):
                raise ValueError(
                    "hardware_mode requires joint_test_required=true unless "
                    "allow_hardware_without_joint_test=true is explicitly set"
                )
            if self.joint_test_required and not self.joint_test_verify_feedback:
                raise ValueError(
                    "hardware joint precheck requires "
                    "joint_test_verify_feedback=true"
                )
            if self.initialization_sec < 10.0:
                raise ValueError("hardware_mode requires initialization_sec >= 10 seconds")
            if self.joint_name == "all" and not self.allow_all_joints:
                raise ValueError(
                    "all-joint hardware testing requires allow_all_joints=true"
                )

            hardware_ceilings = (
                ("hardware_max_amplitude_rad", self.hardware_max_amplitude_rad),
                ("hardware_max_frequency_hz", self.hardware_max_frequency_hz),
                ("hardware_max_velocity_rad_s", self.hardware_max_velocity_rad_s),
                (
                    "hardware_max_acceleration_rad_s2",
                    self.hardware_max_acceleration_rad_s2,
                ),
                (
                    "hardware_max_control_rate_hz",
                    self.hardware_max_control_rate_hz,
                ),
                (
                    "hardware_max_joint_test_amplitude_rad",
                    self.hardware_max_joint_test_amplitude_rad,
                ),
            )
            for name, value in hardware_ceilings:
                # Positive infinity is intentionally accepted as an explicit
                # "no software ceiling" setting; NaN and non-positive values
                # are never meaningful limits.
                if math.isnan(value) or value <= 0.0:
                    raise ValueError("%s must be > 0" % name)

            dedicated_joint_test_limits = (
                (
                    "hardware_joint_test_min_move_sec",
                    self.hardware_joint_test_min_move_sec,
                ),
                (
                    "hardware_joint_test_min_response_ratio",
                    self.hardware_joint_test_min_response_ratio,
                ),
                (
                    "hardware_joint_test_max_tolerance_rad",
                    self.hardware_joint_test_max_tolerance_rad,
                ),
                (
                    "hardware_joint_test_max_velocity_rad_s",
                    self.hardware_joint_test_max_velocity_rad_s,
                ),
                (
                    "hardware_joint_test_max_acceleration_rad_s2",
                    self.hardware_joint_test_max_acceleration_rad_s2,
                ),
            )
            for name, value in dedicated_joint_test_limits:
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("%s must be finite and > 0" % name)
            if self.hardware_joint_test_min_response_ratio > 1.0:
                raise ValueError(
                    "hardware_joint_test_min_response_ratio must be <= 1"
                )

            if self.amplitude_rad > self.hardware_max_amplitude_rad:
                raise ValueError(
                    "amplitude_rad %.6f exceeds hardware_max_amplitude_rad %.6f"
                    % (self.amplitude_rad, self.hardware_max_amplitude_rad)
                )
            if self.joint_test_amplitude_rad > self.hardware_max_amplitude_rad:
                raise ValueError(
                    "joint_test_amplitude_rad %.6f exceeds "
                    "hardware_max_amplitude_rad %.6f"
                    % (
                        self.joint_test_amplitude_rad,
                        self.hardware_max_amplitude_rad,
                    )
                )
            if (
                self.joint_test_amplitude_rad
                > self.hardware_max_joint_test_amplitude_rad
            ):
                raise ValueError(
                    "joint_test_amplitude_rad %.6f exceeds the dedicated "
                    "hardware precheck limit %.6f"
                    % (
                        self.joint_test_amplitude_rad,
                        self.hardware_max_joint_test_amplitude_rad,
                    )
                )
            if self.joint_test_move_sec < self.hardware_joint_test_min_move_sec:
                raise ValueError(
                    "joint_test_move_sec %.6f is below the dedicated hardware "
                    "minimum %.6f"
                    % (
                        self.joint_test_move_sec,
                        self.hardware_joint_test_min_move_sec,
                    )
                )
            minimum_hardware_response = (
                self.hardware_joint_test_min_response_ratio
                * self.joint_test_amplitude_rad
            )
            if self.joint_test_min_motion_rad < minimum_hardware_response:
                raise ValueError(
                    "joint_test_min_motion_rad %.6f is below the dedicated "
                    "hardware minimum %.6f"
                    % (
                        self.joint_test_min_motion_rad,
                        minimum_hardware_response,
                    )
                )
            configured_tolerances = (
                self.joint_test_start_tolerance_rad,
                self.joint_test_tracking_tolerance_rad,
                self.joint_test_center_tolerance_rad,
                self.joint_test_cross_axis_limit_rad,
            )
            if max(configured_tolerances) > self.hardware_joint_test_max_tolerance_rad:
                raise ValueError(
                    "a joint precheck tolerance exceeds the dedicated hardware "
                    "maximum %.6f rad"
                    % self.hardware_joint_test_max_tolerance_rad
                )
            if max_frequency > self.hardware_max_frequency_hz:
                raise ValueError(
                    "maximum frequency %.3f Hz exceeds "
                    "hardware_max_frequency_hz %.3f"
                    % (max_frequency, self.hardware_max_frequency_hz)
                )
            if self.control_rate_hz > self.hardware_max_control_rate_hz:
                raise ValueError(
                    "control_rate_hz %.3f exceeds "
                    "hardware_max_control_rate_hz %.3f"
                    % (
                        self.control_rate_hz,
                        self.hardware_max_control_rate_hz,
                    )
                )

            peak_velocity = 2.0 * math.pi * max_frequency * self.amplitude_rad
            frequency_rate_hz_s = 0.0
            if self.duration_sec > 0.0:
                frequency_rate_hz_s = abs(
                    self.end_frequency_hz - self.start_frequency_hz
                ) / self.duration_sec
            peak_acceleration = self.amplitude_rad * (
                (2.0 * math.pi * max_frequency) ** 2
                + 2.0 * math.pi * frequency_rate_hz_s
            )
            joint_test_peak_velocity = (
                MINIMUM_JERK_MAX_SLOPE
                * self.joint_test_amplitude_rad
                / self.joint_test_move_sec
            )
            joint_test_peak_acceleration = (
                MINIMUM_JERK_MAX_CURVATURE
                * self.joint_test_amplitude_rad
                / (self.joint_test_move_sec * self.joint_test_move_sec)
            )
            if peak_velocity > self.hardware_max_velocity_rad_s:
                raise ValueError(
                    "requested peak velocity %.6f rad/s exceeds "
                    "hardware_max_velocity_rad_s %.6f"
                    % (peak_velocity, self.hardware_max_velocity_rad_s)
                )
            if peak_acceleration > self.hardware_max_acceleration_rad_s2:
                raise ValueError(
                    "requested peak acceleration %.6f rad/s^2 exceeds "
                    "hardware_max_acceleration_rad_s2 %.6f"
                    % (peak_acceleration, self.hardware_max_acceleration_rad_s2)
                )
            if joint_test_peak_velocity > self.hardware_max_velocity_rad_s:
                raise ValueError(
                    "joint precheck peak velocity %.6f rad/s exceeds "
                    "hardware_max_velocity_rad_s %.6f"
                    % (
                        joint_test_peak_velocity,
                        self.hardware_max_velocity_rad_s,
                    )
                )
            if (
                joint_test_peak_acceleration
                > self.hardware_max_acceleration_rad_s2
            ):
                raise ValueError(
                    "joint precheck peak acceleration %.6f rad/s^2 exceeds "
                    "hardware_max_acceleration_rad_s2 %.6f"
                    % (
                        joint_test_peak_acceleration,
                        self.hardware_max_acceleration_rad_s2,
                    )
                )
            if (
                joint_test_peak_velocity
                > self.hardware_joint_test_max_velocity_rad_s
            ):
                raise ValueError(
                    "joint precheck peak velocity %.6f rad/s exceeds the "
                    "dedicated hardware limit %.6f"
                    % (
                        joint_test_peak_velocity,
                        self.hardware_joint_test_max_velocity_rad_s,
                    )
                )
            if (
                joint_test_peak_acceleration
                > self.hardware_joint_test_max_acceleration_rad_s2
            ):
                raise ValueError(
                    "joint precheck peak acceleration %.6f rad/s^2 exceeds "
                    "the dedicated hardware limit %.6f"
                    % (
                        joint_test_peak_acceleration,
                        self.hardware_joint_test_max_acceleration_rad_s2,
                    )
                )
            if self.control_rate_hz > 500.0:
                self.get_logger().warning(
                    "control_rate_hz %.1f is high for a Python ROS 2 node; "
                    "verify command timing before enabling hardware"
                    % self.control_rate_hz
                )
            if all(
                value >= 10000.0
                for name, value in hardware_ceilings
                if name
                in {
                    "hardware_max_amplitude_rad",
                    "hardware_max_frequency_hz",
                    "hardware_max_velocity_rad_s",
                    "hardware_max_acceleration_rad_s2",
                }
            ):
                self.get_logger().warning(
                    "hardware_max_* ceilings are extremely high; the software "
                    "amplitude/frequency/velocity/acceleration guard is effectively "
                    "disabled"
                )

    def _joint_callback(self, msg):
        if not msg.position:
            return

        with self.feedback_lock:
            updated = self.measured_positions.copy()
        seen = np.zeros(DOF_NUM, dtype=bool)
        invalid_names = []
        if msg.name:
            positions_by_name = dict(zip(msg.name, msg.position))
            for index, name in enumerate(JOINT_NAMES):
                if name in positions_by_name:
                    value = float(positions_by_name[name])
                    if math.isfinite(value):
                        updated[index] = value
                        seen[index] = True
                    else:
                        invalid_names.append(name)
            if not np.any(seen) and not invalid_names:
                return
        elif len(msg.position) >= DOF_NUM:
            received = np.asarray(msg.position[:DOF_NUM], dtype=np.float64)
            finite = np.isfinite(received)
            updated[finite] = received[finite]
            seen[finite] = True
            invalid_names.extend(
                name
                for index, name in enumerate(JOINT_NAMES)
                if not finite[index]
            )
        else:
            return

        received_at = time.monotonic()
        if np.any(seen):
            with self.feedback_lock:
                self.measured_positions[seen] = updated[seen]
                self.joint_state_seen |= seen
                self.joint_state_last_seen_at[seen] = received_at
                self.joint_feedback_sequence[seen] += 1
                self.has_joint_state = bool(np.all(self.joint_state_seen))

        if invalid_names:
            reason = "non-finite joint feedback received for: %s" % ", ".join(
                invalid_names
            )
            if self.hardware_mode:
                self._latch_safety_fault(reason)
            elif received_at - self.last_invalid_feedback_log_at >= 1.0:
                self.get_logger().error(reason + "; invalid samples were ignored")
                self.last_invalid_feedback_log_at = received_at

    def _motion_callback(self, msg):
        now = time.monotonic()
        if self.count_publishers(self.motion_topic) > 1:
            if now - self.last_remote_conflict_log_at >= 1.0:
                self.get_logger().error(
                    "multiple motion_commands publishers detected; remote "
                    "input is ignored. Stop duplicate remote controllers and "
                    "use the vibration services if needed"
                )
                self.last_remote_conflict_log_at = now
            return
        with self.state_lock:
            activated = self.remote_button.update(msg.btn_9 != 0, now)
            initialized = self.reset_stage == 2
            enabled = self.test_enabled
            joint_test_running = self.joint_test_running
            self._expire_joint_test_permission_locked(time.monotonic())
            joint_test_needed = (
                self.joint_test_required and not self.joint_test_passed
            )

        if activated and initialized:
            if enabled:
                self._stop_test("btn_9")
            elif joint_test_running:
                self._cancel_joint_rotation_test("btn_9")
            elif joint_test_needed:
                self._start_joint_rotation_test()
            else:
                self._start_test(request_received_at=time.monotonic())

    def _enable_service_callback(self, request, response):
        if request.data:
            response.success = self._start_test(
                request_received_at=time.monotonic()
            )
            with self.state_lock:
                response.message = self.last_start_message
            return response

        cancelled_precheck = self._cancel_joint_rotation_test(
            "vibration disable service"
        )
        stopped = self._stop_test("enable service")
        response.success = True
        with self.state_lock:
            returning = self.returning_to_center
            return_owner = self.return_owner
            joint_test_running = self.joint_test_running
        if cancelled_precheck:
            response.message = (
                "joint rotation precheck cancelled by the vibration stop "
                "request; returning smoothly to center"
            )
        elif stopped:
            response.message = (
                "vibration excitation stopped; returning smoothly to center"
            )
        elif joint_test_running:
            response.message = (
                "vibration is stopped; joint rotation precheck is still "
                "running. Use /joint_rotation_test_enable data=false to cancel it"
            )
        elif returning and return_owner == "vibration":
            response.message = (
                "vibration is already stopped and returning smoothly to center"
            )
        elif returning and return_owner == "joint_precheck":
            response.message = (
                "vibration is stopped; a cancelled joint precheck is returning "
                "smoothly to center"
            )
        else:
            response.message = (
                "vibration test was already stopped; holding center position"
            )
        return response

    def _joint_test_service_callback(self, request, response):
        if request.data:
            response.success = self._start_joint_rotation_test()
            with self.state_lock:
                response.message = self.joint_test_message
            return response

        with self.state_lock:
            if self.test_enabled:
                response.success = False
                response.message = (
                    "vibration remains active; stop it with "
                    "/vibration_test_enable data=false"
                )
                return response
        cancelled = self._cancel_joint_rotation_test("enable service")
        response.success = True
        with self.state_lock:
            if cancelled:
                response.message = (
                    "joint rotation precheck cancelled; returning smoothly "
                    "to center; vibration remains blocked"
                )
            elif self.joint_test_passed:
                self.joint_test_passed = False
                self.joint_test_passed_at = 0.0
                self.joint_test_message = (
                    "joint rotation precheck permission revoked; run the "
                    "precheck again before vibration"
                )
                response.message = self.joint_test_message
            elif (
                self.returning_to_center
                and self.return_owner == "joint_precheck"
            ):
                response.message = (
                    "joint rotation precheck is already cancelled and "
                    "returning smoothly to center"
                )
            else:
                response.message = self._joint_test_status_message_locked()
        return response

    def _joint_test_status_callback(self, _request, response):
        response.success = True
        with self.state_lock:
            response.message = self._joint_test_status_message_locked()
        return response

    def _timer_callback(self):
        now = time.monotonic()

        with self.state_lock:
            self._expire_joint_test_permission_locked(now)
            if self.safety_fault:
                return

            if not self._command_publish_gap_is_safe(now):
                return

            if self.require_joint_state and not self._joint_feedback_ready(now):
                if self.reset_stage == 2:
                    self._latch_safety_fault(
                        "joint feedback became incomplete or stale"
                    )
                else:
                    if self.reset_stage == 1:
                        self.initialization_started_at = now
                    self._log_waiting_for_feedback(now)
                return

            if self.reset_stage == 0:
                if self.reset_pending_step == 1:
                    if self._reset_request_completed(1, now):
                        self.initialization_started_at = time.monotonic()
                        self.reset_stage = 1
                        self._queue_diagnostic_log(
                            "info",
                            "robot reset 1 acknowledged; soft initialization started"
                        )
                elif self._call_robot_reset(1, False, now):
                    self._queue_diagnostic_log("info", "robot reset 1 sent")
                return

            if self.reset_stage == 1:
                elapsed = now - self.initialization_started_at
                ramp = min(elapsed / self.initialization_sec, 1.0)
                if not self._publish_command(
                    JOINT_NOMINAL_POS,
                    JOINT_KP * ramp,
                    JOINT_KD,
                ):
                    return

                if elapsed >= self.initialization_sec:
                    if self.reset_pending_step == 2:
                        if self._reset_request_completed(2, now):
                            self.reset_stage = 2
                            self._queue_diagnostic_log(
                                "info",
                                "robot reset 2 acknowledged; initialization complete"
                            )
                            if self.auto_start:
                                self._start_test()
                    elif self._call_robot_reset(
                        2,
                        self.release_suspension,
                        now,
                    ):
                        self._queue_diagnostic_log(
                            "info",
                            "robot reset 2 sent (release_suspension=%s)"
                            % self.release_suspension
                        )
                return

            command = self.center_positions.copy()
            command_kp = JOINT_KP
            command_kd = JOINT_KD
            log_sample = None
            if self.joint_test_running:
                command = self._joint_rotation_test_command(now)
                if command is None:
                    return
            elif self.test_enabled:
                elapsed = now - self.test_started_at
                if self.duration_sec > 0.0 and elapsed >= self.duration_sec:
                    self._stop_test("duration complete")
                else:
                    frequency, phase = self._frequency_and_phase(elapsed)
                    command[self.active_joint_indices] += (
                        self.amplitude_rad * math.sin(phase)
                    )
                    np.clip(
                        command,
                        JOINT_POSITION_MIN + self.joint_limit_margin_rad,
                        JOINT_POSITION_MAX - self.joint_limit_margin_rad,
                        out=command,
                    )
                    log_sample = (elapsed, frequency)
            else:
                command, command_kp, command_kd = self._idle_command(now)

            if self.returning_to_center:
                command = self._return_to_center_command(now)
                command_kp = JOINT_KP
                command_kd = JOINT_KD

            if (
                self._publish_command(command, command_kp, command_kd)
                and log_sample
            ):
                self._write_csv(log_sample[0], log_sample[1], command, now)

    def _idle_command(self, _now):
        """Return the command held when vibration and precheck are inactive.

        Subclasses can provide another mutually exclusive test mode without
        creating a second actuator publisher or duplicating reset/safety code.
        """
        return self.center_positions.copy(), JOINT_KP, JOINT_KD

    def _publisher_watchdog_callback(self):
        with self.state_lock:
            if self.safety_fault:
                return
        try:
            publisher_count = self.count_publishers(self.actuator_topic)
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warning(
                    "cannot inspect actuator publisher count: %s" % exc
                )
            return
        if publisher_count > 1:
            self._latch_safety_fault(
                "multiple publishers detected on %s (count=%d)"
                % (self.actuator_topic, publisher_count)
            )

    def _start_joint_rotation_test(self):
        now = time.monotonic()
        with self.state_lock:
            self._expire_joint_test_permission_locked(now)
            if self.safety_fault:
                return self._reject_joint_test(
                    "cannot start joint rotation precheck after a latched "
                    "safety fault; restart required"
                )
            if self.reset_stage != 2:
                return self._reject_joint_test(
                    "robot initialization is not complete; wait for "
                    "'robot reset 2 acknowledged; initialization complete'"
                )
            if self.test_enabled:
                return self._reject_joint_test(
                    "cannot run joint rotation precheck while vibration is active"
                )
            if self.returning_to_center:
                return self._reject_joint_test(
                    "cannot start joint rotation precheck while returning to "
                    "center; wait for 'smooth return complete'"
                )
            if self.joint_test_running:
                self.joint_test_message = self._joint_test_status_message_locked()
                return True
            if (
                self.joint_test_passed
                and now - self.joint_test_passed_at
                <= self.joint_test_pass_valid_sec
            ):
                self.joint_test_message = (
                    "joint rotation precheck already passed; confirm safety, "
                    "then start vibration"
                )
                return True
            if (
                self.hardware_mode
                and (
                    self.last_command_publish_at <= 0.0
                    or now - self.last_command_publish_at
                    > self.max_command_gap_sec
                )
            ):
                return self._reject_joint_test(
                    "cannot start joint rotation precheck because the hardware "
                    "hold command is not fresh"
                )
            command_center = self.last_command_positions.copy()

        if (
            self.joint_test_verify_feedback
            and not self._joint_feedback_ready(now)
        ):
            return self._reject_joint_test(
                "cannot start joint rotation precheck without complete fresh "
                "joint feedback"
            )

        try:
            publisher_count = self.count_publishers(self.actuator_topic)
        except Exception as exc:
            return self._reject_joint_test(
                "cannot verify actuator publisher count: %s" % exc
            )
        if publisher_count > 1:
            return self._reject_joint_test(
                "multiple publishers detected on %s (count=%d)"
                % (self.actuator_topic, publisher_count)
            )

        with self.feedback_lock:
            if self.has_joint_state:
                feedback_baseline = self.measured_positions.copy()
            else:
                feedback_baseline = command_center.copy()

        start_error = np.abs(feedback_baseline - command_center)
        worst_start_index = int(np.argmax(start_error))
        if start_error[worst_start_index] > self.joint_test_start_tolerance_rad:
            return self._reject_joint_test(
                "cannot start joint rotation precheck because %s feedback "
                "differs from the current hold command by %.6f rad, exceeding "
                "the %.6f rad start tolerance"
                % (
                    JOINT_NAMES[worst_start_index],
                    start_error[worst_start_index],
                    self.joint_test_start_tolerance_rad,
                )
            )

        all_indices = np.arange(DOF_NUM, dtype=np.int64)
        precheck_violations = self._envelope_violation_details(
            command_center,
            self.joint_test_amplitude_rad,
            all_indices,
        )
        if precheck_violations:
            return self._reject_joint_test(
                "joint rotation precheck envelope exceeds software position "
                "limits: " + "; ".join(precheck_violations)
            )
        precheck_feedback_violations = self._envelope_violation_details(
            feedback_baseline,
            self.joint_test_amplitude_rad,
            all_indices,
        )
        if precheck_feedback_violations:
            return self._reject_joint_test(
                "joint rotation precheck envelope exceeds software position "
                "limits around the measured feedback baseline: "
                + "; ".join(precheck_feedback_violations)
            )

        vibration_violations = self._envelope_violation_details(
            command_center,
            self.amplitude_rad,
            self.active_joint_indices,
        )
        if vibration_violations:
            return self._reject_joint_test(
                "later vibration envelope is unsafe at the current center, so "
                "the joint precheck was not started: "
                + "; ".join(vibration_violations)
            )
        vibration_feedback_violations = self._envelope_violation_details(
            feedback_baseline,
            self.amplitude_rad,
            self.active_joint_indices,
        )
        if vibration_feedback_violations:
            return self._reject_joint_test(
                "later vibration envelope is unsafe around the measured "
                "feedback baseline, so the joint precheck was not started: "
                + "; ".join(vibration_feedback_violations)
            )

        with self.state_lock:
            if self.safety_fault or self.reset_stage != 2:
                return self._reject_joint_test(
                    "robot state changed while starting joint rotation precheck; "
                    "try again"
                )
            if self.test_enabled or self.returning_to_center:
                return self._reject_joint_test(
                    "controller state changed while starting joint rotation "
                    "precheck; try again"
                )
            self.center_positions[:] = command_center
            self.joint_test_center_positions[:] = command_center
            self.joint_test_feedback_baseline_positions[:] = feedback_baseline
            self.joint_test_baseline_finalized = False
            self.joint_test_running = True
            self.joint_test_passed = False
            self.joint_test_passed_at = 0.0
            self.joint_test_failed = False
            self.joint_test_failure_reason = ""
            self.joint_test_current_index = 0
            self.joint_test_segment_index = 0
            self.joint_test_joint_started_at = (
                time.monotonic() + self.joint_test_initial_settle_sec
            )
            self._reset_joint_test_observations_locked()
            total_sec = (
                self.joint_test_initial_settle_sec
                + DOF_NUM * self.joint_test_cycle_sec
            )
            self.joint_test_message = (
                "joint rotation precheck started: 29 joints, approximately "
                "%.1f seconds; vibration remains disabled" % total_sec
            )

        self._queue_diagnostic_log(
            "info",
            "joint rotation precheck started: 29 joints, amplitude=%.6f rad, "
            "move=%.3f s, hold=%.3f s, approximately %.1f seconds; "
            "vibration remains disabled"
            % (
                self.joint_test_amplitude_rad,
                self.joint_test_move_sec,
                self.joint_test_hold_sec,
                total_sec,
            ),
        )
        self._queue_joint_test_joint_start_log()
        return True

    def _cancel_joint_rotation_test(self, reason):
        with self.state_lock:
            if not self.joint_test_running:
                return False
            joint_number = self.joint_test_current_index + 1
            joint_name = JOINT_NAMES[self.joint_test_current_index]
            self.joint_test_running = False
            self.joint_test_passed = False
            self.joint_test_passed_at = 0.0
            self.joint_test_failed = False
            self.joint_test_failure_reason = ""
            self.joint_test_message = (
                "joint rotation precheck cancelled at [%02d/%02d] %s; "
                "returning to center; vibration remains blocked"
                % (joint_number, DOF_NUM, joint_name)
            )
            self._begin_smooth_return_locked("joint_precheck")
        self._queue_diagnostic_log(
            "warning", "%s (%s)" % (self.joint_test_message, reason)
        )
        return True

    def _joint_rotation_test_command(self, now):
        with self.state_lock:
            if not self.joint_test_running:
                return self.center_positions.copy()
            if (
                self.joint_test_verify_feedback
                and not self._joint_feedback_ready(now)
            ):
                self._fail_joint_rotation_test(
                    "joint feedback became incomplete or stale during precheck"
                )
                return None
            if now < self.joint_test_joint_started_at:
                return self.joint_test_center_positions.copy()
            if not self.joint_test_baseline_finalized:
                if self.joint_test_verify_feedback:
                    with self.feedback_lock:
                        settled_feedback = self.measured_positions.copy()
                    settled_error = np.abs(
                        settled_feedback - self.joint_test_center_positions
                    )
                    worst_index = int(np.argmax(settled_error))
                    if (
                        settled_error[worst_index]
                        > self.joint_test_start_tolerance_rad
                    ):
                        self._fail_joint_rotation_test(
                            "%s feedback remained %.6f rad away from the hold "
                            "command after the initial settle; limit is %.6f rad"
                            % (
                                JOINT_NAMES[worst_index],
                                settled_error[worst_index],
                                self.joint_test_start_tolerance_rad,
                            )
                        )
                        return None
                    self.joint_test_feedback_baseline_positions[:] = (
                        settled_feedback
                    )
                self.joint_test_baseline_finalized = True

            self._update_joint_test_cross_axis_peak_locked()
            if (
                self.joint_test_cross_axis_peak_rad
                >= self.joint_test_cross_axis_limit_rad
            ):
                joint_name = JOINT_NAMES[self.joint_test_current_index]
                self._fail_joint_rotation_test(
                    "%s movement caused passive joint %s to deviate %.6f rad; "
                    "cross-axis limit is %.6f rad (possible mapping/coupling issue)"
                    % (
                        joint_name,
                        self.joint_test_cross_axis_joint,
                        self.joint_test_cross_axis_peak_rad,
                        self.joint_test_cross_axis_limit_rad,
                    )
                )
                return None

            elapsed = now - self.joint_test_joint_started_at
            segment_duration = (
                self.joint_test_move_sec + self.joint_test_hold_sec
            )
            segment_index = min(int(elapsed / segment_duration), 4)

            if segment_index > self.joint_test_segment_index:
                if segment_index != self.joint_test_segment_index + 1:
                    self._fail_joint_rotation_test(
                        "control timing skipped a precheck trajectory segment"
                    )
                    return None
                if self.joint_test_hold_active:
                    self._collect_joint_test_feedback_locked()
                if not self._evaluate_joint_test_segment_locked(
                    self.joint_test_segment_index
                ):
                    return None
                self.joint_test_segment_index = segment_index
                self.joint_test_hold_samples = []
                self.joint_test_hold_active = False
                self.joint_test_hold_started_at = 0.0

            if segment_index >= 4:
                if not self._complete_joint_test_joint_locked(now):
                    return None
                return self.joint_test_center_positions.copy()

            segment_elapsed = elapsed - segment_index * segment_duration
            if segment_elapsed >= self.joint_test_move_sec:
                if not self.joint_test_hold_active:
                    hold_started_at = (
                        self.joint_test_joint_started_at
                        + segment_index * segment_duration
                        + self.joint_test_move_sec
                    )
                    self._start_joint_test_hold_locked(hold_started_at)
                else:
                    self._collect_joint_test_feedback_locked()

            command = self.joint_test_center_positions.copy()
            joint_index = self.joint_test_current_index
            command[joint_index] += self._joint_test_offset(
                segment_index,
                segment_elapsed,
            )
            return command

    def _joint_test_offset(self, segment_index, segment_elapsed):
        amplitude = self.joint_test_amplitude_rad
        waypoints = (0.0, amplitude, 0.0, -amplitude, 0.0)
        start = waypoints[segment_index]
        target = waypoints[segment_index + 1]
        if segment_elapsed >= self.joint_test_move_sec:
            return target
        u = min(max(segment_elapsed / self.joint_test_move_sec, 0.0), 1.0)
        smooth = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        return start + (target - start) * smooth

    def _start_joint_test_hold_locked(self, hold_started_at):
        self.joint_test_hold_active = True
        self.joint_test_hold_started_at = hold_started_at
        self.joint_test_hold_samples = []
        if not self.joint_test_verify_feedback:
            return
        joint_index = self.joint_test_current_index
        with self.feedback_lock:
            self.joint_test_last_feedback_sequence = int(
                self.joint_feedback_sequence[joint_index]
            )

    def _collect_joint_test_feedback_locked(self):
        if not self.joint_test_verify_feedback:
            return
        joint_index = self.joint_test_current_index
        with self.feedback_lock:
            sequence = int(self.joint_feedback_sequence[joint_index])
            if sequence == self.joint_test_last_feedback_sequence:
                return
            received_at = float(self.joint_state_last_seen_at[joint_index])
            if received_at < self.joint_test_hold_started_at:
                return
            measured = self.measured_positions.copy()
        self.joint_test_last_feedback_sequence = sequence
        delta = float(
            measured[joint_index]
            - self.joint_test_feedback_baseline_positions[joint_index]
        )
        self.joint_test_hold_samples.append(delta)

    def _update_joint_test_cross_axis_peak_locked(self):
        joint_index = self.joint_test_current_index
        with self.feedback_lock:
            measured = self.measured_positions.copy()
        passive_delta = np.abs(
            measured - self.joint_test_feedback_baseline_positions
        )
        passive_delta[joint_index] = 0.0
        passive_index = int(np.argmax(passive_delta))
        passive_peak = float(passive_delta[passive_index])
        if passive_peak > self.joint_test_cross_axis_peak_rad:
            self.joint_test_cross_axis_peak_rad = passive_peak
            self.joint_test_cross_axis_joint = JOINT_NAMES[passive_index]

    def _evaluate_joint_test_segment_locked(self, segment_index):
        if not self.joint_test_verify_feedback:
            return True
        joint_name = JOINT_NAMES[self.joint_test_current_index]
        sample_count = len(self.joint_test_hold_samples)
        if sample_count < self.joint_test_min_feedback_samples:
            return self._fail_joint_rotation_test(
                "%s segment %d received only %d distinct feedback samples; "
                "at least %d required"
                % (
                    joint_name,
                    segment_index + 1,
                    sample_count,
                    self.joint_test_min_feedback_samples,
                )
            )

        median_delta = float(np.median(self.joint_test_hold_samples))
        amplitude = self.joint_test_amplitude_rad
        if segment_index == 0:
            self.joint_test_observed_positive_rad = median_delta
            if (
                median_delta < self.joint_test_min_motion_rad
                or abs(median_delta - amplitude)
                > self.joint_test_tracking_tolerance_rad
            ):
                return self._fail_joint_rotation_test(
                    "%s positive response %.6f rad did not reach the "
                    "required +%.6f rad motion within %.6f rad tracking "
                    "tolerance"
                    % (
                        joint_name,
                        median_delta,
                        self.joint_test_min_motion_rad,
                        self.joint_test_tracking_tolerance_rad,
                    )
                )
        elif segment_index == 2:
            self.joint_test_observed_negative_rad = -median_delta
            if (
                median_delta > -self.joint_test_min_motion_rad
                or abs(median_delta + amplitude)
                > self.joint_test_tracking_tolerance_rad
            ):
                return self._fail_joint_rotation_test(
                    "%s negative response %.6f rad did not reach the "
                    "required -%.6f rad motion within %.6f rad tracking "
                    "tolerance"
                    % (
                        joint_name,
                        median_delta,
                        self.joint_test_min_motion_rad,
                        self.joint_test_tracking_tolerance_rad,
                    )
                )
        elif abs(median_delta) > self.joint_test_center_tolerance_rad:
            return self._fail_joint_rotation_test(
                "%s failed to return to center: median offset %.6f rad "
                "exceeds %.6f rad"
                % (
                    joint_name,
                    median_delta,
                    self.joint_test_center_tolerance_rad,
                )
            )

        return True

    def _complete_joint_test_joint_locked(self, now):
        joint_number = self.joint_test_current_index + 1
        joint_name = JOINT_NAMES[self.joint_test_current_index]
        if self.joint_test_verify_feedback:
            result = (
                "joint rotation precheck PASS [%02d/%02d] %s: "
                "+%.6f / -%.6f rad"
                % (
                    joint_number,
                    DOF_NUM,
                    joint_name,
                    self.joint_test_observed_positive_rad,
                    self.joint_test_observed_negative_rad,
                )
            )
        else:
            result = (
                "joint rotation precheck command complete [%02d/%02d] %s "
                "(feedback verification disabled)"
                % (joint_number, DOF_NUM, joint_name)
            )
        self._queue_diagnostic_log("info", result)

        self.joint_test_current_index += 1
        if self.joint_test_current_index >= DOF_NUM:
            self.joint_test_running = False
            self.joint_test_passed = True
            self.joint_test_failed = False
            self.joint_test_passed_at = now
            self.joint_test_message = (
                "JOINT ROTATION PRECHECK PASSED: 29/29 joints responded; "
                "vibration remains stopped. Wait at least %.1f seconds, "
                "confirm safety, then call /vibration_test_enable data=true "
                "or press X" % self.joint_test_confirmation_delay_sec
            )
            self._queue_diagnostic_log("info", self.joint_test_message)
            return True

        self.joint_test_segment_index = 0
        self.joint_test_joint_started_at = now
        self._reset_joint_test_observations_locked()
        self._queue_joint_test_joint_start_log()
        return True

    def _reset_joint_test_observations_locked(self):
        self.joint_test_observed_positive_rad = 0.0
        self.joint_test_observed_negative_rad = 0.0
        self.joint_test_hold_samples = []
        self.joint_test_last_feedback_sequence = -1
        self.joint_test_hold_active = False
        self.joint_test_hold_started_at = 0.0
        self.joint_test_cross_axis_peak_rad = 0.0
        self.joint_test_cross_axis_joint = ""

    def _queue_joint_test_joint_start_log(self):
        with self.state_lock:
            joint_number = self.joint_test_current_index + 1
            joint_name = JOINT_NAMES[self.joint_test_current_index]
        self._queue_diagnostic_log(
            "info",
            "joint rotation precheck [%02d/%02d] %s: "
            "center -> positive -> center -> negative -> center"
            % (joint_number, DOF_NUM, joint_name),
        )

    def _fail_joint_rotation_test(self, reason):
        with self.state_lock:
            if not self.joint_test_running:
                return False
            joint_number = self.joint_test_current_index + 1
            joint_name = JOINT_NAMES[self.joint_test_current_index]
            self.joint_test_running = False
            self.joint_test_passed = False
            self.joint_test_passed_at = 0.0
            self.joint_test_failed = True
            self.joint_test_failure_reason = reason
            self.joint_test_message = (
                "JOINT ROTATION PRECHECK FAILED [%02d/%02d] %s: %s; "
                "vibration remains blocked"
                % (joint_number, DOF_NUM, joint_name, reason)
            )
        self._latch_safety_fault(self.joint_test_message)
        return False

    def _reject_joint_test(self, reason):
        with self.state_lock:
            self.joint_test_message = reason
        self.get_logger().error(reason)
        return False

    def _joint_test_status_message_locked(self):
        now = time.monotonic()
        self._expire_joint_test_permission_locked(now)
        if self.joint_test_running:
            return (
                "joint rotation precheck RUNNING [%02d/%02d] %s, segment "
                "%d/4; vibration remains blocked"
                % (
                    self.joint_test_current_index + 1,
                    DOF_NUM,
                    JOINT_NAMES[self.joint_test_current_index],
                    self.joint_test_segment_index + 1,
                )
            )
        if self.joint_test_passed:
            elapsed = now - self.joint_test_passed_at
            if elapsed < self.joint_test_confirmation_delay_sec:
                return (
                    "joint rotation precheck PASSED; vibration remains "
                    "stopped. Wait another %.1f seconds, review the result, "
                    "then send a new vibration confirmation"
                    % (self.joint_test_confirmation_delay_sec - elapsed)
                )
            remaining = max(
                0.0,
                self.joint_test_pass_valid_sec
                - elapsed,
            )
            return (
                "joint rotation precheck PASSED; confirmation valid for "
                "another %.1f seconds" % remaining
            )
        if self.joint_test_failed:
            return "joint rotation precheck FAILED: %s" % self.joint_test_failure_reason
        return self.joint_test_message

    def _expire_joint_test_permission_locked(self, now):
        if (
            self.joint_test_passed
            and now - self.joint_test_passed_at
            > self.joint_test_pass_valid_sec
        ):
            self.joint_test_passed = False
            self.joint_test_passed_at = 0.0
            self.joint_test_message = (
                "joint rotation precheck permission expired; run it again"
            )
            return True
        return False

    def _envelope_violation_details(
        self, center, amplitude, indices, slack_rad=0.0
    ):
        lower = JOINT_POSITION_MIN + self.joint_limit_margin_rad
        upper = JOINT_POSITION_MAX - self.joint_limit_margin_rad
        active_center = center[indices]
        active_lower = lower[indices]
        active_upper = upper[indices]
        available = np.minimum(
            active_center - active_lower,
            active_upper - active_center,
        )
        violations = np.flatnonzero(
            np.logical_or(
                np.logical_not(np.isfinite(active_center)),
                amplitude > available + slack_rad + 1.0e-12,
            )
        )
        details = []
        for local_index in violations:
            joint_index = int(indices[local_index])
            details.append(
                "%s: center=%.6f rad, allowed=[%.6f, %.6f] rad, "
                "requested_amplitude=%.6f rad, "
                "max_symmetric_amplitude=%.6f rad"
                % (
                    JOINT_NAMES[joint_index],
                    active_center[local_index],
                    active_lower[local_index],
                    active_upper[local_index],
                    amplitude,
                    max(0.0, float(available[local_index])),
                )
            )
        return details

    def _frequency_and_phase(self, elapsed):
        if self.duration_sec <= 0.0:
            frequency = self.start_frequency_hz
            phase = 2.0 * math.pi * frequency * elapsed
            return frequency, phase

        sweep_rate = (
            self.end_frequency_hz - self.start_frequency_hz
        ) / self.duration_sec
        frequency = self.start_frequency_hz + sweep_rate * elapsed
        cycles = (
            self.start_frequency_hz * elapsed
            + 0.5 * sweep_rate * elapsed * elapsed
        )
        return frequency, 2.0 * math.pi * cycles

    def _start_test(
        self,
        request_received_at=None,
        envelope_slack_rad=0.0,
        center_override=None,
    ):
        now = time.monotonic()
        if center_override is not None:
            center_override = np.asarray(center_override, dtype=np.float64)
            if (
                center_override.shape != (DOF_NUM,)
                or not np.all(np.isfinite(center_override))
            ):
                return self._reject_start(
                    "invalid vibration center override; expected %d finite "
                    "joint positions" % DOF_NUM
                )
        with self.state_lock:
            if self.safety_fault:
                return self._reject_start(
                    "cannot start after a latched safety fault; restart required"
                )
            if self.reset_stage != 2:
                return self._reject_start(
                    "robot initialization is not complete; wait for "
                    "'robot reset 2 acknowledged; initialization complete'"
                )
            if self.test_enabled:
                self.last_start_message = "vibration test is already running"
                return True
            precheck_reason = self._vibration_precheck_gate_reason_locked(
                now,
                request_received_at,
            )
            if precheck_reason:
                return self._reject_start(precheck_reason)
            if self.returning_to_center:
                return self._reject_start(
                    "cannot start while returning to center; wait for completion"
                )
            if (
                self.hardware_mode
                and (
                    self.last_command_publish_at <= 0.0
                    or now - self.last_command_publish_at
                    > self.max_command_gap_sec
                )
            ):
                return self._reject_start(
                    "cannot start because the hardware hold command is not fresh"
                )

        if self.require_joint_state and not self._joint_feedback_ready(now):
            return self._reject_start(
                "cannot start without complete fresh joint feedback"
            )

        try:
            publisher_count = self.count_publishers(self.actuator_topic)
        except Exception as exc:
            return self._reject_start(
                "cannot verify actuator publisher count: %s" % exc
            )
        if publisher_count > 1:
            return self._reject_start(
                "multiple publishers detected on %s (count=%d)"
                % (self.actuator_topic, publisher_count)
            )

        with self.state_lock:
            use_precheck_center = (
                self.joint_test_required and self.joint_test_passed
            )
            precheck_center = self.joint_test_center_positions.copy()
            precheck_feedback_baseline = (
                self.joint_test_feedback_baseline_positions.copy()
            )
        with self.feedback_lock:
            has_joint_state = self.has_joint_state
            measured_center = self.measured_positions.copy()

        if center_override is not None:
            candidate_center = center_override.copy()
        elif use_precheck_center:
            candidate_center = precheck_center
            if has_joint_state:
                center_error = np.abs(
                    measured_center - precheck_feedback_baseline
                )
                worst_index = int(np.argmax(center_error))
                if (
                    center_error[worst_index]
                    > self.joint_test_center_tolerance_rad
                ):
                    with self.state_lock:
                        self.joint_test_passed = False
                        self.joint_test_passed_at = 0.0
                        self.joint_test_message = (
                            "robot moved away from the prechecked center; "
                            "precheck permission invalidated"
                        )
                    return self._reject_start(
                        "robot moved away from the prechecked center: %s error "
                        "%.6f rad exceeds %.6f rad; run the joint rotation "
                        "precheck again"
                        % (
                            JOINT_NAMES[worst_index],
                            center_error[worst_index],
                            self.joint_test_center_tolerance_rad,
                        )
                    )
        elif has_joint_state:
            candidate_center = measured_center
        else:
            candidate_center = JOINT_NOMINAL_POS.copy()

        violations = self._envelope_violation_details(
            candidate_center,
            self.amplitude_rad,
            self.active_joint_indices,
            slack_rad=envelope_slack_rad,
        )
        if violations:
            return self._reject_start(
                "vibration envelope exceeds software position limits: "
                + "; ".join(violations)
            )
        if use_precheck_center:
            feedback_violations = self._envelope_violation_details(
                precheck_feedback_baseline,
                self.amplitude_rad,
                self.active_joint_indices,
                slack_rad=envelope_slack_rad,
            )
            if feedback_violations:
                return self._reject_start(
                    "vibration envelope exceeds software position limits "
                    "around the prechecked feedback baseline: "
                    + "; ".join(feedback_violations)
                )

        with self.state_lock:
            if self.safety_fault:
                return self._reject_start(
                    "cannot start after a latched safety fault; restart required"
                )
            if self.reset_stage != 2:
                return self._reject_start(
                    "robot initialization changed while starting; try again"
                )
            if self.test_enabled:
                self.last_start_message = "vibration test is already running"
                return True
            precheck_reason = self._vibration_precheck_gate_reason_locked(
                time.monotonic(),
                request_received_at,
            )
            if precheck_reason:
                return self._reject_start(precheck_reason)
            if self.returning_to_center:
                return self._reject_start(
                    "cannot start while returning to center; wait for completion"
                )
            self.center_positions[:] = candidate_center
            self.test_started_at = time.monotonic()
            self._open_csv()
            self.test_enabled = True
            self.last_start_message = "vibration test started"
            if self.joint_test_required:
                self.joint_test_passed = False
                self.joint_test_passed_at = 0.0
                self.joint_test_message = (
                    "joint rotation precheck permission consumed by the "
                    "current vibration test"
                )

        self.get_logger().info(
            "vibration test started on %s (%d joints)"
            % (self.joint_name, len(self.active_joint_indices))
        )
        return True

    def _vibration_precheck_gate_reason_locked(
        self,
        now,
        request_received_at,
    ):
        if not self.joint_test_required:
            return None
        if self.joint_test_running:
            return (
                "joint rotation precheck is still running; wait for "
                "'JOINT ROTATION PRECHECK PASSED'"
            )
        if not self.joint_test_passed:
            return (
                "joint rotation precheck has not passed in this node session; "
                "call /joint_rotation_test_enable with data=true and wait for "
                "'JOINT ROTATION PRECHECK PASSED'"
            )
        if self._expire_joint_test_permission_locked(now):
            return self.joint_test_message
        if (
            now - self.joint_test_passed_at
            < self.joint_test_confirmation_delay_sec
        ):
            return (
                "joint rotation precheck just passed; wait %.1f seconds, "
                "review the result, then confirm vibration again"
                % self.joint_test_confirmation_delay_sec
            )
        if (
            request_received_at is None
            or request_received_at <= self.joint_test_passed_at
        ):
            return (
                "a new confirmation is required after the joint rotation "
                "precheck passes"
            )
        return None

    def _reject_start(self, reason):
        with self.state_lock:
            self.last_start_message = reason
        self.get_logger().error(reason)
        return False

    def _stop_test(self, reason):
        with self.state_lock:
            if not self.test_enabled:
                return False
            self.test_enabled = False
            self._begin_smooth_return_locked("vibration")
            self._close_csv()
        self._queue_diagnostic_log(
            "info",
            "vibration excitation stopped (%s); returning to center over "
            "%.3f seconds" % (reason, self.stop_ramp_sec),
        )
        return True

    def _begin_smooth_return_locked(self, owner, duration_sec=None):
        self.return_start_positions[:] = self.last_command_positions
        self.return_started_at = time.monotonic()
        self.return_duration_sec = (
            self.stop_ramp_sec
            if duration_sec is None
            else float(duration_sec)
        )
        if not math.isfinite(self.return_duration_sec) or self.return_duration_sec <= 0.0:
            raise ValueError("smooth return duration must be finite and > 0")
        self.returning_to_center = True
        self.return_owner = owner

    def _return_to_center_command(self, now):
        with self.state_lock:
            progress = (now - self.return_started_at) / self.return_duration_sec
            progress = min(max(progress, 0.0), 1.0)
            smooth = (
                10.0 * progress**3
                - 15.0 * progress**4
                + 6.0 * progress**5
            )
            command = (
                self.return_start_positions
                + smooth
                * (self.center_positions - self.return_start_positions)
            )
            if progress >= 1.0:
                self.returning_to_center = False
                self.return_owner = ""
                self._queue_diagnostic_log(
                    "info", "smooth return complete; holding center position"
                )
            return command

    def _joint_feedback_ready(self, now):
        with self.feedback_lock:
            last_seen = self.joint_state_last_seen_at.copy()
        return bool(
            np.all(last_seen > 0.0)
            and np.all(now - last_seen <= self.joint_state_timeout_sec)
        )

    def _log_waiting_for_feedback(self, now):
        if now - self.last_feedback_wait_log_at < 1.0:
            return
        with self.feedback_lock:
            unseen = self.joint_state_last_seen_at <= 0.0
            stale = np.logical_and(
                np.logical_not(unseen),
                now - self.joint_state_last_seen_at
                > self.joint_state_timeout_sec,
            )
        if np.any(unseen):
            detail = "missing: " + ", ".join(
                name for index, name in enumerate(JOINT_NAMES) if unseen[index]
            )
        else:
            detail = "stale: " + ", ".join(
                name for index, name in enumerate(JOINT_NAMES) if stale[index]
            )
        self.get_logger().warning(
            "waiting for complete fresh joint feedback; %s" % detail
        )
        self.last_feedback_wait_log_at = now

    def _latch_safety_fault(self, reason):
        with self.state_lock:
            if self.safety_fault:
                return
            self.safety_fault = True
            self.test_enabled = False
            self.joint_test_running = False
            self.joint_test_passed = False
            self.joint_test_passed_at = 0.0
            if not self.joint_test_failed:
                self.joint_test_failed = True
                self.joint_test_failure_reason = reason
                self.joint_test_message = (
                    "joint rotation precheck invalidated by safety fault: %s"
                    % reason
                )
            self.returning_to_center = False
            self.return_owner = ""
            self.last_start_message = "latched safety fault; restart required"
            self._close_csv()
            shutdown_context = self.shutdown_on_safety_fault
            if shutdown_context:
                # Hardware shutdown takes priority over preserving the tail of
                # the CSV log. Do not let a slow filesystem delay process exit
                # and the launch-level shutdown of the hardware driver.
                self.csv_force_shutdown.set()
                self.csv_wakeup.set()
        self.get_logger().fatal(
            "SAFETY FAULT: %s; command publishing stopped, restart required" % reason
        )
        if shutdown_context:
            self.get_logger().fatal(
                "shutting down the vibration controller so launch can stop "
                "the hardware driver"
            )
            rclpy.try_shutdown(context=self.context)

    def _open_csv(self):
        with self.state_lock:
            self._close_csv()
            if not self.log_csv_path:
                return
            self.csv_session_counter += 1
            self.csv_active_session = self.csv_session_counter
            self.csv_last_enqueued_sequence = 0
            self.csv_next_log_elapsed = 0.0
            self.csv_dropped_since_warning = 0
            header = ["elapsed_sec", "frequency_hz"]
            for name in self.active_joint_names:
                header.extend(
                    [
                        name + "_command_position_rad",
                        name + "_measured_position_rad",
                    ]
                )
            self.csv_control_queue.put(
                (
                    "open",
                    self.csv_active_session,
                    self.log_csv_path,
                    header,
                )
            )
            self.csv_wakeup.set()

    def _write_csv(self, elapsed, frequency, command_positions, now):
        with self.state_lock:
            if self.csv_active_session == 0:
                return
            if elapsed + 1.0e-12 < self.csv_next_log_elapsed:
                return
            self.csv_next_log_elapsed = elapsed + 1.0 / self.log_rate_hz

            with self.feedback_lock:
                if self.has_joint_state:
                    measured = self.measured_positions[
                        self.active_joint_indices
                    ].copy()
                else:
                    measured = np.full(
                        len(self.active_joint_indices),
                        float("nan"),
                    )
            commanded = np.asarray(command_positions, dtype=np.float64)[
                self.active_joint_indices
            ].copy()
            sequence = self.csv_last_enqueued_sequence + 1
            try:
                self.csv_row_queue.put_nowait(
                    (
                        self.csv_active_session,
                        sequence,
                        float(elapsed),
                        float(frequency),
                        commanded,
                        measured,
                    )
                )
                self.csv_last_enqueued_sequence = sequence
                self.csv_wakeup.set()
            except Full:
                self.csv_dropped_since_warning += 1

    def _close_csv(self):
        with self.state_lock:
            if self.csv_active_session == 0:
                return
            session = self.csv_active_session
            final_sequence = self.csv_last_enqueued_sequence
            self.csv_active_session = 0
            self.csv_control_queue.put(("close", session, final_sequence))
            self.csv_wakeup.set()

    def _csv_writer_main(self):
        csv_file = None
        csv_writer = None
        current_session = 0
        last_written_sequence = 0
        close_target = None
        deferred_record = None
        last_flush_at = time.monotonic()
        next_drop_report_at = time.monotonic() + 1.0

        def close_stream():
            nonlocal csv_file, csv_writer
            if csv_file is not None:
                try:
                    csv_file.flush()
                    csv_file.close()
                except OSError as exc:
                    try:
                        self.get_logger().error("cannot close CSV log: %s" % exc)
                    except Exception:
                        pass
            csv_file = None
            csv_writer = None

        def close_session():
            nonlocal current_session, last_written_sequence, close_target
            close_stream()
            current_session = 0
            last_written_sequence = 0
            close_target = None

        while not self.csv_force_shutdown.is_set():
            report_now = time.monotonic()
            if report_now >= next_drop_report_at:
                with self.state_lock:
                    dropped = self.csv_dropped_since_warning
                    self.csv_dropped_since_warning = 0
                    self.csv_last_drop_warning_at = report_now
                if dropped:
                    self.get_logger().warning(
                        "CSV queue is full; dropped %d log samples in the "
                        "last interval (actuator publishing is unaffected)"
                        % dropped
                    )
                next_drop_report_at = report_now + 1.0

            if close_target is None:
                try:
                    control = self.csv_control_queue.get_nowait()
                except Empty:
                    control = None

                if control is not None:
                    action = control[0]
                    if action == "open":
                        close_session()
                        _, session, path, header = control
                        current_session = session
                        try:
                            csv_file = open(
                                path,
                                "w",
                                newline="",
                                encoding="utf-8",
                            )
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(header)
                            csv_file.flush()
                            last_flush_at = time.monotonic()
                            self.get_logger().info(
                                "logging vibration data asynchronously to %s"
                                % path
                            )
                        except OSError as exc:
                            close_stream()
                            self.get_logger().error(
                                "cannot open CSV log %s: %s" % (path, exc)
                            )
                        continue
                    if action == "close":
                        _, session, final_sequence = control
                        if session == current_session:
                            close_target = final_sequence
                            if last_written_sequence >= close_target:
                                close_session()
                        continue
                    if action == "shutdown":
                        close_session()
                        return
                    if action == "log":
                        _, level, message = control
                        self._write_diagnostic_log(level, message)
                        continue

            if deferred_record is not None:
                record = deferred_record
                deferred_record = None
            else:
                try:
                    record = self.csv_row_queue.get_nowait()
                except Empty:
                    record = None

            if record is None:
                if (
                    close_target is not None
                    and last_written_sequence >= close_target
                ):
                    close_session()
                    continue
                self.csv_wakeup.clear()
                if (
                    not self.csv_control_queue.empty()
                    or not self.csv_row_queue.empty()
                ):
                    continue
                self.csv_wakeup.wait(timeout=0.05)
                continue

            try:
                (
                    session,
                    sequence,
                    elapsed,
                    frequency,
                    commanded,
                    measured,
                ) = record
                if session > current_session:
                    # A producer always queues the matching "open" control
                    # before its first row. If that control arrives in the
                    # tiny interval between our control and row checks, keep
                    # the row locally and process the control on the next
                    # loop instead of dropping or reordering the sample.
                    deferred_record = record
                    record = None
                    continue
                if session != current_session:
                    continue

                if csv_writer is not None:
                    row = ["%.9f" % elapsed, "%.9f" % frequency]
                    for command_position, measured_position in zip(
                        commanded,
                        measured,
                    ):
                        row.extend(
                            [
                                "%.9f" % command_position,
                                "%.9f" % measured_position,
                            ]
                        )
                    try:
                        csv_writer.writerow(row)
                    except OSError as exc:
                        self.get_logger().error(
                            "cannot write CSV log; remaining samples will be "
                            "discarded: %s" % exc
                        )
                        close_stream()

                last_written_sequence = sequence
                if csv_file is not None and time.monotonic() - last_flush_at >= 1.0:
                    try:
                        csv_file.flush()
                        last_flush_at = time.monotonic()
                    except OSError as exc:
                        self.get_logger().error(
                            "cannot flush CSV log; remaining samples will be "
                            "discarded: %s" % exc
                        )
                        close_stream()
                if (
                    close_target is not None
                    and last_written_sequence >= close_target
                ):
                    close_session()
            finally:
                if record is not None:
                    self.csv_row_queue.task_done()

        close_session()

    def _shutdown_csv_worker(self):
        with self.state_lock:
            if self.csv_worker_stopped:
                return
            self._close_csv()
            self.csv_control_queue.put(("shutdown",))
            self.csv_wakeup.set()
            self.csv_worker_stopped = True

        self.csv_thread.join(timeout=5.0)
        if self.csv_thread.is_alive():
            self.csv_force_shutdown.set()
            self.csv_wakeup.set()
            self.csv_thread.join(timeout=1.0)
        if self.csv_thread.is_alive():
            self.get_logger().warning(
                "CSV writer did not stop before shutdown; it is a daemon thread"
            )

    def _queue_diagnostic_log(self, level, message):
        with self.state_lock:
            if self.csv_worker_stopped:
                return
            self.csv_control_queue.put(("log", level, message))
            self.csv_wakeup.set()

    def _write_diagnostic_log(self, level, message):
        """Log each severity from a stable call site for rclpy Humble."""
        try:
            if level == "warning":
                self.get_logger().warning(message)
            elif level == "error":
                self.get_logger().error(message)
            else:
                self.get_logger().info(message)
        except (RuntimeError, ValueError):
            # A diagnostic message must never terminate the CSV worker. This
            # worker also drains open/close requests used by vibration logs.
            pass

    def _publish_command(self, positions, kp, kd):
        with self.state_lock:
            if self.safety_fault:
                return False
            position_array = np.asarray(positions, dtype=np.float64)
            if not np.all(np.isfinite(position_array)):
                invalid = [
                    name
                    for index, name in enumerate(JOINT_NAMES)
                    if not math.isfinite(float(position_array[index]))
                ]
                self._latch_safety_fault(
                    "refusing to publish non-finite joint commands for: %s"
                    % ", ".join(invalid)
                )
                return False
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = ROBOT_NAME
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = list(JOINT_NAMES)
            msg.pos = position_array.tolist()
            msg.vel = [0.0] * DOF_NUM
            msg.torque = [0.0] * DOF_NUM
            msg.kp = (np.asarray(kp) * self.kp_scale).tolist()
            msg.kd = (np.asarray(kd) * self.kd_scale).tolist()
            if not self._command_publish_gap_is_safe(time.monotonic()):
                return False
            self.actuator_pub.publish(msg)
            published_at = time.monotonic()
            if not self._command_publish_gap_is_safe(published_at):
                return False
            self.last_command_publish_at = published_at
            self.last_command_positions[:] = position_array
            return True

    def _command_publish_gap_is_safe(self, now):
        if (
            not self.hardware_mode
            or self.last_command_publish_at <= 0.0
        ):
            return True
        actual_gap = now - self.last_command_publish_at
        if actual_gap <= self.max_command_gap_sec:
            return True
        self._latch_safety_fault(
            "command publish gap %.6f seconds exceeded the %.6f second "
            "limit (nominal period %.6f seconds)"
            % (
                actual_gap,
                self.max_command_gap_sec,
                1.0 / self.control_rate_hz,
            )
        )
        return False

    def _call_robot_reset(self, reset_step, release, now):
        if self.reset_pending_step != 0 or now < self.reset_retry_after_at:
            return False
        if not rclpy.ok():
            return False
        try:
            service_ready = self.reset_client.service_is_ready()
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warning(
                    "cannot check robot_reset service: %s" % exc
                )
            return False
        if not service_ready:
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
        if (
            self.reset_pending_step != expected_step
            or self.reset_future is None
        ):
            return False
        if not self.reset_future.done():
            if now - self.reset_request_sent_at <= self.reset_response_timeout_sec:
                return False
            self.reset_future.cancel()
            self.reset_future = None
            self.reset_pending_step = 0
            self.reset_retry_after_at = now + 1.0
            self.get_logger().error(
                "robot reset %d response timed out after %.3f seconds; will retry"
                % (expected_step, self.reset_response_timeout_sec)
            )
            return False
        try:
            response = self.reset_future.result()
        except Exception as exc:
            self.reset_future = None
            self.reset_pending_step = 0
            self.reset_retry_after_at = now + 1.0
            self.get_logger().error(
                "robot reset %d failed: %s; will retry"
                % (expected_step, exc)
            )
            return False
        if response is None or not response.is_success:
            self.reset_future = None
            self.reset_pending_step = 0
            self.reset_retry_after_at = now + 1.0
            self.get_logger().error(
                "robot reset %d was rejected by the driver; will retry"
                % expected_step
            )
            return False
        self.reset_future = None
        self.reset_pending_step = 0
        self.reset_retry_after_at = 0.0
        return True

    def destroy_node(self):
        try:
            self.timer.cancel()
            if self.publisher_watchdog_timer is not None:
                self.publisher_watchdog_timer.cancel()
            self._shutdown_csv_worker()
        finally:
            super().destroy_node()


def main(args=None):
    node = None
    executor = None
    rclpy.init(args=args)
    try:
        node = VibrationTestNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.timer.cancel()
                if node.publisher_watchdog_timer is not None:
                    node.publisher_watchdog_timer.cancel()
            except (Exception, KeyboardInterrupt):
                pass
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
