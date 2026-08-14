from __future__ import annotations

import math
import time
from threading import Lock
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
import communication.msg as bxi_msg
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.inference import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.mod_api import (
    JointCommandComposer,
    JointCommandLayer,
    JointLayout,
    JointTargetBuffer,
    ResourceHandle,
    RobotControlState,
    StateBehavior,
)
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

from .gripper import (
    BxiMotor,
    CalibrationPhase,
    CalibrationSettings,
    GripperCalibrator,
    JointControl,
    MotorFeedback,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import LoggerLike, RobotControlContext


HEAD_JOINT_NAMES = ("head_y_joint", "head_z_joint")
SONIC_HEAD_JOINTS = JointLayout(HEAD_JOINT_NAMES, label="SONIC PICO head command")
SONIC_OUTPUT_JOINTS = JointLayout(
    (*ELF3_POLICY_JOINTS.names, *SONIC_HEAD_JOINTS.names),
    label="SONIC state output",
)


class SonicPolicy(Protocol):
    output: PolicyOutput
    last_status: str
    head_joint_target: np.ndarray

    def bind_logger(self, logger: LoggerLike) -> None:
        ...

    def reset(self, frame: InferenceFrame | None = None) -> None:
        ...

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        ...

    def configure_runtime(
        self,
        *,
        yaw_bias_rad: float,
        live_ref_timeout_s: float,
        idle_frame_start: int,
        source_blend_duration_s: float,
    ) -> None:
        ...

    def has_fresh_live_reference(self, timeout_s: float | None = None) -> bool:
        ...

    def reset_yaw_alignment(self) -> None:
        ...


class SonicTeleopState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
):
    """Named-joint SONIC policy state with PICO head and optional gripper control."""

    HEAD_KP = 16.747
    HEAD_KD = 1.066

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[SonicPolicy],
        *,
        require_live_reference: bool = False,
        yaw_bias_rad: float = math.pi / 2.0,
        live_reference_timeout_s: float = 0.5,
        idle_frame_start: int = 3509,
        source_blend_seconds: float = 0.4,
        head_control_enabled: bool = True,
        head_pitch_limit_rad: float = 0.5,
        head_yaw_limit_rad: float = 1.0,
        head_pitch_speed_rad_s: float = 1.5,
        head_yaw_speed_rad_s: float = 2.0,
        head_deadband_rad: float = 0.015,
        hardware_gripper: bool = False,
        gripper_enable_interval_s: float = 1.0,
        gripper_left_bus: int = 5,
        gripper_right_bus: int = 6,
        gripper_can_id: int = 1,
        gripper_master_id: int = 0x11,
        gripper_kp: float = 20.0,
        gripper_kd: float = 1.0,
        gripper_calibration_speed_rad_s: float = 0.2,
        gripper_calibration_kp: float = 5.0,
        gripper_calibration_kd: float = 0.5,
        gripper_contact_torque: float = 2.0,
        gripper_abort_torque: float = 8.0,
        gripper_contact_confirm_s: float = 0.25,
        gripper_stopped_velocity_rad_s: float = 0.1,
        gripper_tracking_error_rad: float = 0.08,
        gripper_limit_margin_rad: float = 0.15,
        gripper_minimum_span_rad: float = 1.0,
        gripper_maximum_search_travel_rad: float = 7.0,
        gripper_response_timeout_s: float = 1.0,
        gripper_feedback_timeout_s: float = 0.3,
        gripper_phase_timeout_s: float = 45.0,
        gripper_maximum_mos_temperature_c: int = 80,
        gripper_maximum_motor_temperature_c: int = 80,
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        if gripper_enable_interval_s <= 0.0:
            raise ValueError("gripper_enable_interval_s must be positive")
        self._policy = policy
        self.require_live_reference = bool(require_live_reference)
        self.yaw_bias_rad = float(yaw_bias_rad)
        self.live_reference_timeout_s = float(live_reference_timeout_s)
        self.idle_frame_start = int(idle_frame_start)
        self.source_blend_seconds = float(source_blend_seconds)
        self.head_control_enabled = bool(head_control_enabled)
        self.head_pitch_limit_rad = float(head_pitch_limit_rad)
        self.head_yaw_limit_rad = float(head_yaw_limit_rad)
        self.head_pitch_speed_rad_s = float(head_pitch_speed_rad_s)
        self.head_yaw_speed_rad_s = float(head_yaw_speed_rad_s)
        self.head_deadband_rad = float(head_deadband_rad)
        self.hardware_gripper = bool(hardware_gripper)
        self.gripper_enable_interval_s = float(gripper_enable_interval_s)
        self._left_bus = int(gripper_left_bus)
        self._right_bus = int(gripper_right_bus)
        self._gripper_can_id = int(gripper_can_id)
        self._gripper_master_id = int(gripper_master_id)
        self._gripper_kp = float(gripper_kp)
        self._gripper_kd = float(gripper_kd)
        self._gripper_calibration_kp = float(gripper_calibration_kp)
        self._gripper_calibration_kd = float(gripper_calibration_kd)
        calibration_settings = CalibrationSettings(
            speed_rad_s=float(gripper_calibration_speed_rad_s),
            contact_torque=float(gripper_contact_torque),
            abort_torque=float(gripper_abort_torque),
            contact_confirm_s=float(gripper_contact_confirm_s),
            stopped_velocity_rad_s=float(gripper_stopped_velocity_rad_s),
            tracking_error_rad=float(gripper_tracking_error_rad),
            limit_margin_rad=float(gripper_limit_margin_rad),
            minimum_span_rad=float(gripper_minimum_span_rad),
            maximum_search_travel_rad=float(gripper_maximum_search_travel_rad),
            response_timeout_s=float(gripper_response_timeout_s),
            feedback_timeout_s=float(gripper_feedback_timeout_s),
            phase_timeout_s=float(gripper_phase_timeout_s),
            maximum_mos_temperature_c=int(gripper_maximum_mos_temperature_c),
            maximum_motor_temperature_c=int(gripper_maximum_motor_temperature_c),
        )
        self._gripper_calibrators = {
            self._left_bus: GripperCalibrator("left", calibration_settings),
            self._right_bus: GripperCalibrator("right", calibration_settings),
        }
        self._validate_config()
        self._last_running_frame: Optional[MotorFrame] = None
        self._policy_logger_bound = False
        self._head_command = JointTargetBuffer(SONIC_HEAD_JOINTS)
        self._command_composer: JointCommandComposer | None = None

        self._gripper_session_active = False
        self._gripper_armed = False
        self._gripper_calibrated = False
        self._gripper_faulted = False
        self._last_gripper_enable_time: Optional[float] = None
        self._left_trigger = 0.0
        self._right_trigger = 0.0
        self._gripper_subscriptions = []
        self._gripper_publisher = None
        self._gripper_available = not self.hardware_gripper
        self._gripper_feedback_lock = Lock()
        self._gripper_feedback: dict[int, MotorFeedback] = {}
        self._gripper_phase_snapshot = {
            bus: calibrator.phase
            for bus, calibrator in self._gripper_calibrators.items()
        }
        self._bad_gripper_feedback_warned = False

    @property
    def policy(self) -> SonicPolicy:
        return self._policy.get()

    def on_bind(self, ctx: RobotControlContext) -> None:
        if not self.hardware_gripper:
            return
        packet_type = getattr(
            bxi_msg,
            "CANFDPacket",
            getattr(bxi_msg, "CanfdPacket", None),
        )
        if packet_type is None:
            # The gripper is an optional SONIC peripheral.  Some simulation
            # and deployment message packages do not expose the CAN FD packet
            # type, so degrade to body-only teleoperation instead of making
            # the complete SONIC state unavailable.
            self.hardware_gripper = False
            self._gripper_available = True
            self.logger.warning(
                "SONIC夹爪已禁用：缺少communication.msg.CANFDPacket；"
                "全身遥操仍可用"
            )
            return
        qos = QoSProfile(depth=1)
        self._gripper_subscriptions = [
            ctx.ros_node.create_subscription(
                Float32,
                "pico/left_trigger",
                self._left_trigger_callback,
                qos,
            ),
            ctx.ros_node.create_subscription(
                Float32,
                "pico/right_trigger",
                self._right_trigger_callback,
                qos,
            ),
            ctx.ros_node.create_subscription(
                packet_type,
                "canfd_packet/rx",
                self._gripper_feedback_callback,
                QoSProfile(depth=100),
            ),
        ]
        self._gripper_publisher = ctx.ros_node.create_publisher(
            packet_type,
            "canfd_packet/tx",
            QoSProfile(depth=100),
        )
        self._gripper_available = True

    def on_unbind(self, ctx: RobotControlContext) -> None:
        if self._gripper_session_active and self._gripper_publisher is not None:
            self._disable_grippers()
        for subscription in self._gripper_subscriptions:
            ctx.ros_node.destroy_subscription(subscription)
        self._gripper_subscriptions.clear()
        with self._gripper_feedback_lock:
            self._gripper_feedback.clear()
        if self._gripper_publisher is not None:
            ctx.ros_node.destroy_publisher(self._gripper_publisher)
            self._gripper_publisher = None

    def _validate_config(self) -> None:
        if (
            min(
                self._left_bus,
                self._right_bus,
                self._gripper_can_id,
                self._gripper_master_id,
            )
            < 0
        ):
            raise ValueError("gripper bus and CAN IDs must be non-negative")
        if self._left_bus == self._right_bus:
            raise ValueError("left and right gripper buses must differ")
        if max(self._left_bus, self._right_bus) > 0xFF:
            raise ValueError("gripper bus must fit in uint8")
        if max(self._gripper_can_id, self._gripper_master_id) > 0x7FF:
            raise ValueError("gripper CAN IDs must be standard 11-bit IDs")
        finite_values = (
            self.yaw_bias_rad,
            self.live_reference_timeout_s,
            self.source_blend_seconds,
            self.head_pitch_limit_rad,
            self.head_yaw_limit_rad,
            self.head_pitch_speed_rad_s,
            self.head_yaw_speed_rad_s,
            self.head_deadband_rad,
            self.gripper_enable_interval_s,
            self._gripper_kp,
            self._gripper_kd,
            self._gripper_calibration_kp,
            self._gripper_calibration_kd,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("SONIC numeric parameters must be finite")
        if self.live_reference_timeout_s <= 0.0:
            raise ValueError("live_reference_timeout_s must be positive")
        if self.idle_frame_start < 0:
            raise ValueError("idle_frame_start must be non-negative")
        if self.source_blend_seconds < 0.0:
            raise ValueError("source_blend_seconds must be non-negative")
        if min(
            self.head_pitch_limit_rad,
            self.head_yaw_limit_rad,
            self.head_pitch_speed_rad_s,
            self.head_yaw_speed_rad_s,
        ) <= 0.0:
            raise ValueError("SONIC head limits and speeds must be positive")
        if self.head_deadband_rad < 0.0:
            raise ValueError("SONIC head_deadband_rad must be non-negative")
        if (
            min(
                self._gripper_kp,
                self._gripper_kd,
                self._gripper_calibration_kp,
                self._gripper_calibration_kd,
            )
            < 0.0
        ):
            raise ValueError("gripper gains must be non-negative")

    def is_available(self, ctx: RobotControlContext) -> bool:
        if not self._gripper_available:
            return False
        if self._policy.status != "ready":
            return True
        return not self.require_live_reference or self.policy.has_fresh_live_reference(
            self.live_reference_timeout_s
        )

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        if not self._policy_logger_bound:
            self.policy.bind_logger(self.logger)
            self._policy_logger_bound = True
        self.policy.configure_runtime(
            yaw_bias_rad=self.yaw_bias_rad,
            live_ref_timeout_s=self.live_reference_timeout_s,
            idle_frame_start=self.idle_frame_start,
            source_blend_duration_s=self.source_blend_seconds,
        )
        self.policy.reset(ctx.inference_frame)
        self._prepare_command_source()
        self._last_running_frame = None
        self._gripper_session_active = False

    def _prepare_command_source(self) -> None:
        self._head_command.position.fill(0.0)
        self._head_command.kp.fill(self.HEAD_KP)
        self._head_command.kd.fill(self.HEAD_KD)
        if not self.head_control_enabled:
            self._command_composer = JointCommandComposer(
                ELF3_POLICY_JOINTS,
                (JointCommandLayer("sonic_policy", self.policy.output.joints),),
            )
            return
        self._command_composer = JointCommandComposer(
            SONIC_OUTPUT_JOINTS,
            (
                JointCommandLayer("sonic_policy", self.policy.output.joints),
                JointCommandLayer("pico_head", self._head_command.view),
            ),
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._compose_motor_frame()

    def _update_head_command(self, desired: object, dt: float) -> None:
        if not self.head_control_enabled:
            return
        target = np.asarray(desired, dtype=np.float32)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("SONIC head target must contain two finite joint angles")
        target = np.clip(
            target,
            (-self.head_pitch_limit_rad, -self.head_yaw_limit_rad),
            (self.head_pitch_limit_rad, self.head_yaw_limit_rad),
        )
        target[np.abs(target) < self.head_deadband_rad] = 0.0
        max_step = np.asarray(
            (
                self.head_pitch_speed_rad_s * dt,
                self.head_yaw_speed_rad_s * dt,
            ),
            dtype=np.float32,
        )
        delta = np.clip(
            target - self._head_command.position,
            -max_step,
            max_step,
        )
        self._head_command.position += delta

    def _compose_motor_frame(self) -> MotorFrame:
        if self._command_composer is None:
            raise RuntimeError("SONIC command composer is not prepared")
        return self._command_composer.compose()

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        if not advance:
            return self._last_running_frame or self.get_entry_frame(ctx)
        self.policy.step(ctx.inference_frame, dt, advance=True)
        self._update_head_command(self.policy.head_joint_target, dt)
        frame = self._compose_motor_frame()
        self._last_running_frame = frame
        return frame

    def on_enter(self, ctx: RobotControlContext) -> None:
        mode = "SONIC遥操（夹爪）" if self.hardware_gripper else "SONIC遥操"
        head_status = (
            "头部跟踪已开启"
            if self.head_control_enabled
            else "头部跟踪已关闭"
        )
        self.logger.info(
            f"{mode}已启动；{head_status}；"
            "PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE"
        )
        if self.hardware_gripper:
            self._left_trigger = self._right_trigger = 0.0
            now = time.monotonic()
            self._start_gripper_session(now)
            self._publish_gripper_enable(now)
            self._gripper_armed = True
            self.logger.info("SONIC夹爪已使能，等待左右电机响应后开始低速限位校准；" "校准完成前PICO trigger不会接管夹爪")

    def on_exit(self, ctx: RobotControlContext) -> None:
        if self.hardware_gripper and self._gripper_publisher is not None:
            self._disable_grippers()
        self._gripper_session_active = False
        self._gripper_armed = False
        self._gripper_calibrated = False
        self._gripper_faulted = False
        self._last_gripper_enable_time = None
        with self._gripper_feedback_lock:
            self._gripper_feedback.clear()

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        # if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
        #     ctx.request_state(
        #         "com.bxi.basic_actions/zero_torque",
        #         trigger="sonic_orientation_safety",
        #     )
        #     return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        self._update_gripper(dt)

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "reset_alignment":
            return False
        self.policy.reset_yaw_alignment()
        return True

    @staticmethod
    def _valid_trigger(value: object) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return float(np.clip(result, 0.0, 1.0))

    def _left_trigger_callback(self, msg: Float32) -> None:
        value = self._valid_trigger(msg.data)
        if value is not None and self._gripper_session_active:
            self._left_trigger = value

    def _right_trigger_callback(self, msg: Float32) -> None:
        value = self._valid_trigger(msg.data)
        if value is not None and self._gripper_session_active:
            self._right_trigger = value

    def _gripper_feedback_callback(self, msg) -> None:
        if not self._gripper_session_active:
            return
        bus = int(msg.bus)
        if bus not in self._gripper_calibrators:
            return
        response_id = int(msg.frame.can_id) & 0x7FF
        if response_id != self._gripper_master_id:
            return
        try:
            if int(msg.frame.len) != 8:
                raise ValueError(f"expected 8 feedback bytes, got {int(msg.frame.len)}")
            feedback = BxiMotor.unpack_feedback(
                msg.frame.data,
                received_at=time.monotonic(),
            )
            if feedback.motor_id != self._gripper_can_id:
                return
        except (TypeError, ValueError) as exc:
            if not self._bad_gripper_feedback_warned:
                self.logger.warning(f"忽略非法夹爪响应帧：{exc}")
                self._bad_gripper_feedback_warned = True
            return
        with self._gripper_feedback_lock:
            self._gripper_feedback[bus] = feedback

    def _start_gripper_session(self, now: float) -> None:
        if not self.hardware_gripper:
            return
        self._left_trigger = self._right_trigger = 0.0
        self._gripper_armed = False
        self._gripper_calibrated = False
        self._gripper_faulted = False
        self._bad_gripper_feedback_warned = False
        self._last_gripper_enable_time = None
        with self._gripper_feedback_lock:
            self._gripper_feedback.clear()
        for bus, calibrator in self._gripper_calibrators.items():
            calibrator.reset(now)
            self._gripper_phase_snapshot[bus] = calibrator.phase
        self._gripper_session_active = True

    def _publish_gripper_enable(self, now: float) -> None:
        for bus in (self._left_bus, self._right_bus):
            self._gripper_publisher.publish(
                BxiMotor.build_motor_packet(
                    bus, self._gripper_can_id, BxiMotor.enter_motor_mode()
                )
            )
        self._last_gripper_enable_time = now

    def _refresh_gripper_enable(self, now: float) -> None:
        last_enable_time = self._last_gripper_enable_time
        if (
            last_enable_time is None
            or now - last_enable_time >= self.gripper_enable_interval_s
        ):
            self._publish_gripper_enable(now)

    def _disable_grippers(self) -> None:
        for bus in (self._left_bus, self._right_bus):
            self._gripper_publisher.publish(
                BxiMotor.build_motor_packet(
                    bus,
                    self._gripper_can_id,
                    BxiMotor.exit_motor_mode(),
                )
            )

    def _publish_gripper_target(
        self,
        bus: int,
        target_position: float,
        *,
        kp: float,
        kd: float,
    ) -> None:
        command = JointControl(
            p_des=float(target_position),
            kp=float(kp),
            kd=float(kd),
        )
        data = BxiMotor.pack_cmd(
            command,
            p_range=(-12.5, 12.5),
            v_range=(-45.0, 45.0),
            t_range=(-40.0, 40.0),
            kp_range=(0.0, 500.0),
            kd_range=(0.0, 5.0),
        )
        self._gripper_publisher.publish(
            BxiMotor.build_motor_packet(bus, self._gripper_can_id, data)
        )

    def _publish_gripper(self, bus: int, trigger: float) -> None:
        calibrator = self._gripper_calibrators[bus]
        if not calibrator.ready:
            return
        assert calibrator.open_position is not None
        assert calibrator.closed_position is not None
        target_position = calibrator.closed_position + (1.0 - trigger) * (
            calibrator.open_position - calibrator.closed_position
        )
        self._publish_gripper_target(
            bus,
            target_position,
            kp=self._gripper_kp,
            kd=self._gripper_kd,
        )

    def _log_gripper_phase_change(
        self,
        bus: int,
        calibrator: GripperCalibrator,
    ) -> None:
        previous = self._gripper_phase_snapshot[bus]
        if calibrator.phase is previous:
            return
        self._gripper_phase_snapshot[bus] = calibrator.phase
        side = "左" if bus == self._left_bus else "右"
        labels = {
            CalibrationPhase.SETTLING: "收到响应，正在稳定当前位置",
            CalibrationPhase.SEEKING_OPEN: "开始低速寻找张开限位",
            CalibrationPhase.BACKING_OFF_OPEN: "已检测张开限位，正在回退",
            CalibrationPhase.SEEKING_CLOSED: "开始低速寻找闭合限位",
            CalibrationPhase.BACKING_OFF_CLOSED: "已检测闭合限位，正在回退",
            CalibrationPhase.RETURNING_OPEN: "正在低速返回张开位置",
            CalibrationPhase.READY: "校准完成",
        }
        message = labels.get(calibrator.phase)
        if message is not None:
            self.logger.info(f"SONIC{side}夹爪：{message}")

    def _fail_gripper_session(self, reason: str) -> None:
        if self._gripper_faulted:
            return
        self._gripper_faulted = True
        self._gripper_calibrated = False
        self._disable_grippers()
        self.logger.error(f"SONIC夹爪校准失败：{reason}；左右夹爪已退出电机模式")

    def _update_gripper(self, dt: float) -> None:
        if not self.hardware_gripper or not self._gripper_session_active:
            return
        if self._gripper_faulted:
            return
        now = time.monotonic()
        if not self._gripper_armed:
            self._publish_gripper_enable(now)
            self._gripper_armed = True

        with self._gripper_feedback_lock:
            feedback = dict(self._gripper_feedback)

        waiting_buses = tuple(
            bus
            for bus, calibrator in self._gripper_calibrators.items()
            if calibrator.phase is CalibrationPhase.WAITING_FEEDBACK
        )
        if waiting_buses and not all(bus in feedback for bus in waiting_buses):
            for bus in waiting_buses:
                calibrator = self._gripper_calibrators[bus]
                if bus not in feedback:
                    calibrator.update(None, now, dt)
                if calibrator.failed:
                    self._fail_gripper_session(
                        calibrator.failure_reason or "unknown calibration error"
                    )
                    return
            return

        for bus, calibrator in self._gripper_calibrators.items():
            target = calibrator.update(feedback.get(bus), now, dt)
            self._log_gripper_phase_change(bus, calibrator)
            if calibrator.failed:
                self._fail_gripper_session(
                    calibrator.failure_reason or "unknown calibration error"
                )
                return
            if target is not None and not calibrator.ready:
                self._publish_gripper_target(
                    bus,
                    target,
                    kp=self._gripper_calibration_kp,
                    kd=self._gripper_calibration_kd,
                )

        if not all(
            calibrator.ready for calibrator in self._gripper_calibrators.values()
        ):
            return

        if not self._gripper_calibrated:
            self._gripper_calibrated = True
            details = []
            for bus, calibrator in self._gripper_calibrators.items():
                side = "左" if bus == self._left_bus else "右"
                details.append(
                    f"{side}[闭={calibrator.closed_position:.3f}, "
                    f"开={calibrator.open_position:.3f}]"
                )
            self.logger.info("SONIC夹爪校准完成，PICO trigger开始接管：" + ", ".join(details))

        self._refresh_gripper_enable(now)
        self._publish_gripper(self._left_bus, self._left_trigger)
        self._publish_gripper(self._right_bus, self._right_trigger)


__all__ = [
    "HEAD_JOINT_NAMES",
    "SONIC_HEAD_JOINTS",
    "SONIC_OUTPUT_JOINTS",
    "SonicTeleopState",
]
