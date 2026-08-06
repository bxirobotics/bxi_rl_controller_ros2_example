from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
import communication.msg as bxi_msg
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.inference import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.mod_api import ResourceHandle, RobotControlState
from bxi_example_py_elf3.framework.mod_api import StateBehavior
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

from .gripper import BxiMotor, JointControl

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import LoggerLike, RobotControlContext


class SonicPolicy(Protocol):
    output: PolicyOutput
    last_status: str

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
    """Named-joint SONIC policy state with optional PICO gripper control."""

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
        hardware_gripper: bool = False,
        gripper_enable_interval_s: float = 1.0,
        gripper_left_bus: int = 5,
        gripper_right_bus: int = 6,
        gripper_can_id: int = 1,
        gripper_kp: float = 20.0,
        gripper_kd: float = 1.0,
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
        self.hardware_gripper = bool(hardware_gripper)
        self.gripper_enable_interval_s = float(gripper_enable_interval_s)
        self._left_bus = int(gripper_left_bus)
        self._right_bus = int(gripper_right_bus)
        self._gripper_can_id = int(gripper_can_id)
        self._gripper_kp = float(gripper_kp)
        self._gripper_kd = float(gripper_kd)
        self._validate_config()
        self._last_running_frame: Optional[MotorFrame] = None
        self._policy_logger_bound = False

        self._gripper_session_active = False
        self._gripper_armed = False
        self._last_gripper_enable_time: Optional[float] = None
        self._left_trigger = 0.0
        self._right_trigger = 0.0
        self._gripper_subscriptions = []
        self._gripper_publisher = None
        self._gripper_available = not self.hardware_gripper

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
            self.logger.error("SONIC夹爪不可用：缺少communication.msg.CANFDPacket")
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
        ]
        self._gripper_publisher = ctx.ros_node.create_publisher(
            packet_type,
            "canfd_packet/tx",
            QoSProfile(depth=100),
        )
        self._gripper_available = True

    def on_unbind(self, ctx: RobotControlContext) -> None:
        for subscription in self._gripper_subscriptions:
            ctx.ros_node.destroy_subscription(subscription)
        self._gripper_subscriptions.clear()
        if self._gripper_publisher is not None:
            ctx.ros_node.destroy_publisher(self._gripper_publisher)
            self._gripper_publisher = None

    def _validate_config(self) -> None:
        if min(self._left_bus, self._right_bus, self._gripper_can_id) < 0:
            raise ValueError("gripper bus and CAN IDs must be non-negative")
        finite_values = (
            self.yaw_bias_rad,
            self.live_reference_timeout_s,
            self.source_blend_seconds,
            self.gripper_enable_interval_s,
            self._gripper_kp,
            self._gripper_kd,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("SONIC numeric parameters must be finite")
        if self.live_reference_timeout_s <= 0.0:
            raise ValueError("live_reference_timeout_s must be positive")
        if self.idle_frame_start < 0:
            raise ValueError("idle_frame_start must be non-negative")
        if self.source_blend_seconds < 0.0:
            raise ValueError("source_blend_seconds must be non-negative")

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
        self._last_running_frame = None
        self._start_gripper_session()

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        if not advance:
            return self._last_running_frame or self.get_entry_frame(ctx)
        output = self.policy.step(ctx.inference_frame, dt, advance=True)
        frame = self._motor_frame_from_target(ctx, output.joints)
        self._last_running_frame = frame
        return frame

    def on_enter(self, ctx: RobotControlContext) -> None:
        mode = "SONIC遥操（夹爪）" if self.hardware_gripper else "SONIC遥操"
        self.logger.info(
            f"{mode}已启动；PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE"
        )
        if self.hardware_gripper:
            self._left_trigger = self._right_trigger = 0.0
            now = time.monotonic()
            self._publish_gripper_enable(now)
            self._gripper_armed = True
            self._publish_gripper(self._left_bus, self._left_trigger)
            self._publish_gripper(self._right_bus, self._right_trigger)
            self.logger.info("SONIC夹爪已使能并默认打开")

    def on_exit(self, ctx: RobotControlContext) -> None:
        self._gripper_session_active = False
        self._gripper_armed = False
        self._last_gripper_enable_time = None

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        # if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
        #     ctx.request_state(
        #         "com.bxi.basic_actions/zero_torque",
        #         trigger="sonic_orientation_safety",
        #     )
        #     return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        self._update_gripper()

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

    def _start_gripper_session(self) -> None:
        if not self.hardware_gripper:
            return
        self._left_trigger = self._right_trigger = 0.0
        self._gripper_armed = False
        self._last_gripper_enable_time = None
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

    def _publish_gripper(self, bus: int, trigger: float) -> None:
        command = JointControl(
            p_des=float((1.0 - trigger) * 5.0 - 0.1),
            kp=self._gripper_kp,
            kd=self._gripper_kd,
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

    def _update_gripper(self) -> None:
        if not self.hardware_gripper or not self._gripper_session_active:
            return
        now = time.monotonic()
        if not self._gripper_armed:
            self._publish_gripper_enable(now)
            self._gripper_armed = True
        self._refresh_gripper_enable(now)
        self._publish_gripper(self._left_bus, self._left_trigger)
        self._publish_gripper(self._right_bus, self._right_trigger)


__all__ = ["SonicTeleopState"]
