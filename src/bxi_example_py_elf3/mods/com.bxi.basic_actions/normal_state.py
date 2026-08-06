from __future__ import annotations

import math
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.framework.mod_api import StateBehavior
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class NormalState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    _CMD_VEL_TOPIC = "/cmd_vel"
    _CMD_VEL_MINIMUM = np.asarray((-1.0, -0.5, -1.5), dtype=np.float32)
    _CMD_VEL_MAXIMUM = np.asarray((1.0, 0.5, 1.5), dtype=np.float32)

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self._navigation_control = False
        self._navigation_command = np.zeros(3, dtype=np.float32)
        self._navigation_command_lock = Lock()
        self._cmd_vel_subscription = None
        self._invalid_cmd_vel_warned = False

    def on_bind(self, ctx: RobotControlContext) -> None:
        self._cmd_vel_subscription = ctx.ros_node.create_subscription(
            Twist,
            self._CMD_VEL_TOPIC,
            self._cmd_vel_callback,
            QoSProfile(depth=1),
        )
        self.logger.info(
            f"normal state navigation input: {self._CMD_VEL_TOPIC}"
        )

    def on_unbind(self, ctx: RobotControlContext) -> None:
        subscription = self._cmd_vel_subscription
        self._cmd_vel_subscription = None
        if subscription is not None:
            ctx.ros_node.destroy_subscription(subscription)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self._navigation_control = False

    def _cmd_vel_callback(self, msg: Twist) -> None:
        command = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(value) for value in command):
            if not self._invalid_cmd_vel_warned:
                self.logger.warning(
                    f"ignoring non-finite geometry_msgs/Twist from "
                    f"{self._CMD_VEL_TOPIC}"
                )
                self._invalid_cmd_vel_warned = True
            return

        with self._navigation_command_lock:
            self._navigation_command[:] = command
        self._invalid_cmd_vel_warned = False

    def get_cmd_vel(self, ctx: RobotControlContext) -> np.ndarray:
        if not self._navigation_control:
            return super().get_cmd_vel(ctx)

        with self._navigation_command_lock:
            np.clip(
                self._navigation_command,
                self._CMD_VEL_MINIMUM,
                self._CMD_VEL_MAXIMUM,
                out=self._cmd_vel_buffer,
            )
        return self._publish_cmd_vel(ctx, self._cmd_vel_buffer)

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        ctx.preheat_model(
            self._policy.get(), command=self.get_cmd_vel(ctx)
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        policy = self._policy.get()
        return self._motor_frame_from_target(ctx, policy.output.joints)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        policy = self._policy.get()
        self.get_cmd_vel(ctx)
        output = policy.step(
            ctx.inference_frame,
            dt,
            advance=advance,
        )
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_navigation_control":
            return False
        self._navigation_control = not self._navigation_control
        control_source = (
            f"navigation control: {self._CMD_VEL_TOPIC}"
            if self._navigation_control
            else "manual control"
        )
        self.logger.info(f"normal state switched to {control_source}")
        return True
