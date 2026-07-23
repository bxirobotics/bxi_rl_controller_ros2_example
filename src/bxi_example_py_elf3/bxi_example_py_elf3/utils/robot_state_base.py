from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.utils.state_machine import StateBehavior
from bxi_example_py_elf3.utils.transition_core import MotorFrame

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
else:
    BxiExample = Any


class RobotControlState(StateBehavior[BxiExample], ABC):
    """Base for robot states; transition-specific capabilities live in Protocols."""

    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.speed_profile_name: str | None = None
        self._missing_speed_profile_warned = False
        self._cmd_vel_buffer = np.zeros(3, dtype=np.float32)

    def on_bind(self, ctx: BxiExample) -> None:
        """Called once after construction and before the state machine starts."""

    @abstractmethod
    def on_update(self, ctx: BxiExample, dt: float) -> None:
        """Produce the state's normal runtime behavior and motor output."""

    def on_prepare(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
    ) -> None:
        """Prepare resources before a transition starts; do not emit motor output."""

    def on_prepare_cancel(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
    ) -> None:
        """Called if a transition is interrupted after this state was prepared."""

    def on_exit(self, ctx: BxiExample) -> None:
        ctx.pos_last_state = ctx.qpos.copy()
        ctx.kp_last_state = ctx.kp_last.copy()
        ctx.kd_last_state = ctx.kd_last.copy()

    def get_cmd_vel(self, ctx: BxiExample) -> NDArray[np.float32]:
        cmd_vel = self._profile_cmd_vel(ctx)
        processed_cmd_vel = self.process_cmd_vel(ctx, cmd_vel)
        if processed_cmd_vel is None:
            processed_cmd_vel = cmd_vel
        return self._publish_cmd_vel(ctx, processed_cmd_vel)

    def process_cmd_vel(
        self,
        ctx: BxiExample,
        cmd_vel: NDArray[np.float32],
    ) -> NDArray[np.float32] | None:
        """Override to filter or otherwise transform the configured command."""
        return cmd_vel

    def _profile_cmd_vel(self, ctx: BxiExample) -> NDArray[np.float32]:
        self._cmd_vel_buffer.fill(0.0)
        raw_cmd_vel = getattr(ctx, "current_raw_cmd_vel", None)
        if raw_cmd_vel is None or not self.speed_profile_name:
            return self._cmd_vel_buffer

        profile = getattr(ctx, "speed_profiles", {}).get(self.speed_profile_name)
        if profile is None:
            if not self._missing_speed_profile_warned:
                logger = getattr(ctx, "get_logger", None)
                message = (
                    f"state '{self.name}' references unknown speed_profile "
                    f"'{self.speed_profile_name}'"
                )
                if callable(logger):
                    logger().warning(message)
                else:
                    print(message)
                self._missing_speed_profile_warned = True
            return self._cmd_vel_buffer

        self._cmd_vel_buffer[0] = np.clip(
            raw_cmd_vel[0] * float(profile.get("vx_scale", 1.0)),
            float(profile.get("vx_min", -np.inf)),
            float(profile.get("vx_max", np.inf)),
        )
        self._cmd_vel_buffer[1] = np.clip(
            raw_cmd_vel[1] * float(profile.get("vy_scale", 1.0)),
            float(profile.get("vy_min", -np.inf)),
            float(profile.get("vy_max", np.inf)),
        )
        self._cmd_vel_buffer[2] = np.clip(
            raw_cmd_vel[2] * float(profile.get("yaw_scale", 1.0)),
            float(profile.get("yaw_min", -np.inf)),
            float(profile.get("yaw_max", np.inf)),
        )
        return self._cmd_vel_buffer

    def _publish_cmd_vel(
        self,
        ctx: BxiExample,
        cmd_vel: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        self._cmd_vel_buffer[:] = np.asarray(cmd_vel, dtype=np.float32).reshape(3)
        current_cmd_vel = getattr(ctx, "current_cmd_vel", None)
        if current_cmd_vel is not None:
            current_cmd_vel[:] = self._cmd_vel_buffer
        return self._cmd_vel_buffer

    @staticmethod
    def _motor_frame(qpos: object, kp: object, kd: object) -> MotorFrame:
        return MotorFrame.create(qpos, kp, kd)

    @staticmethod
    def _apply_frame(ctx: BxiExample, frame: MotorFrame | None) -> None:
        if frame is not None:
            ctx.set_motor_target(frame.qpos, frame.kp, frame.kd)


__all__ = ["RobotControlState"]
