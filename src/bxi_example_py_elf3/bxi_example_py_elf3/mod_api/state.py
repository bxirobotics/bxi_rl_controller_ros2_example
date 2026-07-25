from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .context import RobotControlContext
from .frame import MotorFrame


CtxT = TypeVar("CtxT")


class StateBehavior(Generic[CtxT]):
    """Transition-agnostic lifecycle shared by all user-defined states."""

    def __init__(self, name: str, state_id: int):
        self.name = name
        self.state_id = state_id
        self.manifest: dict[str, object] = {
            "label": "Unknown",
            "index": None,
            "group": "Base",
            "icon": "warning",
            "confirm": False,
            "confirm_message": "",
        }

    def on_bind(self, ctx: CtxT) -> None:
        pass

    def on_unbind(self, ctx: CtxT) -> None:
        """Release subscriptions, timers, or other state-owned handles."""
        pass

    def on_prepare(self, ctx: CtxT, from_state: "StateBehavior[CtxT]") -> None:
        pass

    def on_prepare_cancel(
        self,
        ctx: CtxT,
        from_state: "StateBehavior[CtxT]",
    ) -> None:
        """Release resources when a prepared transition is cancelled."""
        pass

    def on_enter(self, ctx: CtxT) -> None:
        pass

    def on_update(self, ctx: CtxT, dt: float) -> None:
        pass

    def on_exit(self, ctx: CtxT) -> None:
        pass

    def on_action(self, ctx: CtxT, action_name: str) -> bool:
        return False


class RobotControlState(StateBehavior[RobotControlContext], ABC):
    """Main extensibility point for states that control the robot."""

    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.speed_profile_name: str | None = None
        self._missing_speed_profile_warned = False
        self._cmd_vel_buffer = np.zeros(3, dtype=np.float32)

    def on_bind(self, ctx: RobotControlContext) -> None:
        """Called once after construction and before the state machine starts."""

    @abstractmethod
    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        """Produce normal runtime behavior and motor output."""

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        """Prepare resources before a transition; do not emit motor output."""

    def on_prepare_cancel(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        """Undo preparation if the transition is interrupted."""

    def on_exit(self, ctx: RobotControlContext) -> None:
        ctx.pos_last_state = ctx.qpos.copy()  # type: ignore[attr-defined]
        ctx.kp_last_state = ctx.kp_last.copy()
        ctx.kd_last_state = ctx.kd_last.copy()

    def get_cmd_vel(self, ctx: RobotControlContext) -> NDArray[np.float32]:
        cmd_vel = self._profile_cmd_vel(ctx)
        processed_cmd_vel = self.process_cmd_vel(ctx, cmd_vel)
        if processed_cmd_vel is None:
            processed_cmd_vel = cmd_vel
        return self._publish_cmd_vel(ctx, processed_cmd_vel)

    def process_cmd_vel(
        self,
        ctx: RobotControlContext,
        cmd_vel: NDArray[np.float32],
    ) -> NDArray[np.float32] | None:
        """Override to filter or otherwise transform the configured command."""
        return cmd_vel

    def _profile_cmd_vel(self, ctx: RobotControlContext) -> NDArray[np.float32]:
        self._cmd_vel_buffer.fill(0.0)
        raw_cmd_vel = getattr(ctx, "current_raw_cmd_vel", None)
        if raw_cmd_vel is None or not self.speed_profile_name:
            return self._cmd_vel_buffer

        profiles = getattr(ctx, "speed_profiles", {})
        profile = profiles.get(self.speed_profile_name)
        if not isinstance(profile, Mapping):
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
        ctx: RobotControlContext,
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
    def _apply_frame(ctx: RobotControlContext, frame: MotorFrame | None) -> None:
        if frame is not None:
            ctx.set_motor_target(frame.qpos, frame.kp, frame.kd)


__all__ = ["RobotControlState", "StateBehavior"]
