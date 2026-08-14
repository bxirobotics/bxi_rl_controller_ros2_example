from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .context import LoggerLike, RobotControlContext
from .frame import MotorFrame
from .resource import ResourceHandle
from bxi_example_py_elf3.framework.joints import JointLayout, JointTargetView


CtxT = TypeVar("CtxT")


class StateBehavior(Generic[CtxT]):
    """Transition-agnostic lifecycle shared by all user-defined states."""

    def __init__(self, name: str, state_id: int):
        self.name = name
        self.state_id = state_id
        # ``None`` inherits the platform-wide control rate.  The runtime sets
        # this from the state-level ``inference_hz`` config field after the
        # factory has returned, so state implementations do not need to parse
        # scheduler configuration themselves.
        self.inference_hz: float | None = None
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

    def is_available(self, ctx: CtxT) -> bool:
        """Return whether this state can be entered right now.

        Implementations must be non-blocking and must not load resources or
        mutate control state.  The state machine calls this immediately before
        preparing a transition; delayed transitions are checked both when the
        request is accepted and again when their delay expires.
        """
        return True

    def on_prepare(self, ctx: CtxT, from_state: "StateBehavior[CtxT]") -> None:
        """Prepare non-blocking state data immediately before a transition."""

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

    def __init__(
        self,
        name: str,
        state_id: int,
        *,
        resources: Sequence[ResourceHandle[Any]] = (),
    ):
        super().__init__(name, state_id)
        self._required_resources = tuple(resources)
        self._logger: LoggerLike | None = None
        self.speed_profile_name: str | None = None
        self._missing_speed_profile_warned = False
        self._cmd_vel_buffer = np.zeros(3, dtype=np.float32)
        self._motor_frame_buffer: MotorFrame | None = None

    @property
    def required_resources(self) -> tuple[ResourceHandle[Any], ...]:
        """Resources that must be READY before a transition can start."""
        return self._required_resources

    @property
    def logger(self) -> LoggerLike:
        logger = self._logger
        if logger is None:
            raise RuntimeError(f"state '{self.name}' logger is not bound")
        return logger

    def _bind_logger(self, logger: LoggerLike) -> None:
        self._logger = logger

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
        """Prepare state data without loading resources or blocking control."""

    def on_prepare_cancel(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        """Undo preparation if the transition is interrupted."""

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
                message = (
                    f"state '{self.name}' references unknown speed_profile "
                    f"'{self.speed_profile_name}'"
                )
                self.logger.warning(message)
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

    def _motor_frame(
        self,
        ctx: RobotControlContext,
        qpos: object,
        kp: object,
        kd: object,
        *,
        layout: JointLayout | None = None,
        vel: object | None = None,
        torque: object | None = None,
    ) -> MotorFrame:
        layout = ctx.robot_layout if layout is None else layout
        frame = self._motor_frame_buffer
        if (
            frame is None
            or (frame.layout is not layout and frame.layout != layout)
        ):
            frame = MotorFrame.empty(layout)
            self._motor_frame_buffer = frame
        return frame.update(qpos, kp, kd, vel=vel, torque=torque)

    def _motor_frame_from_target(
        self,
        ctx: RobotControlContext,
        target: JointTargetView,
    ) -> MotorFrame:
        frame = self._motor_frame_buffer
        if (
            frame is None
            or (frame.layout is not target.layout and frame.layout != target.layout)
        ):
            frame = MotorFrame.empty(target.layout)
            self._motor_frame_buffer = frame
        return frame.update(target.position, target.kp, target.kd)

    @staticmethod
    def _apply_frame(ctx: RobotControlContext, frame: MotorFrame | None) -> None:
        if frame is not None:
            ctx.set_motor_target(frame)


__all__ = ["RobotControlState", "StateBehavior"]
