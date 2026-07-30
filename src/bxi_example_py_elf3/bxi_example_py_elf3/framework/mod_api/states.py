from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.inference import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.joints import JointLayout

from .resource import ResourceHandle
from .state import RobotControlState, StateBehavior
from .transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
    TransitionSpec,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import RobotControlContext


class ReplayPolicy(Protocol):
    output: PolicyOutput

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        ...

    def reset(self, frame: InferenceFrame) -> None:
        ...

    def finished(self, trim: int = 0) -> bool:
        ...


ReplayPolicyT = TypeVar("ReplayPolicyT", bound=ReplayPolicy)
ParamsT = TypeVar("ParamsT")
PolicyT = TypeVar("PolicyT")


class _FrameState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
    ABC,
):
    def gains(self, ctx: RobotControlContext) -> tuple[object, object]:
        """Return state-owned kp/kd when ``frame()`` omits explicit gains."""
        raise NotImplementedError(
            f"state '{self.name}' must override gains(ctx) or pass kp/kd to frame()"
        )

    def frame(
        self,
        ctx: RobotControlContext,
        qpos: object,
        *,
        layout: JointLayout | None = None,
        kp: object | None = None,
        kd: object | None = None,
    ) -> MotorFrame:
        if kp is None or kd is None:
            default_kp, default_kd = self.gains(ctx)
            kp = default_kp if kp is None else kp
            kd = default_kd if kd is None else kd
        return self._motor_frame(ctx, qpos, kp, kd, layout=layout)


class PoseState(_FrameState, Generic[ParamsT], ABC):
    """A fixed-pose state with transition capabilities supplied by the library."""

    def __init__(
        self,
        name: str,
        state_id: int,
        params: ParamsT | None = None,
    ) -> None:
        super().__init__(name, state_id)
        self.params = params

    @abstractmethod
    def target_position(self, ctx: RobotControlContext) -> object:
        """Return the target joint positions for this update."""

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self.frame(ctx, self.target_position(ctx))

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        return self.get_entry_frame(ctx)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))


class ProceduralState(_FrameState, Generic[ParamsT], ABC):
    """A time-varying state with transition-safe sampling semantics."""

    def __init__(
        self,
        name: str,
        state_id: int,
        params: ParamsT | None = None,
    ) -> None:
        super().__init__(name, state_id)
        self.params = params
        self.elapsed = 0.0

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.elapsed = 0.0

    @abstractmethod
    def compute_frame(self, ctx: RobotControlContext, elapsed: float) -> MotorFrame:
        """Compute output at ``elapsed`` without mutating time."""

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self.compute_frame(ctx, 0.0)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        frame = self.compute_frame(ctx, self.elapsed)
        if advance:
            self.elapsed += dt
        return frame

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))


class PolicyState(_FrameState, Generic[PolicyT], ABC):
    """Shared lifecycle for prepared policy resources and live inference states."""

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[PolicyT],
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy_handle = policy

    @property
    def policy(self) -> PolicyT:
        return self._policy_handle.get()

    def resolve_policy(self, ctx: RobotControlContext) -> PolicyT:
        return self._policy_handle.get()

    def reset_policy(self, ctx: RobotControlContext, policy: PolicyT) -> None:
        """Override when a model has recurrent state or a playback cursor."""

    def preheat_policy(self, ctx: RobotControlContext, policy: PolicyT) -> None:
        """Use the controller's standard preheater for compatible model APIs."""
        preheat = getattr(ctx, "preheat_model", None)
        has_step = callable(getattr(policy, "step", None))
        if callable(preheat) and has_step:
            preheat(policy)

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        policy = self.resolve_policy(ctx)
        self.reset_policy(ctx, policy)
        self.preheat_policy(ctx, policy)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.reset_policy(ctx, self.resolve_policy(ctx))

    @abstractmethod
    def policy_entry_position(
        self, ctx: RobotControlContext, policy: PolicyT
    ) -> object:
        """Return the stable position used by entry transitions."""

    @abstractmethod
    def infer_position(
        self,
        ctx: RobotControlContext,
        policy: PolicyT,
        dt: float,
        *,
        advance: bool,
    ) -> object:
        """Run or sample inference; advance=False must not mutate policy time."""

    def policy_gains(
        self,
        ctx: RobotControlContext,
        policy: PolicyT,
    ) -> tuple[object, object]:
        kp = getattr(policy, "kps", None)
        kd = getattr(policy, "kds", None)
        return self.gains(ctx) if kp is None or kd is None else (kp, kd)

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        policy = self.resolve_policy(ctx)
        kp, kd = self.policy_gains(ctx, policy)
        return self.frame(
            ctx,
            self.policy_entry_position(ctx, policy),
            layout=policy.output.joints.layout,
            kp=kp,
            kd=kd,
        )

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        policy = self.resolve_policy(ctx)
        kp, kd = self.policy_gains(ctx, policy)
        return self.frame(
            ctx,
            self.infer_position(ctx, policy, dt, advance=advance),
            layout=policy.output.joints.layout,
            kp=kp,
            kd=kd,
        )

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))


class MotionReplayState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
    Generic[ReplayPolicyT],
):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[ReplayPolicyT],
        *,
        finish_state: str,
        finish_trigger: str,
        end_frame_trim: int = 0,
        end_transition: TransitionSpec = None,
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy_handle = policy
        self.finish_state = finish_state
        self.finish_trigger = finish_trigger
        self.end_frame_trim = end_frame_trim
        self.end_transition = end_transition
        self.playing = True

    @property
    def policy(self) -> ReplayPolicyT:
        return self._policy_handle.get()

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        policy = self.policy
        ctx.preheat_model(policy)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        policy = self.policy
        return self._motor_frame_from_target(ctx, policy.output.joints)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        policy = self.policy
        output = policy.step(
            ctx.inference_frame,
            dt,
            advance=self.playing and advance,
        )
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        policy = self.policy
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        if policy.finished(self.end_frame_trim):
            ctx.request_state(
                self.finish_state,
                trigger=self.finish_trigger,
                transition=self.end_transition,
            )

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True


__all__ = [
    "MotionReplayState",
    "PolicyState",
    "PoseState",
    "ProceduralState",
    "ReplayPolicy",
]
