from __future__ import annotations

from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.utils.mod_system import ResourceHandle
from bxi_example_py_elf3.utils.robot_state_base import RobotControlState
from bxi_example_py_elf3.utils.state_machine import StateBehavior
from bxi_example_py_elf3.utils.transition_core import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
    TransitionSpec,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample


NORMAL_STATE = "com.bxi.normal/normal"
ZERO_TORQUE_STATE = "com.bxi.zero_torque/zero_torque"


class ReplayPolicy(Protocol):
    timestep: float
    start_frame: int
    end_frame: int
    target_dof_pos: NDArray[np.floating]
    kps: NDArray[np.floating]
    kds: NDArray[np.floating]

    def inference_step(
        self,
        q: NDArray[np.floating],
        dq: NDArray[np.floating],
        quat: NDArray[np.floating],
        omega: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        ...


ReplayPolicyT = TypeVar("ReplayPolicyT", bound=ReplayPolicy)


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
        finish_trigger: str,
        end_frame_trim: int = 0,
        end_transition: TransitionSpec = None,
    ) -> None:
        super().__init__(name, state_id)
        self._policy_handle = policy
        self.finish_trigger = finish_trigger
        self.end_frame_trim = end_frame_trim
        self.end_transition = end_transition
        self.playing = True

    @property
    def policy(self) -> ReplayPolicyT:
        return self._policy_handle.get()

    def on_prepare(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
    ) -> None:
        policy = self.policy
        policy.timestep = policy.start_frame
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0
        ctx.preheat_model(policy)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        policy = self.policy
        policy.timestep = policy.start_frame
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0

    def get_entry_frame(self, ctx: BxiExample) -> MotorFrame:
        policy = self.policy
        qpos = getattr(policy, "target_dof_pos", None)
        if qpos is None:
            qpos = getattr(policy, "default_dof_pos", None)
        if qpos is None:
            raise ValueError(f"state '{self.name}' policy has no entry position")
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def sample_running_frame(
        self, ctx: BxiExample, dt: float, *, advance: bool
    ) -> MotorFrame:
        policy = self.policy
        qpos = policy.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )
        if self.playing and advance:
            policy.timestep += 50 * dt
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        policy = self.policy
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        if policy.timestep > policy.end_frame - self.end_frame_trim:
            ctx.request_state(
                NORMAL_STATE,
                trigger=self.finish_trigger,
                transition=self.end_transition,
            )

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True


__all__ = [
    "MotionReplayState",
    "NORMAL_STATE",
    "ReplayPolicy",
    "ZERO_TORQUE_STATE",
]
