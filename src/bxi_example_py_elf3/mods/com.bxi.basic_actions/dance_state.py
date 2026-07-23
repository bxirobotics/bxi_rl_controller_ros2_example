from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.beyondmimic import DanceMotionPolicyGravityIsaaclabV3
from bxi_example_py_elf3.utils.mod_system import ResourceHandle
from bxi_example_py_elf3.utils.robot_state_base import RobotControlState
from bxi_example_py_elf3.utils.state_library import NORMAL_STATE, ZERO_TORQUE_STATE
from bxi_example_py_elf3.utils.state_machine import StateBehavior
from bxi_example_py_elf3.utils.transition_core import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample


class DanceState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[DanceMotionPolicyGravityIsaaclabV3],
        *,
        start_frame: int
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy
        self.start_frame = start_frame
        self.playing = True

    @property
    def policy(self) -> DanceMotionPolicyGravityIsaaclabV3:
        return self._policy.get()

    def on_prepare(
        self, ctx: BxiExample, from_state: StateBehavior[BxiExample]
    ) -> None:
        self.policy.timestep = self.start_frame
        self.policy.timeinit = 0.0
        ctx.preheat_model(self.policy)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        self.policy.timestep = self.start_frame

    def get_entry_frame(self, ctx: BxiExample) -> MotorFrame:
        return self._motor_frame(
            self.policy.target_dof_pos, self.policy.kps, self.policy.kds
        )

    def sample_running_frame(
        self, ctx: BxiExample, dt: float, *, advance: bool
    ) -> MotorFrame | None:
        policy = self.policy
        if policy.timestep >= policy.motionpos.shape[0]:
            return None
        qpos = policy.inference_step(
            ctx.current_q, ctx.current_dq, ctx.current_quat_wxyz, ctx.current_omega
        )
        if self.playing and advance:
            policy.timestep += 50 * dt
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if self.policy.timestep >= self.policy.motionpos.shape[0]:
            self.policy.timestep = self.start_frame
            ctx.request_state(
                NORMAL_STATE,
                trigger="motion_finished",
                transition={
                    "profile": "dual_running_blend",
                    "duration": 0.5,
                    "sample_from": False,
                },
            )
            return
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(ZERO_TORQUE_STATE, trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
