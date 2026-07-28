from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.beyondmimic import DanceMotionPolicyGravityIsaaclabV3
from bxi_example_py_elf3.mod_api import ResourceHandle
from bxi_example_py_elf3.mod_api import RobotControlState
from bxi_example_py_elf3.mod_api import NORMAL_STATE, ZERO_TORQUE_STATE
from bxi_example_py_elf3.mod_api import StateBehavior
from bxi_example_py_elf3.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


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
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.policy.reset(self.start_frame)
        ctx.preheat_model(self.policy)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame(self.policy.target, self.policy.kp, self.policy.kd)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame | None:
        policy = self.policy
        if policy.finished():
            return None
        qpos = policy.step(
            ctx.current_q, ctx.current_dq, ctx.current_quat_wxyz, ctx.current_omega
        )
        if self.playing and advance:
            policy.advance(dt)
        return self._motor_frame(qpos, policy.kp, policy.kd)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if self.policy.finished():
            self.policy.reset(self.start_frame)
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

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
