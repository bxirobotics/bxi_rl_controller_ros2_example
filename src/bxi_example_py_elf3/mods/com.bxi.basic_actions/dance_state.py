from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.policies import DanceMotionPolicyGravityIsaaclabV3
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


class DanceState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[DanceMotionPolicyGravityIsaaclabV3],
        *,
        start_frame: int
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self.start_frame = start_frame
        self.playing = True

    @property
    def policy(self) -> DanceMotionPolicyGravityIsaaclabV3:
        return self._policy.get()

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.policy.configure_range(start_frame=self.start_frame)
        ctx.preheat_model(self.policy)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame | None:
        policy = self.policy
        if policy.finished():
            return None
        output = policy.step(
            ctx.inference_frame,
            dt,
            advance=self.playing and advance,
        )
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if self.policy.finished():
            self.policy.configure_range(start_frame=self.start_frame)
            self.policy.reset(ctx.inference_frame)
            ctx.request_state(
                "com.bxi.basic_actions/normal",
                trigger="motion_finished",
                transition={
                    "profile": "dual_running_blend",
                    "duration": 0.5,
                    "sample_from": False,
                },
            )
            return
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
