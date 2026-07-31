from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.policies import NormalMotionPolicyMjlab
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


class NormalRunState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[NormalMotionPolicyMjlab]
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy

    @property
    def policy(self) -> NormalMotionPolicyMjlab:
        return self._policy.get()

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        ctx.preheat_model(self.policy, command=self.get_cmd_vel(ctx))

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.policy.reset(ctx.inference_frame)

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        self.get_cmd_vel(ctx)
        output = self.policy.step(
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
