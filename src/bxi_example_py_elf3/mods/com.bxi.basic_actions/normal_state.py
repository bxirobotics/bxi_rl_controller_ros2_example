from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.amp import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.utils.mod_system import ResourceHandle
from bxi_example_py_elf3.utils.robot_state_base import RobotControlState
from bxi_example_py_elf3.utils.state_library import ZERO_TORQUE_STATE
from bxi_example_py_elf3.utils.state_machine import StateBehavior
from bxi_example_py_elf3.utils.transition_core import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample


class NormalState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy

    def on_prepare(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
    ) -> None:
        ctx.preheat_model(
            self._policy.get(), with_cmd_vel=True, cmd_vel=self.get_cmd_vel(ctx)
        )

    def get_entry_frame(self, ctx: BxiExample) -> MotorFrame:
        policy = self._policy.get()
        return self._motor_frame(policy.target_dof_pos, policy.kps, policy.kds)

    def sample_running_frame(
        self, ctx: BxiExample, dt: float, *, advance: bool
    ) -> MotorFrame:
        policy = self._policy.get()
        qpos, _ = policy.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            self.get_cmd_vel(ctx),
        )
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(ZERO_TORQUE_STATE, trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
