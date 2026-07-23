from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from bxi_example_py_elf3.inference.normal import NormalMotionPolicyMjlab
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


class NormalRunState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[NormalMotionPolicyMjlab]
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy

    @property
    def policy(self) -> NormalMotionPolicyMjlab:
        return self._policy.get()

    def on_prepare(
        self, ctx: BxiExample, from_state: StateBehavior[BxiExample]
    ) -> None:
        ctx.preheat_model(self.policy, with_cmd_vel=True, cmd_vel=self.get_cmd_vel(ctx))

    def on_enter(self, ctx: BxiExample) -> None:
        self.policy.action = np.zeros_like(self.policy.action)

    def get_entry_frame(self, ctx: BxiExample) -> MotorFrame:
        qpos = self.policy.default_joint_pos.copy() + self.policy.target_q
        return self._motor_frame(
            qpos, self.policy.joint_stiffness, self.policy.joint_damping
        )

    def sample_running_frame(
        self, ctx: BxiExample, dt: float, *, advance: bool
    ) -> MotorFrame:
        qpos = self.policy.infer_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_xyzw,
            ctx.current_omega,
            self.get_cmd_vel(ctx),
        )
        return self._motor_frame(
            qpos, self.policy.joint_stiffness, self.policy.joint_damping
        )

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(ZERO_TORQUE_STATE, trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
