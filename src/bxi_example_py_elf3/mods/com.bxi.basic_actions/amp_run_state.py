from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

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


class AmpRunState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.zeros(3, dtype=np.float32)
        self.cmd_vel_run = np.zeros(3, dtype=np.float32)

    @property
    def policy(self) -> HumanoidGaitPolicyLiteIsaaclab:
        return self._policy.get()

    def on_prepare(
        self, ctx: BxiExample, from_state: StateBehavior[BxiExample]
    ) -> None:
        ctx.preheat_model(self.policy, with_cmd_vel=True, cmd_vel=self.get_cmd_vel(ctx))

    def on_enter(self, ctx: BxiExample) -> None:
        self.max_vel = 0.0
        self.pre_cmd_vel_run.fill(0.0)
        self.cmd_vel_run.fill(0.0)

    def get_entry_frame(self, ctx: BxiExample) -> MotorFrame:
        return self._motor_frame(
            self.policy.target_dof_pos, self.policy.kps, self.policy.kds
        )

    def process_cmd_vel(
        self, ctx: BxiExample, cmd_vel: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        self.cmd_vel_run[:2] = 0.98 * self.pre_cmd_vel_run[:2] + 0.02 * cmd_vel[:2]
        self.cmd_vel_run[2] = cmd_vel[2]
        self.pre_cmd_vel_run[:] = self.cmd_vel_run
        return self.cmd_vel_run

    def sample_running_frame(
        self, ctx: BxiExample, dt: float, *, advance: bool
    ) -> MotorFrame:
        qpos, velocity = self.policy.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            self.get_cmd_vel(ctx),
        )
        self.max_vel = max(self.max_vel, float(velocity[0]))
        if ctx.loop_count >= 100 + int(0.3 / ctx.dt):
            print(self.max_vel)
            ctx.loop_count = int(0.3 / ctx.dt)
            self.max_vel = 0.0
        return self._motor_frame(qpos, self.policy.kps, self.policy.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(ZERO_TORQUE_STATE, trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
