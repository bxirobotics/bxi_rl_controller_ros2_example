from __future__ import annotations

import math
from typing import TYPE_CHECKING
import numpy as np

from bxi_example_py_elf3.inference.amp import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.mod_api import ResourceHandle
from bxi_example_py_elf3.mod_api import RobotControlState
from bxi_example_py_elf3.mod_api import ZERO_TORQUE_STATE
from bxi_example_py_elf3.mod_api import StateBehavior
from bxi_example_py_elf3.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


class HelloState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy
        self.playing = True
        self.shaketime = 0
        self.kp = np.zeros(29, dtype=np.float32)

    @property
    def policy(self) -> HumanoidGaitPolicyLiteIsaaclab:
        return self._policy.get()

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.playing = True
        self.shaketime = 0
        self.kp = np.zeros_like(self.policy.kps)
        ctx.preheat_model(self.policy, with_cmd_vel=True, cmd_vel=self.get_cmd_vel(ctx))

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True
        self.shaketime = 0

    def _wave(self, qpos: np.ndarray) -> np.ndarray:
        qpos[22] = -0.9
        qpos[24] = math.sin(self.shaketime / 10) * 0.5
        qpos[25] = -0.3
        return qpos

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        qpos = self.policy.target_dof_pos.copy()
        qpos[22], qpos[24], qpos[25] = -0.9, 0.0, -0.3
        return self._motor_frame(qpos, self.policy.kps, self.policy.kds)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        if self.shaketime < 50:
            self.kp = self.shaketime / 50 * self.policy.kps
        qpos, _ = self.policy.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            self.get_cmd_vel(ctx),
        )
        self._wave(qpos)
        if self.playing and advance:
            self.shaketime += 1
        return self._motor_frame(qpos, self.kp, self.policy.kds)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(ZERO_TORQUE_STATE, trigger="safety")
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
