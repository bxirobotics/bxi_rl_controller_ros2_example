from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.amp import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.utils.mod_system import ResourceHandle
from bxi_example_py_elf3.utils.robot_state_base import RobotControlState
from bxi_example_py_elf3.utils.transition_core import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample


class PdBrakeState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy

    def _frame(self, ctx: BxiExample) -> MotorFrame:
        policy = self._policy.get()
        return self._motor_frame(policy.default_dof_pos, policy.kps, policy.kds)

    def get_entry_frame(self, ctx: BxiExample) -> MotorFrame:
        return self._frame(ctx)

    def sample_running_frame(
        self, ctx: BxiExample, dt: float, *, advance: bool
    ) -> MotorFrame:
        return self._frame(ctx)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        self._apply_frame(ctx, self._frame(ctx))
