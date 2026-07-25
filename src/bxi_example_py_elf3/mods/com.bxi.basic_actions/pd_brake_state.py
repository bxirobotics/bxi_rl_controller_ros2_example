from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.amp import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.mod_api import ResourceHandle
from bxi_example_py_elf3.mod_api import RobotControlState
from bxi_example_py_elf3.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


class PdBrakeState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy

    def _frame(self, ctx: RobotControlContext) -> MotorFrame:
        policy = self._policy.get()
        return self._motor_frame(policy.default_dof_pos, policy.kps, policy.kds)

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._frame(ctx)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        return self._frame(ctx)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self._frame(ctx))
