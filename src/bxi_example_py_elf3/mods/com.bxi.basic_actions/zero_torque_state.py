from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from bxi_example_py_elf3.mod_api import RobotControlState
from bxi_example_py_elf3.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


class ZeroTorqueState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def _frame(self, ctx: RobotControlContext) -> MotorFrame:
        zeros = np.zeros(ctx.dof_num, dtype=np.float32)
        return self._motor_frame(ctx.joint_nominal_pos, zeros, zeros)

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._frame(ctx)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        return self._frame(ctx)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self._frame(ctx))
