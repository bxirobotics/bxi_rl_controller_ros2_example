from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class InitialPosState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    _TARGET_POSITION = np.zeros(29, dtype=np.float32)
    _KP = np.array(
        [
            500, 500, 300,
            300, 100, 100, 300, 50, 50,
            300, 100, 100, 300, 50, 50,
            100, 80, 80, 100, 20, 20, 20,
            100, 80, 80, 100, 20, 20, 20,
        ],
        dtype=np.float32,
    )
    _KD = np.array(
        [
            3, 3, 3,
            2.5, 2, 2, 2.5, 2, 2,
            2.5, 2, 2, 2.5, 2, 2,
            2.5, 2, 2, 2.5, 1, 1, 1,
            2.5, 2, 2, 2.5, 1, 1, 1,
        ],
        dtype=np.float32,
    )

    def _frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame(
            ctx,
            self._TARGET_POSITION,
            self._KP,
            self._KD,
            layout=ELF3_POLICY_JOINTS,
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._frame(ctx)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        return self._frame(ctx)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self._frame(ctx))
