from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class ZeroTorqueState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def _frame(self, ctx: RobotControlContext) -> MotorFrame:
        frame = self._motor_frame_buffer
        if frame is None or frame.layout != ctx.robot_layout:
            frame = MotorFrame.empty(ctx.robot_layout)
            frame.kp.fill(0.0)
            frame.kd.fill(0.0)
            self._motor_frame_buffer = frame
        np.copyto(frame.qpos, ctx.robot_joints.position, casting="same_kind")
        return frame

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._frame(ctx)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        return self._frame(ctx)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self._frame(ctx))
