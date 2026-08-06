from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bxi_example_py_elf3.framework.mod_api.transition import (
    ConfigReader,
    MotorFrame,
    SingleClassTransition,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext
    from bxi_example_py_elf3.framework.mod_api import StateBehavior


class HoldTransition(SingleClassTransition):
    type_name = "hold"

    def __init__(self, name: str, duration: float):
        super().__init__(name, duration)
        self._frame: MotorFrame | None = None

    @classmethod
    def from_config(
        cls,
        name: str,
        raw: Mapping[str, object],
    ) -> "HoldTransition":
        reader = ConfigReader(raw, name)
        duration = reader.float("duration", minimum=0.0)
        reader.finish()
        return cls(name, duration)

    def on_start(
        self,
        ctx: "RobotControlContext",
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        self._frame = MotorFrame.create(
            ctx.robot_layout,
            ctx.last_motor_frame.qpos,
            ctx.last_motor_frame.kp,
            ctx.last_motor_frame.kd,
            vel=ctx.last_motor_frame.vel,
            torque=ctx.last_motor_frame.torque,
        )

    def apply(self, ctx: "RobotControlContext", dt: float, progress: float) -> None:
        frame = self._frame
        if frame is None:
            raise RuntimeError("hold transition has not started")
        ctx.set_motor_target(frame)
