from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bxi_example_py_elf3.utils.transition_core import (
    ConfigReader,
    MotorFrame,
    SingleClassTransition,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
    from bxi_example_py_elf3.utils.state_machine import StateBehavior


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
        ctx: "BxiExample",
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> None:
        self._frame = MotorFrame.create(ctx.pos_last, ctx.kp_last, ctx.kd_last)

    def apply(self, ctx: "BxiExample", dt: float, progress: float) -> None:
        frame = self._frame
        if frame is None:
            raise RuntimeError("hold transition has not started")
        ctx.set_motor_target(frame.qpos, frame.kp, frame.kd)
