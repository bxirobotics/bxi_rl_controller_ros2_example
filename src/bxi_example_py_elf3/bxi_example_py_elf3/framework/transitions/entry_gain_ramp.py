from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

from bxi_example_py_elf3.framework.mod_api.transition import (
    ConfigReader,
    FloatArray,
    MotorFrame,
    SingleClassTransition,
    require_entry_frame_provider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext
    from bxi_example_py_elf3.framework.mod_api import StateBehavior


GainStart = Literal["current", "zero", "target"]


class EntryGainRampTransition(SingleClassTransition):
    type_name = "entry_gain_ramp"

    def __init__(
        self,
        name: str,
        duration: float,
        kp_from: GainStart,
        kd_from: GainStart,
    ):
        super().__init__(name, duration)
        self._kp_from = kp_from
        self._kd_from = kd_from
        self._target: MotorFrame | None = None
        self._kp_start: FloatArray | None = None
        self._kd_start: FloatArray | None = None
        self._frame: MotorFrame | None = None

    @classmethod
    def from_config(
        cls,
        name: str,
        raw: Mapping[str, object],
    ) -> "EntryGainRampTransition":
        reader = ConfigReader(raw, name)
        duration = reader.float("duration", minimum=0.0)
        kp_from = reader.literal(
            "kp_from",
            ("current", "zero", "target"),
            default="current",
        )
        kd_from = reader.literal(
            "kd_from",
            ("current", "zero", "target"),
            default="target",
        )
        reader.finish()
        return cls(name, duration, kp_from, kd_from)

    def validate_states(
        self,
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        require_entry_frame_provider(to_state)

    def on_start(
        self,
        ctx: "RobotControlContext",
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        natural_target = require_entry_frame_provider(to_state).get_entry_frame(ctx)
        target = MotorFrame.empty(ctx.robot_layout)
        ctx.resolve_motor_frame(natural_target, target)
        self._target = target
        last = ctx.last_motor_frame
        self._kp_start = self._gain_start(self._kp_from, target.kp, last.kp)
        self._kd_start = self._gain_start(self._kd_from, target.kd, last.kd)
        self._frame = MotorFrame.empty(ctx.robot_layout)

    def apply(self, ctx: "RobotControlContext", dt: float, progress: float) -> None:
        target = self._target
        kp_start = self._kp_start
        kd_start = self._kd_start
        if target is None or kp_start is None or kd_start is None:
            raise RuntimeError("entry gain ramp transition has not started")
        frame = self._frame
        if frame is None:
            raise RuntimeError("entry gain ramp transition has no output frame")
        np.copyto(frame.qpos, target.qpos)
        np.copyto(frame.vel, target.vel)
        np.copyto(frame.torque, target.torque)
        for start, end, output in (
            (kp_start, target.kp, frame.kp),
            (kd_start, target.kd, frame.kd),
        ):
            np.subtract(end, start, out=output)
            output *= progress
            output += start
        ctx.set_motor_target(frame)

    def config_snapshot(self) -> dict[str, object]:
        return {
            "kp_from": self._kp_from,
            "kd_from": self._kd_from,
        }

    @staticmethod
    def _gain_start(
        mode: GainStart,
        target: FloatArray,
        current: object,
    ) -> FloatArray:
        if mode == "target":
            return target.copy()
        if mode == "zero":
            return np.zeros_like(target)
        current_array = np.asarray(current, dtype=np.float32)
        if current_array.shape != target.shape:
            raise ValueError(
                f"current gain shape {current_array.shape} does not match "
                f"target {target.shape}"
            )
        return current_array.copy()
