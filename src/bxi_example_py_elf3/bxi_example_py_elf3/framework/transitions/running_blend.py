from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

from bxi_example_py_elf3.framework.mod_api.transition import (
    ConfigReader,
    MotorFrame,
    RunningFrameProvider,
    SingleClassTransition,
    require_entry_frame_provider,
    require_running_frame_provider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext
    from bxi_example_py_elf3.framework.mod_api import StateBehavior


BlendCurve = Literal["linear", "smoothstep", "smootherstep"]


class RunningBlendTransition(SingleClassTransition):
    type_name = "running_blend"

    def __init__(
        self,
        name: str,
        duration: float,
        curve: BlendCurve,
        sample_from: bool,
        sample_to: bool,
        advance_from: bool,
        advance_to: bool,
    ):
        super().__init__(name, duration)
        self._curve = curve
        self._sample_from = sample_from
        self._sample_to = sample_to
        self._advance_from = advance_from
        self._advance_to = advance_to
        self._from_provider: RunningFrameProvider | None = None
        self._to_provider: RunningFrameProvider | None = None
        self._to_entry: MotorFrame | None = None
        self._last_frame: MotorFrame | None = None
        self._from_frame: MotorFrame | None = None
        self._to_frame: MotorFrame | None = None
        self._output_frame: MotorFrame | None = None

    @classmethod
    def from_config(
        cls,
        name: str,
        raw: Mapping[str, object],
    ) -> "RunningBlendTransition":
        reader = ConfigReader(raw, name)
        duration = reader.float("duration", minimum=0.0)
        curve = reader.literal(
            "curve",
            ("linear", "smoothstep", "smootherstep"),
            default="smoothstep",
        )
        sample_from = reader.boolean("sample_from", default=True)
        sample_to = reader.boolean("sample_to", default=True)
        advance_from = reader.boolean("advance_from", default=True)
        advance_to = reader.boolean("advance_to", default=False)
        reader.finish()
        return cls(
            name,
            duration,
            curve,
            sample_from,
            sample_to,
            advance_from,
            advance_to,
        )

    def validate_states(
        self,
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        if self._sample_from:
            require_running_frame_provider(from_state)
        if self._sample_to:
            require_running_frame_provider(to_state)
        require_entry_frame_provider(to_state)

    def on_start(
        self,
        ctx: "RobotControlContext",
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        self._from_provider = (
            require_running_frame_provider(from_state) if self._sample_from else None
        )
        self._to_provider = (
            require_running_frame_provider(to_state) if self._sample_to else None
        )
        natural_entry = require_entry_frame_provider(to_state).get_entry_frame(ctx)
        self._to_entry = MotorFrame.empty(ctx.robot_layout)
        ctx.resolve_motor_frame(natural_entry, self._to_entry)
        self._last_frame = MotorFrame.empty(ctx.robot_layout)
        self._last_frame.update(
            ctx.last_motor_frame.qpos,
            ctx.last_motor_frame.kp,
            ctx.last_motor_frame.kd,
            vel=ctx.last_motor_frame.vel,
            torque=ctx.last_motor_frame.torque,
        )
        self._from_frame = MotorFrame.empty(ctx.robot_layout)
        self._to_frame = MotorFrame.empty(ctx.robot_layout)
        self._output_frame = MotorFrame.empty(ctx.robot_layout)

    def apply(self, ctx: "RobotControlContext", dt: float, progress: float) -> None:
        to_entry = self._to_entry
        last_frame = self._last_frame
        if to_entry is None or last_frame is None:
            raise RuntimeError("running blend transition has not started")

        from_frame = last_frame
        if self._from_provider is not None:
            natural_from = (
                self._from_provider.sample_running_frame(
                    ctx,
                    dt,
                    advance=self._advance_from,
                )
                or last_frame
            )
            resolved_from = self._from_frame
            if resolved_from is None:
                raise RuntimeError("running blend transition has no source buffer")
            from_frame = ctx.resolve_motor_frame(natural_from, resolved_from)

        to_frame = to_entry
        if self._to_provider is not None:
            natural_to = (
                self._to_provider.sample_running_frame(
                    ctx,
                    dt,
                    advance=self._advance_to,
                )
                or to_entry
            )
            resolved_to = self._to_frame
            if resolved_to is None:
                raise RuntimeError("running blend transition has no target buffer")
            to_frame = ctx.resolve_motor_frame(natural_to, resolved_to)

        alpha = self._curve_alpha(progress)
        output_frame = self._output_frame
        if output_frame is None:
            raise RuntimeError("running blend transition has no output frame")
        for source, target, output in (
            (from_frame.qpos, to_frame.qpos, output_frame.qpos),
            (from_frame.kp, to_frame.kp, output_frame.kp),
            (from_frame.kd, to_frame.kd, output_frame.kd),
            (from_frame.vel, to_frame.vel, output_frame.vel),
            (from_frame.torque, to_frame.torque, output_frame.torque),
        ):
            np.subtract(target, source, out=output)
            output *= alpha
            output += source
        ctx.set_motor_target(output_frame)

    def config_snapshot(self) -> dict[str, object]:
        return {
            "curve": self._curve,
            "sample_from": self._sample_from,
            "sample_to": self._sample_to,
            "advance_from": self._advance_from,
            "advance_to": self._advance_to,
        }

    def _curve_alpha(self, alpha: float) -> float:
        if self._curve == "linear":
            return alpha
        if self._curve == "smoothstep":
            return alpha * alpha * (3.0 - 2.0 * alpha)
        return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
