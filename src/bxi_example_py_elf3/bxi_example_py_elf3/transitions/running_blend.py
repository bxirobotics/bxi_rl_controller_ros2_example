from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from bxi_example_py_elf3.utils.transition_core import (
    ConfigReader,
    MotorFrame,
    RunningFrameProvider,
    SingleClassTransition,
    require_entry_frame_provider,
    require_running_frame_provider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
    from bxi_example_py_elf3.utils.state_machine import StateBehavior


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
        advance_from = reader.boolean("advance_from", default=False)
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
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> None:
        if self._sample_from:
            require_running_frame_provider(from_state)
        if self._sample_to:
            require_running_frame_provider(to_state)
        require_entry_frame_provider(to_state)

    def on_start(
        self,
        ctx: "BxiExample",
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> None:
        self._from_provider = (
            require_running_frame_provider(from_state) if self._sample_from else None
        )
        self._to_provider = (
            require_running_frame_provider(to_state) if self._sample_to else None
        )
        self._to_entry = require_entry_frame_provider(to_state).get_entry_frame(ctx)
        self._last_frame = MotorFrame.create(
            ctx.pos_last,
            ctx.kp_last,
            ctx.kd_last,
        )

    def apply(self, ctx: "BxiExample", dt: float, progress: float) -> None:
        to_entry = self._to_entry
        last_frame = self._last_frame
        if to_entry is None or last_frame is None:
            raise RuntimeError("running blend transition has not started")

        from_frame = last_frame
        if self._from_provider is not None:
            from_frame = (
                self._from_provider.sample_running_frame(
                    ctx,
                    dt,
                    advance=self._advance_from,
                )
                or last_frame
            )

        to_frame = to_entry
        if self._to_provider is not None:
            to_frame = (
                self._to_provider.sample_running_frame(
                    ctx,
                    dt,
                    advance=self._advance_to,
                )
                or to_entry
            )

        alpha = self._curve_alpha(progress)
        ctx.set_motor_target(
            from_frame.qpos + (to_frame.qpos - from_frame.qpos) * alpha,
            from_frame.kp + (to_frame.kp - from_frame.kp) * alpha,
            from_frame.kd + (to_frame.kd - from_frame.kd) * alpha,
        )

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
