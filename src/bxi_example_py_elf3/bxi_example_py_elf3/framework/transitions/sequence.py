from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bxi_example_py_elf3.framework.runtime.transition import compile_transition
from bxi_example_py_elf3.framework.mod_api.transition import (
    ConfigReader,
    SingleClassTransition,
    TransitionPlan,
    TransitionSession,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext
    from bxi_example_py_elf3.framework.mod_api import StateBehavior


class SequenceTransition(SingleClassTransition):
    type_name = "sequence"

    def __init__(self, name: str, plans: tuple[TransitionPlan, ...]):
        super().__init__(name, sum(plan.duration for plan in plans))
        self._plans = plans
        self._from_state: StateBehavior[RobotControlContext] | None = None
        self._to_state: StateBehavior[RobotControlContext] | None = None
        self._index = 0
        self._current: TransitionSession | None = None

    @classmethod
    def from_config(
        cls,
        name: str,
        raw: Mapping[str, object],
    ) -> "SequenceTransition":
        reader = ConfigReader(raw, name)
        raw_steps = reader.mappings("steps")
        reader.finish()
        if not raw_steps:
            raise ValueError(f"{name}.steps must not be empty")
        plans = tuple(
            compile_transition(f"{name}.steps[{index}]", step)
            for index, step in enumerate(raw_steps)
        )
        return cls(name, plans)

    def validate_states(
        self,
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        for plan in self._plans:
            plan.validate_states(from_state, to_state)

    def on_start(
        self,
        ctx: "RobotControlContext",
        from_state: "StateBehavior[RobotControlContext]",
        to_state: "StateBehavior[RobotControlContext]",
    ) -> None:
        self._from_state = from_state
        self._to_state = to_state
        self._index = 0
        self._current = None

    def apply(self, ctx: "RobotControlContext", dt: float, progress: float) -> None:
        from_state = self._from_state
        to_state = self._to_state
        if from_state is None or to_state is None:
            raise RuntimeError("sequence transition has not started")

        remaining = max(float(dt), 0.0)
        while self._index < len(self._plans):
            if self._current is None:
                self._current = self._plans[self._index].create_session(
                    ctx,
                    from_state,
                    to_state,
                )
            session = self._current
            available = max(session.duration - session.elapsed, 0.0)
            step_dt = min(remaining, available)
            if not session.update(ctx, step_dt):
                break
            self._index += 1
            self._current = None
            remaining = max(remaining - step_dt, 0.0)

    def config_snapshot(self) -> dict[str, object]:
        return {"steps": [plan.snapshot() for plan in self._plans]}
