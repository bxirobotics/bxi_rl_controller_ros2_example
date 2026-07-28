"""Explicit composition of policy targets into a complete control target."""

from __future__ import annotations

import numpy as np

from .layout import JointLayout
from .mapping import CompiledJointMap
from .target import JointTargetBuffer, JointTargetView


class ExactJointTargetAssembler:
    """Reorder a target that contains exactly the runtime control joints."""

    def __init__(self, control_layout: JointLayout) -> None:
        self.control_layout = control_layout
        self._buffer = JointTargetBuffer(control_layout)
        self._source_layout: JointLayout | None = None
        self._mapping: CompiledJointMap | None = None

    @property
    def target(self) -> JointTargetView:
        return self._buffer.view

    def compose(self, target: JointTargetView) -> JointTargetView:
        if self._source_layout != target.layout:
            self._mapping = CompiledJointMap.compile(
                target.layout,
                self.control_layout,
                require_exact=True,
            )
            self._source_layout = target.layout
        assert self._mapping is not None
        self._mapping.map_into(target.position, self._buffer.position)
        self._mapping.map_into(target.kp, self._buffer.kp)
        self._mapping.map_into(target.kd, self._buffer.kd)
        return self._buffer.view


class PartialJointTargetAssembler:
    """Overlay an explicit subset on an explicit complete fallback target.

    This is the only built-in path from a partial policy action to a complete
    runtime command. Missing joints are never silently filled with zero.
    """

    def __init__(self, control_layout: JointLayout) -> None:
        self.control_layout = control_layout
        self._buffer = JointTargetBuffer(control_layout)
        self._fallback_layout: JointLayout | None = None
        self._fallback_map: CompiledJointMap | None = None
        self._partial_layout: JointLayout | None = None
        self._partial_indices: np.ndarray | None = None

    @property
    def target(self) -> JointTargetView:
        return self._buffer.view

    def compose(
        self,
        partial: JointTargetView,
        *,
        fallback: JointTargetView,
    ) -> JointTargetView:
        if self._fallback_layout != fallback.layout:
            self._fallback_map = CompiledJointMap.compile(
                fallback.layout,
                self.control_layout,
                require_exact=True,
            )
            self._fallback_layout = fallback.layout
        if self._partial_layout != partial.layout:
            missing = partial.layout.missing_from(self.control_layout)
            if missing:
                raise ValueError(
                    f"partial target contains joints outside control layout: {missing}"
                )
            self._partial_indices = np.fromiter(
                (self.control_layout.index(name) for name in partial.layout.names),
                dtype=np.intp,
                count=partial.layout.dof_num,
            )
            self._partial_layout = partial.layout

        assert self._fallback_map is not None
        assert self._partial_indices is not None
        for source, output in (
            (fallback.position, self._buffer.position),
            (fallback.kp, self._buffer.kp),
            (fallback.kd, self._buffer.kd),
        ):
            self._fallback_map.map_into(source, output)
        self._buffer.position[self._partial_indices] = partial.position
        self._buffer.kp[self._partial_indices] = partial.kp
        self._buffer.kd[self._partial_indices] = partial.kd
        return self._buffer.view


__all__ = ["ExactJointTargetAssembler", "PartialJointTargetAssembler"]
