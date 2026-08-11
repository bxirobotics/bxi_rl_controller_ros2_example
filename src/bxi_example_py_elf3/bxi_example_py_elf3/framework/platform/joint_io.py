"""Reusable named and fixed-order platform joint I/O helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bxi_example_py_elf3.framework.joints import (
    CompiledJointMap,
    JointLayout,
    JointStateBuffer,
    JointStateView,
    JointTargetBuffer,
    JointTargetView,
)


def _same_names(values: Sequence[str], expected: tuple[str, ...]) -> bool:
    if len(values) != len(expected):
        return False
    for index in range(len(expected)):
        if values[index] != expected[index]:
            return False
    return True


class NamedJointStateSource:
    """Normalize name-carrying messages into one stable state layout."""

    def __init__(
        self,
        canonical_layout: JointLayout | None = None,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> None:
        self._canonical_layout = canonical_layout
        self._dtype = np.dtype(dtype)
        self._buffer = (
            JointStateBuffer(canonical_layout, dtype=self._dtype)
            if canonical_layout is not None
            else None
        )
        self._message_names: tuple[str, ...] | None = None
        self._message_map: CompiledJointMap | None = None

    @property
    def ready(self) -> bool:
        return self._buffer is not None and self._message_map is not None

    @property
    def layout(self) -> JointLayout:
        if self._buffer is None:
            raise RuntimeError("named joint source has not received its first message")
        return self._buffer.layout

    @property
    def view(self) -> JointStateView:
        if not self.ready or self._buffer is None:
            raise RuntimeError("named joint source is not ready")
        return self._buffer.view

    def update(
        self,
        names: Sequence[str],
        position: object,
        velocity: object,
        *,
        timestamp_ns: int = 0,
    ) -> JointStateView:
        if self._message_names is None or not _same_names(names, self._message_names):
            message_layout = JointLayout.create(names, label="incoming joint state")
            if self._buffer is None:
                canonical = self._canonical_layout or message_layout
                self._buffer = JointStateBuffer(canonical, dtype=self._dtype)
            self._message_map = CompiledJointMap.compile(
                message_layout,
                self._buffer.layout,
            )
            self._message_names = message_layout.names

        assert self._buffer is not None
        assert self._message_map is not None
        self._message_map.map_into(position, self._buffer.position)
        self._message_map.map_into(velocity, self._buffer.velocity)
        self._buffer.view.timestamp_ns = int(timestamp_ns)
        return self._buffer.view


class FixedOrderJointStateSource(JointStateBuffer):
    """State source for transports whose array order is declared once."""


class NamedJointCommandEncoder:
    """Return targets unchanged for a transport that carries joint names."""

    def __init__(self, *, supports_partial: bool = False) -> None:
        self.supports_partial = bool(supports_partial)

    def encode(
        self,
        target: JointTargetView,
        *,
        required_layout: JointLayout | None = None,
    ) -> JointTargetView:
        if required_layout is not None:
            target_names = set(target.layout.names)
            required_names = set(required_layout.names)
            extra = target_names - required_names
            if extra:
                raise ValueError(
                    f"named command contains joints outside required layout: "
                    f"{tuple(sorted(extra))}"
                )
            if not self.supports_partial and target_names != required_names:
                raise ValueError(
                    "named command transport requires a complete target; "
                    "enable supports_partial explicitly for subset commands"
                )
        return target


class FixedOrderJointCommandEncoder:
    """Reorder complete named targets into a hardware fixed layout."""

    def __init__(self, hardware_layout: JointLayout) -> None:
        self.hardware_layout = hardware_layout
        self._buffer = JointTargetBuffer(hardware_layout)
        self._source_layout: JointLayout | None = None
        self._mapping: CompiledJointMap | None = None

    @property
    def target(self) -> JointTargetView:
        return self._buffer.view

    def encode(self, target: JointTargetView) -> JointTargetView:
        if self._source_layout != target.layout:
            self._mapping = CompiledJointMap.compile(
                target.layout,
                self.hardware_layout,
                require_exact=True,
            )
            self._source_layout = target.layout
        assert self._mapping is not None
        self._mapping.map_into(target.position, self._buffer.position)
        self._mapping.map_into(target.kp, self._buffer.kp)
        self._mapping.map_into(target.kd, self._buffer.kd)
        return self._buffer.view


__all__ = [
    "FixedOrderJointCommandEncoder",
    "FixedOrderJointStateSource",
    "NamedJointCommandEncoder",
    "NamedJointStateSource",
]
