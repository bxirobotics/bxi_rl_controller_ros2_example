"""Precompiled allocation-free resolution of partial named motor frames."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .defaults import JointCommandDefaults
from .layout import JointLayout


Float32Array = NDArray[np.float32]


class CommandFrame(Protocol):
    layout: JointLayout
    qpos: Float32Array
    kp: Float32Array
    kd: Float32Array


class CompiledCommandBinding:
    """One source-layout to robot-layout mapping with fixed reusable metadata."""

    __slots__ = (
        "source_layout",
        "robot_layout",
        "is_identity",
        "_source_indices",
        "_missing_indices",
        "_missing_position",
        "_missing_kp",
        "_missing_kd",
    )

    def __init__(
        self,
        source_layout: JointLayout,
        robot_layout: JointLayout,
        defaults: JointCommandDefaults,
    ) -> None:
        robot_names = set(robot_layout.names)
        source_names = set(source_layout.names)
        unknown = tuple(
            name for name in source_layout.names if name not in robot_names
        )
        if unknown:
            raise ValueError(
                "state output contains joints that do not exist in the robot "
                f"layout: {unknown}"
            )

        missing_names = tuple(
            name for name in robot_layout.names if name not in source_names
        )
        missing_defaults = defaults.require(missing_names)
        self.source_layout = source_layout
        self.robot_layout = robot_layout
        self.is_identity = source_layout.names == robot_layout.names
        self._source_indices = self._readonly_indices(
            tuple(robot_layout.index(name) for name in source_layout.names)
        )
        self._missing_indices = self._readonly_indices(
            tuple(robot_layout.index(name) for name in missing_names)
        )
        self._missing_position = self._readonly_values(
            tuple(value.position for value in missing_defaults)
        )
        self._missing_kp = self._readonly_values(
            tuple(value.kp for value in missing_defaults)
        )
        self._missing_kd = self._readonly_values(
            tuple(value.kd for value in missing_defaults)
        )

    def resolve_into(self, source: CommandFrame, output: CommandFrame) -> None:
        """Resolve ``source`` into a caller-owned full robot frame."""
        if (
            source.layout is not self.source_layout
            and source.layout.names != self.source_layout.names
        ):
            raise ValueError("source motor frame does not match compiled joint layout")
        if (
            output.layout is not self.robot_layout
            and output.layout.names != self.robot_layout.names
        ):
            raise ValueError("output motor frame does not match robot joint layout")

        if self.is_identity:
            for source_values, output_values in (
                (source.qpos, output.qpos),
                (source.kp, output.kp),
                (source.kd, output.kd),
            ):
                np.copyto(output_values, source_values)
            return

        for source_values, output_values, missing_values in (
            (source.qpos, output.qpos, self._missing_position),
            (source.kp, output.kp, self._missing_kp),
            (source.kd, output.kd, self._missing_kd),
        ):
            output_values[self._missing_indices] = missing_values
            output_values[self._source_indices] = source_values

    @staticmethod
    def _readonly_indices(values: tuple[int, ...]) -> NDArray[np.intp]:
        result = np.asarray(values, dtype=np.intp)
        result.flags.writeable = False
        return result

    @staticmethod
    def _readonly_values(values: tuple[float, ...]) -> Float32Array:
        result = np.asarray(values, dtype=np.float32)
        result.flags.writeable = False
        return result


class JointCommandResolver:
    """Resolve N-joint state outputs into one complete robot command layout."""

    __slots__ = (
        "robot_layout",
        "_defaults",
        "_bindings",
        "_last_source_layout",
        "_last_binding",
    )

    def __init__(
        self,
        robot_layout: JointLayout,
        defaults: JointCommandDefaults,
    ) -> None:
        self.robot_layout = robot_layout
        self._defaults = defaults
        self._bindings: dict[tuple[str, ...], CompiledCommandBinding] = {}
        self._last_source_layout: JointLayout | None = None
        self._last_binding: CompiledCommandBinding | None = None

    def compile(self, source_layout: JointLayout) -> CompiledCommandBinding:
        if self._last_source_layout is source_layout:
            assert self._last_binding is not None
            return self._last_binding
        key = source_layout.names
        binding = self._bindings.get(key)
        if binding is None:
            binding = CompiledCommandBinding(
                source_layout,
                self.robot_layout,
                self._defaults,
            )
            self._bindings[key] = binding
        self._last_source_layout = source_layout
        self._last_binding = binding
        return binding

    def resolve_into(self, source: CommandFrame, output: CommandFrame) -> None:
        self.compile(source.layout).resolve_into(source, output)


__all__ = [
    "CommandFrame",
    "CompiledCommandBinding",
    "JointCommandResolver",
]
