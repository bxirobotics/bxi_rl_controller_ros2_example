"""Immutable parameter arrays bound to an explicit named joint layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layout import JointLayout


@dataclass(frozen=True, slots=True, eq=False)
class JointParameterSet:
    """Default position, gains and action scale in one named joint layout."""

    layout: JointLayout
    default_position: NDArray[np.float32]
    kp: NDArray[np.float32]
    kd: NDArray[np.float32]
    action_scale: NDArray[np.float32]

    @classmethod
    def from_arrays(
        cls,
        layout: JointLayout,
        *,
        default_position: object,
        kp: object,
        kd: object,
        action_scale: object,
    ) -> "JointParameterSet":
        """Bind four one-dimensional arrays to one explicit joint layout."""

        expected = (layout.dof_num,)
        columns = []
        for name, values in (
            ("default_position", default_position),
            ("kp", kp),
            ("kd", kd),
            ("action_scale", action_scale),
        ):
            column = np.array(values, dtype=np.float32, copy=True, order="C")
            if column.shape != expected:
                raise ValueError(
                    f"{name} has shape {column.shape}, expected {expected} for "
                    f"'{layout.label}'"
                )
            column.flags.writeable = False
            columns.append(column)
        return cls(layout, *columns)

    @classmethod
    def from_rows(
        cls,
        layout: JointLayout,
        rows: tuple[tuple[str, float, float, float, float], ...],
    ) -> "JointParameterSet":
        """Build from ``(name, default, kp, kd, scale)`` rows."""

        names = tuple(row[0] for row in rows)
        if names != layout.names:
            raise ValueError(
                f"parameter rows do not match '{layout.label}' layout: {names}"
            )
        values = np.asarray([row[1:] for row in rows], dtype=np.float32)
        return cls.from_arrays(
            layout,
            default_position=values[:, 0],
            kp=values[:, 1],
            kd=values[:, 2],
            action_scale=values[:, 3],
        )

    def select(self, layout: JointLayout) -> "JointParameterSet":
        """Return these parameters in a named subset/order."""

        if layout.names == self.layout.names:
            return self
        indices = tuple(self.layout.index(name) for name in layout.names)
        return type(self).from_arrays(
            layout,
            default_position=self.default_position[list(indices)],
            kp=self.kp[list(indices)],
            kd=self.kd[list(indices)],
            action_scale=self.action_scale[list(indices)],
        )


__all__ = ["JointParameterSet"]
