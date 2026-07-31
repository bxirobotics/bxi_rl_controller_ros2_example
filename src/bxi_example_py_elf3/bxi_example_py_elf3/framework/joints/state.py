"""Reusable named joint state buffers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layout import JointLayout


FloatArray = NDArray[np.floating]


@dataclass(slots=True)
class JointStateView:
    layout: JointLayout
    position: FloatArray
    velocity: FloatArray
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        expected = (self.layout.dof_num,)
        if self.position.shape != expected or self.velocity.shape != expected:
            raise ValueError(
                "joint state shapes must match layout: "
                f"position={self.position.shape}, velocity={self.velocity.shape}, "
                f"expected={expected}"
            )
        if not np.issubdtype(self.position.dtype, np.floating) or not np.issubdtype(
            self.velocity.dtype, np.floating
        ):
            raise TypeError("joint state position and velocity must be floating arrays")


class JointStateBuffer:
    """Own stable position/velocity arrays and one stable view object."""

    def __init__(
        self,
        layout: JointLayout,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> None:
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("joint state dtype must be floating")
        self.layout = layout
        self.position = np.zeros(layout.dof_num, dtype=dtype)
        self.velocity = np.zeros(layout.dof_num, dtype=dtype)
        self.view = JointStateView(layout, self.position, self.velocity)

    def update(
        self,
        position: object,
        velocity: object,
        *,
        timestamp_ns: int = 0,
    ) -> JointStateView:
        position_array = np.asarray(position)
        velocity_array = np.asarray(velocity)
        expected = (self.layout.dof_num,)
        if position_array.shape != expected or velocity_array.shape != expected:
            raise ValueError(
                "joint state update shapes do not match layout: "
                f"position={position_array.shape}, velocity={velocity_array.shape}, "
                f"expected={expected}"
            )
        np.copyto(self.position, position_array, casting="same_kind")
        np.copyto(self.velocity, velocity_array, casting="same_kind")
        self.view.timestamp_ns = int(timestamp_ns)
        return self.view


__all__ = ["FloatArray", "JointStateBuffer", "JointStateView"]
