"""Hardware coordinate calibration independent from joint ordering."""

from __future__ import annotations

import numpy as np

from .layout import JointLayout


class JointCalibration:
    """Convert hardware coordinates to semantic joint coordinates and back."""

    def __init__(
        self,
        layout: JointLayout,
        *,
        direction: object | None = None,
        zero_offset: object | None = None,
    ) -> None:
        self.layout = layout
        count = layout.dof_num
        self.direction = (
            np.ones(count, dtype=np.float64)
            if direction is None
            else np.asarray(direction, dtype=np.float64).reshape(count).copy()
        )
        self.zero_offset = (
            np.zeros(count, dtype=np.float64)
            if zero_offset is None
            else np.asarray(zero_offset, dtype=np.float64).reshape(count).copy()
        )
        if not np.all((self.direction == 1.0) | (self.direction == -1.0)):
            raise ValueError("joint calibration direction must contain only +1 or -1")

    def state_into(
        self,
        hardware_position: object,
        hardware_velocity: object,
        semantic_position: object,
        semantic_velocity: object,
    ) -> None:
        np.multiply(hardware_position, self.direction, out=semantic_position)
        np.add(semantic_position, self.zero_offset, out=semantic_position)
        np.multiply(hardware_velocity, self.direction, out=semantic_velocity)

    def position_to_hardware_into(
        self,
        semantic_position: object,
        hardware_position: object,
    ) -> None:
        np.subtract(semantic_position, self.zero_offset, out=hardware_position)
        np.multiply(hardware_position, self.direction, out=hardware_position)


__all__ = ["JointCalibration"]
