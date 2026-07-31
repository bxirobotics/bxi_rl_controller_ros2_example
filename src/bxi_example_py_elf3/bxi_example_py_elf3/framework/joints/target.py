"""Joint target views and persistent buffers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layout import JointLayout


Float32Array = NDArray[np.float32]


@dataclass(slots=True)
class JointTargetView:
    layout: JointLayout
    position: Float32Array
    kp: Float32Array
    kd: Float32Array

    def __post_init__(self) -> None:
        expected = (self.layout.dof_num,)
        arrays = (self.position, self.kp, self.kd)
        if any(array.shape != expected for array in arrays):
            raise ValueError(
                "joint target shapes must match layout: "
                f"position={self.position.shape}, kp={self.kp.shape}, "
                f"kd={self.kd.shape}, expected={expected}"
            )
        if any(array.dtype != np.float32 for array in arrays):
            raise TypeError("joint target arrays must use float32")


class JointTargetBuffer:
    """Own one reusable full target in a declared layout."""

    def __init__(self, layout: JointLayout) -> None:
        self.layout = layout
        self.position = np.zeros(layout.dof_num, dtype=np.float32)
        self.kp = np.zeros(layout.dof_num, dtype=np.float32)
        self.kd = np.zeros(layout.dof_num, dtype=np.float32)
        self.view = JointTargetView(
            layout,
            self.position,
            self.kp,
            self.kd,
        )

    def update(self, position: object, kp: object, kd: object) -> JointTargetView:
        values = (position, kp, kd)
        outputs = (self.position, self.kp, self.kd)
        expected = (self.layout.dof_num,)
        for name, value, output in zip(("position", "kp", "kd"), values, outputs):
            array = np.asarray(value)
            if array.shape != expected:
                raise ValueError(
                    f"joint target {name} has shape {array.shape}, expected {expected}"
                )
            if array is not output:
                np.copyto(output, array, casting="same_kind")
        return self.view

    def update_position(self, position: object) -> JointTargetView:
        array = np.asarray(position)
        expected = (self.layout.dof_num,)
        if array.shape != expected:
            raise ValueError(
                f"joint target position has shape {array.shape}, expected {expected}"
            )
        if array is not self.position:
            np.copyto(self.position, array, casting="same_kind")
        return self.view


__all__ = ["Float32Array", "JointTargetBuffer", "JointTargetView"]
