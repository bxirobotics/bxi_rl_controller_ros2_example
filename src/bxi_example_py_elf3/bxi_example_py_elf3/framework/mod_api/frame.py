from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import JointLayout


FloatArray: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class MotorFrame:
    """A named state command in its natural N-joint output layout."""

    layout: JointLayout
    qpos: FloatArray
    kp: FloatArray
    kd: FloatArray

    def __post_init__(self) -> None:
        expected = (self.layout.dof_num,)
        for name, array in (
            ("qpos", self.qpos),
            ("kp", self.kp),
            ("kd", self.kd),
        ):
            if array.shape != expected:
                raise ValueError(
                    f"motor frame {name} has shape {array.shape}, expected {expected}"
                )
            if array.dtype != np.float32:
                raise TypeError(f"motor frame {name} must use float32")

    @classmethod
    def create(
        cls,
        layout: JointLayout,
        qpos: object,
        kp: object,
        kd: object,
    ) -> "MotorFrame":
        arrays = tuple(
            np.asarray(value, dtype=np.float32).copy() for value in (qpos, kp, kd)
        )
        qpos_array, kp_array, kd_array = arrays
        expected = (layout.dof_num,)
        if (
            qpos_array.shape != expected
            or kp_array.shape != expected
            or kd_array.shape != expected
        ):
            raise ValueError(
                "motor frame shapes must match its joint layout: "
                f"qpos={qpos_array.shape}, kp={kp_array.shape}, kd={kd_array.shape}"
                f", expected={expected}"
            )
        return cls(layout=layout, qpos=qpos_array, kp=kp_array, kd=kd_array)

    @classmethod
    def empty(cls, layout: JointLayout) -> "MotorFrame":
        """Allocate one reusable command frame for a long-lived owner."""
        return cls(
            layout=layout,
            qpos=np.empty(layout.dof_num, dtype=np.float32),
            kp=np.empty(layout.dof_num, dtype=np.float32),
            kd=np.empty(layout.dof_num, dtype=np.float32),
        )

    def update(self, qpos: object, kp: object, kd: object) -> "MotorFrame":
        """Overwrite this frame in place and return it."""
        expected = (self.layout.dof_num,)
        for name, source, target in (
            ("qpos", qpos, self.qpos),
            ("kp", kp, self.kp),
            ("kd", kd, self.kd),
        ):
            array = np.asarray(source)
            if array.shape != expected:
                raise ValueError(
                    f"motor frame {name} has shape {array.shape}, expected {expected}"
                )
            np.copyto(target, array, casting="same_kind")
        return self

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.qpos
        yield self.kp
        yield self.kd


__all__ = ["FloatArray", "MotorFrame"]
