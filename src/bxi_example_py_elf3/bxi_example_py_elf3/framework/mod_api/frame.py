from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import (
    CompiledJointMap,
    JointLayout,
    JointTargetView,
)


FloatArray: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class MotorFrame:
    """A complete motor command with owned float32 arrays."""

    layout: JointLayout
    qpos: FloatArray
    kp: FloatArray
    kd: FloatArray

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

    @classmethod
    def from_target(
        cls,
        target: JointTargetView,
        control_layout: JointLayout,
    ) -> "MotorFrame":
        mapping = CompiledJointMap.compile(
            target.layout,
            control_layout,
            require_exact=True,
        )
        qpos = np.empty(control_layout.dof_num, dtype=np.float32)
        kp = np.empty(control_layout.dof_num, dtype=np.float32)
        kd = np.empty(control_layout.dof_num, dtype=np.float32)
        mapping.map_into(target.position, qpos)
        mapping.map_into(target.kp, kp)
        mapping.map_into(target.kd, kd)
        return cls(control_layout, qpos, kp, kd)

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.qpos
        yield self.kp
        yield self.kd


__all__ = ["FloatArray", "MotorFrame"]
