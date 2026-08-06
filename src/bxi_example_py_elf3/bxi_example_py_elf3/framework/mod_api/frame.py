from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import JointLayout


FloatArray: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True, init=False)
class MotorFrame:
    """A complete named MIT motor command in its natural N-joint layout.

    The five MIT fields are always present. Callers may omit velocity and
    feed-forward torque when constructing or updating a frame; those fields
    are then explicitly filled with zero.
    """

    layout: JointLayout
    qpos: FloatArray
    kp: FloatArray
    kd: FloatArray
    vel: FloatArray
    torque: FloatArray

    def __init__(
        self,
        layout: JointLayout,
        qpos: FloatArray,
        kp: FloatArray,
        kd: FloatArray,
        vel: FloatArray | None = None,
        torque: FloatArray | None = None,
    ) -> None:
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "kd", kd)
        object.__setattr__(
            self,
            "vel",
            np.zeros(layout.dof_num, dtype=np.float32) if vel is None else vel,
        )
        object.__setattr__(
            self,
            "torque",
            np.zeros(layout.dof_num, dtype=np.float32) if torque is None else torque,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        expected = (self.layout.dof_num,)
        for name, array in (
            ("qpos", self.qpos),
            ("kp", self.kp),
            ("kd", self.kd),
            ("vel", self.vel),
            ("torque", self.torque),
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
        *,
        vel: object | None = None,
        torque: object | None = None,
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
        mit_arrays = tuple(
            np.zeros(layout.dof_num, dtype=np.float32)
            if value is None
            else np.asarray(value, dtype=np.float32).copy()
            for value in (vel, torque)
        )
        vel_array, torque_array = mit_arrays
        for name, array in (("vel", vel_array), ("torque", torque_array)):
            if array.shape != expected:
                raise ValueError(
                    f"motor frame {name} has shape {array.shape}, expected {expected}"
                )
        return cls(
            layout=layout,
            qpos=qpos_array,
            kp=kp_array,
            kd=kd_array,
            vel=vel_array,
            torque=torque_array,
        )

    @classmethod
    def empty(
        cls,
        layout: JointLayout,
    ) -> "MotorFrame":
        """Allocate one reusable command frame for a long-lived owner."""
        return cls(
            layout=layout,
            qpos=np.empty(layout.dof_num, dtype=np.float32),
            kp=np.empty(layout.dof_num, dtype=np.float32),
            kd=np.empty(layout.dof_num, dtype=np.float32),
            vel=np.zeros(layout.dof_num, dtype=np.float32),
            torque=np.zeros(layout.dof_num, dtype=np.float32),
        )

    def update(
        self,
        qpos: object,
        kp: object,
        kd: object,
        *,
        vel: object | None = None,
        torque: object | None = None,
    ) -> "MotorFrame":
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
        self._update_mit_field("vel", vel, self.vel, expected)
        self._update_mit_field("torque", torque, self.torque, expected)
        return self

    @staticmethod
    def _update_mit_field(
        name: str,
        source: object | None,
        target: FloatArray,
        expected: tuple[int],
    ) -> None:
        if source is None:
            target.fill(0.0)
            return
        array = np.asarray(source)
        if array.shape != expected:
            raise ValueError(
                f"motor frame {name} has shape {array.shape}, expected {expected}"
            )
        np.copyto(target, array, casting="same_kind")

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.qpos
        yield self.kp
        yield self.kd


__all__ = ["FloatArray", "MotorFrame"]
