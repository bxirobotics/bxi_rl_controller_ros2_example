from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class MotorFrame:
    """A complete motor command with owned float32 arrays."""

    qpos: FloatArray
    kp: FloatArray
    kd: FloatArray

    @classmethod
    def create(cls, qpos: object, kp: object, kd: object) -> "MotorFrame":
        arrays = tuple(
            np.asarray(value, dtype=np.float32).copy() for value in (qpos, kp, kd)
        )
        qpos_array, kp_array, kd_array = arrays
        if qpos_array.shape != kp_array.shape or qpos_array.shape != kd_array.shape:
            raise ValueError(
                "motor frame shapes must match: "
                f"qpos={qpos_array.shape}, kp={kp_array.shape}, kd={kd_array.shape}"
            )
        return cls(qpos=qpos_array, kp=kp_array, kd=kd_array)

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.qpos
        yield self.kp
        yield self.kd


__all__ = ["FloatArray", "MotorFrame"]
