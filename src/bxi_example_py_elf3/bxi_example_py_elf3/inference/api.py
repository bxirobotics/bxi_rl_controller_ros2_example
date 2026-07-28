"""Stable, backend-neutral policy input and output types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(slots=True)
class InferenceFrame:
    q: FloatArray
    dq: FloatArray
    quat_wxyz: FloatArray
    angular_velocity: FloatArray
    command: FloatArray | None = None
    base_linear_velocity: FloatArray | None = None
    world_position: FloatArray | None = None
    depth: FloatArray | None = None
    depth_frame_id: int | None = None
    timestamp_ns: int = 0


@dataclass(slots=True)
class PolicyOutput:
    joint_position: FloatArray
    estimated_velocity: FloatArray | None = None
    completed: bool = False


__all__ = ["FloatArray", "InferenceFrame", "PolicyOutput"]
