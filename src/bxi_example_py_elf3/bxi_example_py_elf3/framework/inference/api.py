"""Stable backend-neutral inference input and output types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import JointStateView, JointTargetView


FloatArray = NDArray[np.float32]


@dataclass(slots=True)
class InferenceFrame:
    joints: JointStateView
    quat_wxyz: NDArray[np.floating]
    angular_velocity: NDArray[np.floating]
    command: NDArray[np.floating] | None = None
    base_linear_velocity: NDArray[np.floating] | None = None
    world_position: NDArray[np.floating] | None = None
    depth: NDArray[np.floating] | None = None
    depth_frame_id: int | None = None
    timestamp_ns: int = 0
    # Raw body-frame specific force from the platform IMU.  Policies that
    # were trained with an accelerometer must reject a missing value rather
    # than silently substituting zeros.
    linear_acceleration: NDArray[np.floating] | None = None


@dataclass(slots=True)
class PolicyOutput:
    joints: JointTargetView
    estimated_velocity: FloatArray | None = None
    completed: bool = False


__all__ = ["FloatArray", "InferenceFrame", "PolicyOutput"]
