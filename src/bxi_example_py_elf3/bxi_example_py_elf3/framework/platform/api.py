"""Public platform boundary types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import JointStateView
from bxi_example_py_elf3.framework.mod_api import MotorFrame


FloatArray = NDArray[np.floating]


@dataclass(slots=True)
class RobotObservation:
    """One coherent platform observation for a control cycle."""

    joints: JointStateView
    quat_xyzw: FloatArray
    quat_wxyz: FloatArray
    omega: FloatArray
    raw_cmd_vel: FloatArray
    linear_acceleration: FloatArray | None = None


class ControlPlatformAdapter(Protocol):
    """Minimal platform boundary used by :class:`RobotControlRuntime`."""

    def startup_step(self, now: float) -> bool:
        ...

    def snapshot_control_inputs(
        self,
    ) -> tuple[RobotObservation, Sequence[str]]:
        ...

    def publish_motor_frame(self, frame: MotorFrame) -> None:
        ...


__all__ = ["ControlPlatformAdapter", "RobotObservation"]
