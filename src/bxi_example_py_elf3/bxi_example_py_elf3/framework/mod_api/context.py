from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import JointLayout, JointStateView

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.inference import InferenceFrame
    from rclpy.node import Node

    from .frame import MotorFrame
    from .transition import TransitionSpec


FloatArray = NDArray[np.floating]


class LoggerLike(Protocol):
    """Logging calls guaranteed by the controller context."""

    def debug(self, message: str) -> None:
        ...

    def info(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...


class RobotControlContext(Protocol):
    """Stable controller surface available to states and transitions.

    ``robot_joints`` is the complete named observation snapshot for the current
    control cycle. ``last_motor_frame`` is always resolved into that same full
    robot layout. Target poses and gains belong to states, not this context.
    Advanced ROS integrations should use :attr:`ros_node` instead of depending
    on the concrete controller class.
    """

    dt: float
    loop_count: int
    robot_layout: JointLayout
    robot_joints: JointStateView
    inference_frame: "InferenceFrame"
    current_quat_xyzw: FloatArray
    current_quat_wxyz: FloatArray
    current_omega: FloatArray
    current_raw_cmd_vel: FloatArray
    current_cmd_vel: FloatArray
    last_motor_frame: "MotorFrame"
    speed_profiles: Mapping[str, object]

    @property
    def ros_node(self) -> "Node":
        """Return the underlying ROS node for advanced integrations."""
        ...

    def resolve_motor_frame(
        self,
        frame: "MotorFrame",
        output: "MotorFrame",
    ) -> "MotorFrame":
        ...

    def set_motor_target(self, frame: "MotorFrame") -> None:
        ...

    def request_state(
        self,
        state_name: str,
        *,
        trigger: str,
        transition: "TransitionSpec" = None,
        delay: float = 0.0,
        force: bool = False,
    ) -> bool:
        """Request a state change and report whether it was accepted.

        ``True`` means the transition started or was queued.  It does not mean
        that a non-instant transition has already finished.  ``False`` means
        the target is unavailable or its Mod node could not be started, so no
        switch was started.  ``force=True`` bypasses only the target's
        availability check and should be reserved for exceptional operations.
        """
        ...

    def preheat_model(
        self,
        model: object,
        command: object | None = None,
    ) -> None:
        ...

    def is_orientation_unsafe(self, quat_xyzw: object) -> bool:
        ...

__all__ = ["LoggerLike", "RobotControlContext"]
