from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from rclpy.node import Node

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

    ``current_*`` values are the observation snapshot for the current control
    cycle.  ``*_last`` arrays describe the most recently published motor
    frame.  Target poses and gains belong to states, not this context.
    Advanced ROS integrations should use :attr:`ros_node` instead of depending
    on the concrete controller class.
    """

    dt: float
    loop_count: int
    dof_num: int
    current_q: FloatArray
    current_dq: FloatArray
    current_quat_xyzw: FloatArray
    current_quat_wxyz: FloatArray
    current_omega: FloatArray
    current_raw_cmd_vel: FloatArray
    current_cmd_vel: FloatArray
    qpos: FloatArray
    quat_xyzw: FloatArray
    pos_last: FloatArray
    kp_last: FloatArray
    kd_last: FloatArray
    pos_last_state: FloatArray
    kp_last_state: FloatArray
    kd_last_state: FloatArray
    speed_profiles: Mapping[str, object]

    @property
    def ros_node(self) -> "Node":
        """Return the underlying ROS node for advanced integrations."""
        ...

    def set_motor_target(self, qpos: object, kp: object, kd: object) -> None:
        ...

    def request_state(
        self,
        state_name: str,
        *,
        trigger: str,
        transition: "TransitionSpec" = None,
        delay: float = 0.0,
    ) -> None:
        ...

    def preheat_model(
        self,
        model: object,
        with_cmd_vel: bool = False,
        cmd_vel: object = None,
    ) -> None:
        ...

    def is_orientation_unsafe(self, quat_xyzw: object) -> bool:
        ...

    def get_logger(self) -> LoggerLike:
        ...


__all__ = ["LoggerLike", "RobotControlContext"]
