"""Public robot platform integration API."""

from typing import TYPE_CHECKING

from .api import ControlPlatformAdapter, RobotObservation
from .joint_io import (
    FixedOrderJointCommandEncoder,
    FixedOrderJointStateSource,
    NamedJointCommandEncoder,
    NamedJointStateSource,
)
if TYPE_CHECKING:
    from .runtime import ControlRuntimeConfig, RobotControlRuntime


def __getattr__(name: str):
    if name in {"ControlRuntimeConfig", "RobotControlRuntime"}:
        from .runtime import ControlRuntimeConfig, RobotControlRuntime

        return {
            "ControlRuntimeConfig": ControlRuntimeConfig,
            "RobotControlRuntime": RobotControlRuntime,
        }[name]
    raise AttributeError(name)

__all__ = [
    "ControlPlatformAdapter",
    "ControlRuntimeConfig",
    "FixedOrderJointCommandEncoder",
    "FixedOrderJointStateSource",
    "NamedJointCommandEncoder",
    "NamedJointStateSource",
    "RobotControlRuntime",
    "RobotObservation",
]
