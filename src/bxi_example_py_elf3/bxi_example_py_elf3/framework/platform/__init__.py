"""Public motor-control platform helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import ControlPlatformAdapter, RobotObservation
    from .joint_io import (
        FixedOrderJointCommandEncoder,
        FixedOrderJointStateSource,
        NamedJointCommandEncoder,
        NamedJointStateSource,
    )


def __getattr__(name: str):
    if name in {"ControlPlatformAdapter", "RobotObservation"}:
        from .api import ControlPlatformAdapter, RobotObservation

        return {
            "ControlPlatformAdapter": ControlPlatformAdapter,
            "RobotObservation": RobotObservation,
        }[name]
    if name in {
        "FixedOrderJointCommandEncoder",
        "FixedOrderJointStateSource",
        "NamedJointCommandEncoder",
        "NamedJointStateSource",
    }:
        from .joint_io import (
            FixedOrderJointCommandEncoder,
            FixedOrderJointStateSource,
            NamedJointCommandEncoder,
            NamedJointStateSource,
        )

        return {
            "FixedOrderJointCommandEncoder": FixedOrderJointCommandEncoder,
            "FixedOrderJointStateSource": FixedOrderJointStateSource,
            "NamedJointCommandEncoder": NamedJointCommandEncoder,
            "NamedJointStateSource": NamedJointStateSource,
        }[name]
    raise AttributeError(name)

__all__ = [
    "ControlPlatformAdapter",
    "FixedOrderJointCommandEncoder",
    "FixedOrderJointStateSource",
    "NamedJointCommandEncoder",
    "NamedJointStateSource",
    "RobotObservation",
]
