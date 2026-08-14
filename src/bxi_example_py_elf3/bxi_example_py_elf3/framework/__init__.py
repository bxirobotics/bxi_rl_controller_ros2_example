"""Portable control framework, independent from any specific robot."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inspection import StateMachineInspector

__all__ = ["StateMachineInspector"]


def __getattr__(name: str):
    if name == "StateMachineInspector":
        from .inspection import StateMachineInspector

        return StateMachineInspector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
