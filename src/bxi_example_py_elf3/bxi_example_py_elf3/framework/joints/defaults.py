"""Explicit platform defaults for joints omitted by a state or policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class JointDefault:
    """One safe fixed command used when a state does not control a joint."""

    position: float
    kp: float
    kd: float

    def __post_init__(self) -> None:
        values = (self.position, self.kp, self.kd)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint default position, kp and kd must be finite")
        if self.kp < 0.0 or self.kd < 0.0:
            raise ValueError("joint default kp and kd must be non-negative")


class JointCommandDefaults:
    """Class-configured, name-addressed platform command defaults.

    The mapping is consulted only while a source layout is compiled. The
    control-cycle path uses prebuilt NumPy indices and arrays.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, JointDefault] | None = None) -> None:
        copied = dict(values or {})
        invalid_names = tuple(
            name for name in copied if not isinstance(name, str) or not name
        )
        if invalid_names:
            raise ValueError("joint default names must be non-empty strings")
        invalid_values = tuple(
            name for name, value in copied.items() if not isinstance(value, JointDefault)
        )
        if invalid_values:
            raise TypeError(
                "joint defaults must be JointDefault instances: "
                f"{invalid_values}"
            )
        self._values = MappingProxyType(copied)

    def require(self, names: tuple[str, ...]) -> tuple[JointDefault, ...]:
        missing = tuple(name for name in names if name not in self._values)
        if missing:
            raise ValueError(
                "cannot expand the state/policy MotorFrame to the complete robot "
                "joint layout: the output does not command robot joints "
                f"{missing}, and they have no explicit JointCommandDefaults. "
                "The framework refuses to guess their position/kp/kd or reuse a "
                "previous command. Add a JointDefault(position=..., kp=..., "
                "kd=...) for every listed joint in the platform's "
                "command_defaults, or make the state/policy output those joints "
                "in its MotorFrame. Full-layout outputs do not use defaults."
            )
        return tuple(self._values[name] for name in names)


__all__ = ["JointCommandDefaults", "JointDefault"]
