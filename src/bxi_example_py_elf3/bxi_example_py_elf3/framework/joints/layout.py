"""Stable semantic joint layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class JointLayout:
    """An immutable ordered set of joint names.

    Names are the semantic identity. Numeric indices are only a compiled
    implementation detail and must never cross a component boundary alone.
    """

    names: tuple[str, ...]
    label: str = field(default="", compare=False)
    _index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names:
            raise ValueError("joint layout must contain at least one joint")
        invalid = [name for name in names if not isinstance(name, str) or not name]
        if invalid:
            raise ValueError("joint names must be non-empty strings")
        if len(set(names)) != len(names):
            duplicates = sorted(
                {name for name in names if names.count(name) > 1}
            )
            raise ValueError(f"joint layout contains duplicate names: {duplicates}")
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "_index",
            MappingProxyType({name: index for index, name in enumerate(names)}),
        )

    @classmethod
    def create(cls, names: Sequence[str], *, label: str = "") -> "JointLayout":
        return cls(tuple(names), label=label)

    @property
    def dof_num(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError as exc:
            prefix = f"joint layout '{self.label}'" if self.label else "joint layout"
            raise KeyError(f"{prefix} does not contain joint '{name}'") from exc

    def missing_from(self, source: "JointLayout") -> tuple[str, ...]:
        return tuple(name for name in self.names if name not in source._index)

    def select(self, names: Sequence[str], *, label: str = "") -> "JointLayout":
        selected = tuple(names)
        missing = tuple(name for name in selected if name not in self._index)
        if missing:
            raise ValueError(f"selected joints are not in source layout: {missing}")
        return JointLayout(selected, label=label)


__all__ = ["JointLayout"]
