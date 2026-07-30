from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Protocol, TypeVar, Union, get_args, get_origin, get_type_hints

from .resource import (
    ResourceFactory,
    ResourceHandle,
    ResourceKey,
    ResourcePolicy,
    ResourceT,
)
from .state import RobotControlState
from .transition import TransitionPlugin


ParamT = TypeVar("ParamT")
ParamsT = TypeVar("ParamsT")
StateFactory = Callable[["StateBuildContext"], RobotControlState]


@dataclass(frozen=True)
class StateBuildContext:
    name: str
    state_id: int
    params: Mapping[str, object]
    _consumed: set[str] = field(default_factory=set, compare=False, repr=False)

    def int_param(self, name: str, default: int) -> int:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"state '{self.name}' param '{name}' must be an integer")
        return value

    def float_param(self, name: str, default: float) -> float:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"state '{self.name}' param '{name}' must be a number")
        return float(value)

    def string_param(self, name: str, default: str) -> str:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if not isinstance(value, str):
            raise ValueError(f"state '{self.name}' param '{name}' must be a string")
        return value

    def bool_param(self, name: str, default: bool) -> bool:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"state '{self.name}' param '{name}' must be a boolean")
        return value

    def param(self, name: str, expected: type[ParamT], default: ParamT) -> ParamT:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if not isinstance(value, expected):
            raise ValueError(
                f"state '{self.name}' param '{name}' must be {expected.__name__}"
            )
        return value

    def dataclass_params(self, params_type: type[ParamsT]) -> ParamsT:
        """Build a typed parameter object with strict YAML validation."""
        if not isinstance(params_type, type) or not is_dataclass(params_type):
            raise TypeError("dataclass_params() expects a dataclass type")
        try:
            annotations = get_type_hints(params_type)
        except (NameError, TypeError) as exc:
            raise TypeError(
                f"state '{self.name}' cannot resolve annotations for "
                f"{params_type.__name__}: {exc}"
            ) from exc

        values: dict[str, object] = {}
        for parameter in fields(params_type):
            name = parameter.name
            self._consumed.add(name)
            if name not in self.params:
                if (
                    parameter.default is MISSING
                    and parameter.default_factory is MISSING
                ):
                    raise ValueError(
                        f"state '{self.name}' is missing required param '{name}'"
                    )
                continue
            values[name] = _typed_dataclass_value(
                self.name,
                name,
                self.params[name],
                annotations.get(name, parameter.type),
            )
        return params_type(**values)

    def finish(self) -> None:
        unknown = set(self.params) - self._consumed
        if unknown:
            raise ValueError(
                f"state '{self.name}' has unknown params: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ModDefinition:
    state_factories: Mapping[str, StateFactory] = field(default_factory=dict)
    transition_plugins: Mapping[str, type[TransitionPlugin]] = field(
        default_factory=dict
    )


class _ResourceRegistry(Protocol):
    def register(
        self,
        key: ResourceKey[ResourceT],
        *,
        owner: str,
        root: Path,
        factory: ResourceFactory[ResourceT],
        policy: ResourcePolicy,
    ) -> None:
        ...

    def handle(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        ...


class ModLoadContext:
    def __init__(
        self,
        mod_id: str,
        mod_root: Path,
        resources: _ResourceRegistry,
    ) -> None:
        self.mod_id = mod_id
        self.mod_root = mod_root
        self.resources = resources

    def register_resource(
        self,
        key: ResourceKey[ResourceT],
        factory: ResourceFactory[ResourceT],
        *,
        policy: ResourcePolicy = "startup",
    ) -> None:
        self.resources.register(
            key,
            owner=self.mod_id,
            root=self.mod_root,
            factory=factory,
            policy=policy,
        )

    def resource(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        return self.resources.handle(key)


def _typed_dataclass_value(
    state_name: str,
    parameter_name: str,
    value: object,
    annotation: object,
) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        allows_none = type(None) in arguments
        candidates = tuple(item for item in arguments if item is not type(None))
        if value is None and allows_none:
            return None
        if len(candidates) == 1:
            return _typed_dataclass_value(
                state_name,
                parameter_name,
                value,
                candidates[0],
            )
        raise TypeError(
            f"state '{state_name}' param '{parameter_name}' uses unsupported "
            f"union annotation {annotation!r}"
        )

    expected = annotation
    valid = False
    converted = value
    expected_name = getattr(expected, "__name__", repr(expected))
    if expected is float:
        valid = not isinstance(value, bool) and isinstance(value, (int, float))
        if valid:
            converted = float(value)
        expected_name = "a number"
    elif expected is int:
        valid = not isinstance(value, bool) and isinstance(value, int)
        expected_name = "an integer"
    elif expected is bool:
        valid = isinstance(value, bool)
        expected_name = "a boolean"
    elif expected is str:
        valid = isinstance(value, str)
        expected_name = "a string"
    elif isinstance(expected, type):
        valid = isinstance(value, expected)
    else:
        raise TypeError(
            f"state '{state_name}' param '{parameter_name}' uses unsupported "
            f"annotation {annotation!r}"
        )
    if not valid:
        raise ValueError(
            f"state '{state_name}' param '{parameter_name}' must be {expected_name}"
        )
    return converted


__all__ = [
    "ModDefinition",
    "ModLoadContext",
    "StateBuildContext",
    "StateFactory",
]
