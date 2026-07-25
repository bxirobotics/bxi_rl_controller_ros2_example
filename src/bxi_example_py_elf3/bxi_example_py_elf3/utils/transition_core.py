from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import importlib
import pkgutil
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
    from bxi_example_py_elf3.utils.state_machine import StateBehavior


# Motor output -----------------------------------------------------------------

FloatArray: TypeAlias = NDArray[np.float32]
TransitionConfig: TypeAlias = Mapping[str, object]
TransitionSpec: TypeAlias = str | TransitionConfig | None


@dataclass(frozen=True)
class MotorFrame:
    """A complete motor command with owned float32 arrays."""

    qpos: FloatArray
    kp: FloatArray
    kd: FloatArray

    @classmethod
    def create(cls, qpos: object, kp: object, kd: object) -> "MotorFrame":
        arrays = tuple(
            np.asarray(value, dtype=np.float32).copy() for value in (qpos, kp, kd)
        )
        qpos_array, kp_array, kd_array = arrays
        if qpos_array.shape != kp_array.shape or qpos_array.shape != kd_array.shape:
            raise ValueError(
                "motor frame shapes must match: "
                f"qpos={qpos_array.shape}, kp={kp_array.shape}, kd={kd_array.shape}"
            )
        return cls(qpos=qpos_array, kp=kp_array, kd=kd_array)

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.qpos
        yield self.kp
        yield self.kd


# Optional state capabilities --------------------------------------------------


class TransitionCapabilityError(ValueError):
    pass


@runtime_checkable
class EntryFrameProvider(Protocol):
    """Optional capability for transitions that need a stable entry target."""

    def get_entry_frame(self, ctx: "BxiExample") -> MotorFrame:
        ...


@runtime_checkable
class RunningFrameProvider(Protocol):
    """Optional capability for transitions that sample a live state output."""

    def sample_running_frame(
        self,
        ctx: "BxiExample",
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame | None:
        ...


def require_entry_frame_provider(state: object) -> EntryFrameProvider:
    if not isinstance(state, EntryFrameProvider):
        name = getattr(state, "name", state.__class__.__name__)
        raise TransitionCapabilityError(
            f"state '{name}' must implement EntryFrameProvider"
        )
    return cast(EntryFrameProvider, state)


def require_running_frame_provider(state: object) -> RunningFrameProvider:
    if not isinstance(state, RunningFrameProvider):
        name = getattr(state, "name", state.__class__.__name__)
        raise TransitionCapabilityError(
            f"state '{name}' must implement RunningFrameProvider"
        )
    return cast(RunningFrameProvider, state)


# Typed plugin configuration ---------------------------------------------------

LiteralT = TypeVar("LiteralT", bound=str)


class ConfigReader:
    def __init__(self, raw: Mapping[str, object], context: str):
        self._raw = raw
        self._context = context
        self._used: set[str] = {"type"}

    def float(
        self,
        key: str,
        *,
        default: float | None = None,
        minimum: float | None = None,
    ) -> float:
        self._used.add(key)
        value = self._raw.get(key, default)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{self._context}.{key} must be a number")
        result = float(value)
        if minimum is not None and result < minimum:
            raise ValueError(f"{self._context}.{key} must be >= {minimum}")
        return result

    def boolean(self, key: str, *, default: bool) -> bool:
        self._used.add(key)
        value = self._raw.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{self._context}.{key} must be a bool")
        return value

    def literal(
        self,
        key: str,
        allowed: tuple[LiteralT, ...],
        *,
        default: LiteralT,
    ) -> LiteralT:
        self._used.add(key)
        value = self._raw.get(key, default)
        if not isinstance(value, str) or value not in allowed:
            options = ", ".join(allowed)
            raise ValueError(f"{self._context}.{key} must be one of: {options}")
        return cast(LiteralT, value)

    def mappings(self, key: str) -> list[Mapping[str, object]]:
        self._used.add(key)
        value = self._raw.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{self._context}.{key} must be a list")
        result: list[Mapping[str, object]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{self._context}.{key}[{index}] must be a map")
            result.append(item)
        return result

    def finish(self) -> None:
        unknown = sorted(set(self._raw) - self._used)
        if unknown:
            raise ValueError(
                f"{self._context} has unknown fields: {', '.join(unknown)}"
            )


# Plugin contracts and one-class implementation -------------------------------


class TransitionSession(Protocol):
    @property
    def duration(self) -> float:
        ...

    @property
    def elapsed(self) -> float:
        ...

    @property
    def progress(self) -> float:
        ...

    def update(self, ctx: "BxiExample", dt: float) -> bool:
        ...


class TransitionPlan(Protocol):
    name: str
    type_name: str

    @property
    def duration(self) -> float:
        ...

    def validate_states(
        self,
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> None:
        ...

    def create_session(
        self,
        ctx: "BxiExample",
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> TransitionSession:
        ...

    def snapshot(self) -> dict[str, object]:
        ...


class TransitionPlugin(ABC):
    type_name: ClassVar[str]

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: object) -> None:
        super().__init_subclass__()
        if not abstract:
            _register_plugin(cls)

    @classmethod
    @abstractmethod
    def compile(cls, name: str, raw: Mapping[str, object]) -> TransitionPlan:
        ...


def normalized_progress(elapsed: float, duration: float) -> float:
    if duration <= 0.0:
        return 1.0
    return min(max(elapsed / duration, 0.0), 1.0)


class SingleClassTransition(TransitionPlugin, abstract=True):
    """Base for a self-contained, automatically discovered transition class."""

    def __init__(self, name: str, duration: float):
        if duration < 0.0:
            raise ValueError(f"{name}.duration must be >= 0")
        self.name = name
        self._duration = float(duration)
        self._elapsed = 0.0

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        name: str,
        raw: Mapping[str, object],
    ) -> "SingleClassTransition":
        ...

    @classmethod
    def compile(cls, name: str, raw: Mapping[str, object]) -> TransitionPlan:
        return cls.from_config(name, raw)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def progress(self) -> float:
        return normalized_progress(self.elapsed, self.duration)

    def validate_states(
        self,
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> None:
        pass

    def create_session(
        self,
        ctx: "BxiExample",
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> TransitionSession:
        session = copy.copy(self)
        session._elapsed = 0.0
        session.on_start(ctx, from_state, to_state)
        return session

    def on_start(
        self,
        ctx: "BxiExample",
        from_state: "StateBehavior[BxiExample]",
        to_state: "StateBehavior[BxiExample]",
    ) -> None:
        pass

    def update(self, ctx: "BxiExample", dt: float) -> bool:
        self._elapsed += dt
        self.apply(ctx, dt, self.progress)
        return self.progress >= 1.0

    @abstractmethod
    def apply(self, ctx: "BxiExample", dt: float, progress: float) -> None:
        ...

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.type_name,
            "duration": self.duration,
            **self.config_snapshot(),
        }

    def config_snapshot(self) -> dict[str, object]:
        return {}


# Automatic discovery ----------------------------------------------------------

_plugins: dict[str, type[TransitionPlugin]] = {}
_discovered = False
TransitionPluginSnapshot: TypeAlias = dict[str, type[TransitionPlugin]]


def _register_plugin(plugin: type[TransitionPlugin]) -> None:
    type_name = getattr(plugin, "type_name", "")
    if not type_name:
        raise TypeError(f"{plugin.__name__} must define type_name")
    existing = _plugins.get(type_name)
    if existing is not None and existing is not plugin:
        raise TypeError(
            f"duplicate transition type '{type_name}': "
            f"{existing.__name__} and {plugin.__name__}"
        )
    _plugins[type_name] = plugin


def discover_plugins() -> None:
    global _discovered
    if _discovered:
        return
    package = importlib.import_module("bxi_example_py_elf3.transitions")
    package_path = getattr(package, "__path__", ())
    prefix = f"{package.__name__}."
    for module in pkgutil.iter_modules(package_path, prefix):
        importlib.import_module(module.name)
    _discovered = True


def snapshot_transition_plugins() -> TransitionPluginSnapshot:
    return dict(_plugins)


def restore_transition_plugins(snapshot: TransitionPluginSnapshot) -> None:
    _plugins.clear()
    _plugins.update(snapshot)


def release_transition_plugins(
    module_prefixes: Sequence[str],
) -> None:
    for type_name, plugin in tuple(_plugins.items()):
        if not any(
            plugin.__module__ == prefix or plugin.__module__.startswith(f"{prefix}.")
            for prefix in module_prefixes
        ):
            continue
        _plugins.pop(type_name, None)


def compile_transition(name: str, raw: Mapping[str, object]) -> TransitionPlan:
    discover_plugins()
    type_name = raw.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise ValueError(f"transition profile '{name}' must define string field 'type'")
    plugin = _plugins.get(type_name)
    if plugin is None:
        available = ", ".join(sorted(_plugins)) or "<none>"
        raise ValueError(
            f"unknown transition type '{type_name}' in profile '{name}'; "
            f"available: {available}"
        )
    return plugin.compile(name, raw)


__all__ = [
    "ConfigReader",
    "EntryFrameProvider",
    "FloatArray",
    "MotorFrame",
    "RunningFrameProvider",
    "SingleClassTransition",
    "TransitionCapabilityError",
    "TransitionConfig",
    "TransitionPlan",
    "TransitionPlugin",
    "TransitionPluginSnapshot",
    "TransitionSession",
    "TransitionSpec",
    "compile_transition",
    "discover_plugins",
    "normalized_progress",
    "release_transition_plugins",
    "require_entry_frame_provider",
    "require_running_frame_provider",
    "restore_transition_plugins",
    "snapshot_transition_plugins",
]
