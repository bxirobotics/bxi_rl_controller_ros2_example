from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import copy
from typing import (
    ClassVar,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

from .context import RobotControlContext
from .frame import FloatArray, MotorFrame


TransitionConfig: TypeAlias = Mapping[str, object]
TransitionSpec: TypeAlias = str | TransitionConfig | None
LiteralT = TypeVar("LiteralT", bound=str)


class TransitionCapabilityError(ValueError):
    pass


@runtime_checkable
class EntryFrameProvider(Protocol):
    """State capability for transitions that need a stable entry target."""

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        ...


@runtime_checkable
class RunningFrameProvider(Protocol):
    """State capability for transitions that sample live motor output.

    ``advance=False`` is a preview: it may overwrite caller-visible output
    buffers, but must not change time, history, previous actions, or any other
    state that can affect a later sample.
    """

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
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

    def update(self, ctx: RobotControlContext, dt: float) -> bool:
        ...


class TransitionPlan(Protocol):
    name: str
    type_name: str

    @property
    def duration(self) -> float:
        ...

    def validate_states(self, from_state: object, to_state: object) -> None:
        ...

    def create_session(
        self,
        ctx: RobotControlContext,
        from_state: object,
        to_state: object,
    ) -> TransitionSession:
        ...

    def snapshot(self) -> dict[str, object]:
        ...


class TransitionPlugin(ABC):
    type_name: ClassVar[str]

    @classmethod
    @abstractmethod
    def compile(cls, name: str, raw: Mapping[str, object]) -> TransitionPlan:
        ...


def normalized_progress(elapsed: float, duration: float) -> float:
    if duration <= 0.0:
        return 1.0
    return min(max(elapsed / duration, 0.0), 1.0)


class SingleClassTransition(TransitionPlugin):
    """Base for a self-contained transition plan and runtime session."""

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

    def validate_states(self, from_state: object, to_state: object) -> None:
        pass

    def create_session(
        self,
        ctx: RobotControlContext,
        from_state: object,
        to_state: object,
    ) -> TransitionSession:
        session = copy.copy(self)
        session._elapsed = 0.0
        session.on_start(ctx, from_state, to_state)
        return session

    def on_start(
        self,
        ctx: RobotControlContext,
        from_state: object,
        to_state: object,
    ) -> None:
        pass

    def update(self, ctx: RobotControlContext, dt: float) -> bool:
        self._elapsed += dt
        self.apply(ctx, dt, self.progress)
        return self.progress >= 1.0

    @abstractmethod
    def apply(
        self,
        ctx: RobotControlContext,
        dt: float,
        progress: float,
    ) -> None:
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
    "TransitionSession",
    "TransitionSpec",
    "normalized_progress",
    "require_entry_frame_provider",
    "require_running_frame_provider",
]
