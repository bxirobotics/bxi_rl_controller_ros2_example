from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar


ResourceT = TypeVar("ResourceT")
ResourceFactory = Callable[["ResourceLoadContext"], ResourceT]
ResourcePolicy: TypeAlias = Literal["startup", "on_demand"]
ResourceStatus: TypeAlias = Literal["unloaded", "loading", "ready", "failed"]


@dataclass(frozen=True)
class ResourceKey(Generic[ResourceT]):
    """A statically typed, globally unique resource identity."""

    id: str

    def __post_init__(self) -> None:
        if not self.id or "/" not in self.id:
            raise ValueError(f"resource id must be namespaced: {self.id!r}")


@dataclass(frozen=True)
class ResourceLoadContext:
    mod_id: str
    mod_root: Path

    def asset(self, relative_path: str) -> Path:
        path = (self.mod_root / relative_path).resolve()
        assets_root = (self.mod_root / "assets").resolve()
        if assets_root not in path.parents:
            raise ValueError(
                f"resource in '{self.mod_id}' must come from its assets folder: "
                f"{relative_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"resource asset does not exist in '{self.mod_id}': {relative_path}"
            )
        return path


class _ResourceResolver(Protocol):
    def get(self, key: ResourceKey[ResourceT]) -> ResourceT:
        ...

    def request(self, key: ResourceKey[ResourceT]) -> None:
        ...

    def status(self, key: ResourceKey[ResourceT]) -> ResourceStatus:
        ...

    def error(self, key: ResourceKey[ResourceT]) -> BaseException | None:
        ...


class ResourceHandle(Generic[ResourceT]):
    """Typed reference to a cached resource owned by the runtime."""

    def __init__(self, manager: _ResourceResolver, key: ResourceKey[ResourceT]):
        self._manager = manager
        self._key = key

    @property
    def key(self) -> ResourceKey[ResourceT]:
        return self._key

    def get(self) -> ResourceT:
        return self._manager.get(self._key)

    def request(self) -> None:
        """Request asynchronous preparation without waiting for completion."""
        self._manager.request(self._key)

    @property
    def status(self) -> ResourceStatus:
        return self._manager.status(self._key)

    @property
    def error(self) -> BaseException | None:
        return self._manager.error(self._key)


__all__ = [
    "ResourceFactory",
    "ResourceHandle",
    "ResourceKey",
    "ResourcePolicy",
    "ResourceStatus",
    "ResourceLoadContext",
]
