"""Internal resource registration, lazy creation and deterministic shutdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

from bxi_example_py_elf3.mod_api.resource import (
    ResourceFactory,
    ResourceHandle,
    ResourceKey,
    ResourceLoadContext,
)


ResourceT = TypeVar("ResourceT")


@dataclass
class _ResourceProvider(Generic[ResourceT]):
    key: ResourceKey[ResourceT]
    owner: str
    root: Path
    factory: ResourceFactory[ResourceT]
    instance: ResourceT | None = None


class ResourceManager:
    """Lazily creates and caches resources for one Mod runtime."""

    def __init__(self) -> None:
        self._providers: dict[str, _ResourceProvider[object]] = {}

    def register(
        self,
        key: ResourceKey[ResourceT],
        *,
        owner: str,
        root: Path,
        factory: ResourceFactory[ResourceT],
    ) -> None:
        previous = self._providers.get(key.id)
        if previous is not None:
            raise ValueError(
                f"duplicate resource '{key.id}' from '{previous.owner}' and '{owner}'"
            )
        provider = _ResourceProvider(key, owner, root, factory)
        self._providers[key.id] = cast(_ResourceProvider[object], provider)

    def handle(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        if key.id not in self._providers:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        return ResourceHandle(self, key)

    def get(self, key: ResourceKey[ResourceT]) -> ResourceT:
        provider = self._providers.get(key.id)
        if provider is None:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        if provider.instance is None:
            context = ResourceLoadContext(
                mod_id=provider.owner,
                mod_root=provider.root,
            )
            provider.instance = provider.factory(context)
        return cast(ResourceT, provider.instance)

    def close(self) -> None:
        first_error: Exception | None = None
        for provider in reversed(tuple(self._providers.values())):
            instance = provider.instance
            if instance is None:
                continue
            close = getattr(instance, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    provider.instance = None
            else:
                provider.instance = None
        if first_error is not None:
            raise first_error


__all__ = ["ResourceManager"]
