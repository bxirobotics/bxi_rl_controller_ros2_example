"""Internal resource registration, lazy creation and deterministic shutdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

from bxi_example_py_elf3.mod_api.resource import (
    ResourceFactory,
    ResourceHandle,
    ResourceKey,
    ResourceLoading,
    ResourceLoadContext,
)


ResourceT = TypeVar("ResourceT")


@dataclass
class _ResourceProvider(Generic[ResourceT]):
    key: ResourceKey[ResourceT]
    owner: str
    root: Path
    factory: ResourceFactory[ResourceT]
    loading: ResourceLoading
    instance: ResourceT | None = None


class ResourceManager:
    """Creates resources by policy, caches them and closes them deterministically."""

    def __init__(self) -> None:
        self._providers: dict[str, _ResourceProvider[object]] = {}

    def register(
        self,
        key: ResourceKey[ResourceT],
        *,
        owner: str,
        root: Path,
        factory: ResourceFactory[ResourceT],
        loading: ResourceLoading,
    ) -> None:
        if loading not in ("lazy", "eager"):
            raise ValueError(f"resource '{key.id}' loading must be 'lazy' or 'eager'")
        previous = self._providers.get(key.id)
        if previous is not None:
            raise ValueError(
                f"duplicate resource '{key.id}' from '{previous.owner}' and '{owner}'"
            )
        provider = _ResourceProvider(key, owner, root, factory, loading)
        self._providers[key.id] = cast(_ResourceProvider[object], provider)

    def handle(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        if key.id not in self._providers:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        return ResourceHandle(self, key)

    def preload_eager(self) -> None:
        for provider in self._providers.values():
            if provider.loading == "eager":
                self._get_provider(provider)

    def get(self, key: ResourceKey[ResourceT]) -> ResourceT:
        provider = self._providers.get(key.id)
        if provider is None:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        return cast(ResourceT, self._get_provider(provider))

    @staticmethod
    def _get_provider(provider: _ResourceProvider[object]) -> object:
        if provider.instance is None:
            context = ResourceLoadContext(
                mod_id=provider.owner,
                mod_root=provider.root,
            )
            provider.instance = provider.factory(context)
        return provider.instance

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
