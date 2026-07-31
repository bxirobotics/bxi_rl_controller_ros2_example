"""Two-phase Mod resource preparation and deterministic shutdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Generic, TypeVar, cast

from bxi_example_py_elf3.framework.mod_api.resource import (
    ResourceFactory,
    ResourceHandle,
    ResourceKey,
    ResourceLoadContext,
    ResourcePolicy,
    ResourceStatus,
)
from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    configure_current_thread,
    format_cpu_set,
)


ResourceT = TypeVar("ResourceT")


@dataclass
class _ResourceProvider(Generic[ResourceT]):
    key: ResourceKey[ResourceT]
    owner: str
    root: Path
    factory: ResourceFactory[ResourceT]
    policy: ResourcePolicy
    instance: ResourceT | None = None
    status: ResourceStatus = "unloaded"
    error: BaseException | None = None
    completed: Event = field(default_factory=Event)


class ResourceManager:
    """Prepare resources off the control thread and publish only ready instances."""

    def __init__(self, cpu_affinity: frozenset[int] | None = None) -> None:
        self._providers: dict[str, _ResourceProvider[object]] = {}
        self._cpu_affinity = cpu_affinity
        self._lock = Lock()
        self._requests: Queue[_ResourceProvider[object] | None] = Queue()
        self._worker_ready = Event()
        self._worker_error: BaseException | None = None
        self._startup_complete = False
        self._closed = False
        self._worker = Thread(
            target=self._run_worker,
            name="bxi-resource",
            daemon=False,
        )
        self._worker.start()
        self._worker_ready.wait()
        if self._worker_error is not None:
            self._closed = True
            affinity = (
                "inherit"
                if cpu_affinity is None
                else format_cpu_set(cpu_affinity)
            )
            raise RuntimeError(
                "cannot initialize resource worker with CPU affinity "
                f"{affinity}: {self._worker_error}"
            ) from self._worker_error

    def register(
        self,
        key: ResourceKey[ResourceT],
        *,
        owner: str,
        root: Path,
        factory: ResourceFactory[ResourceT],
        policy: ResourcePolicy,
    ) -> None:
        if policy not in ("startup", "on_demand"):
            raise ValueError(
                f"resource '{key.id}' policy must be 'startup' or 'on_demand'"
            )
        if self._closed:
            raise RuntimeError("resource manager is closed")
        if self._startup_complete and policy == "startup":
            raise RuntimeError(
                f"startup resource '{key.id}' was registered after startup loading"
            )
        previous = self._providers.get(key.id)
        if previous is not None:
            raise ValueError(
                f"duplicate resource '{key.id}' from '{previous.owner}' and '{owner}'"
            )
        provider = _ResourceProvider(key, owner, root, factory, policy)
        self._providers[key.id] = cast(_ResourceProvider[object], provider)

    def handle(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        self._provider(key)
        return ResourceHandle(self, key)

    def load_startup(self) -> None:
        providers = tuple(
            provider
            for provider in self._providers.values()
            if provider.policy == "startup"
        )
        try:
            for provider in providers:
                self._request_provider(provider)
            for provider in providers:
                provider.completed.wait()
                if provider.error is not None:
                    raise RuntimeError(
                        f"cannot prepare startup resource '{provider.key.id}': "
                        f"{provider.error}"
                    ) from provider.error
        finally:
            self._startup_complete = True

    def request(self, key: ResourceKey[ResourceT]) -> None:
        self._request_provider(self._provider(key))

    def status(self, key: ResourceKey[ResourceT]) -> ResourceStatus:
        return self._provider(key).status

    def error(self, key: ResourceKey[ResourceT]) -> BaseException | None:
        return self._provider(key).error

    def get(self, key: ResourceKey[ResourceT]) -> ResourceT:
        provider = self._provider(key)
        if provider.status != "ready" or provider.instance is None:
            if provider.status == "failed":
                raise RuntimeError(
                    f"resource '{key.id}' preparation failed: {provider.error}"
                ) from provider.error
            raise RuntimeError(
                f"resource '{key.id}' is {provider.status}; state transitions must "
                "wait for resource preparation"
            )
        return cast(ResourceT, provider.instance)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for provider in self._providers.values():
            if provider.status == "loading":
                provider.completed.wait()

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
            provider.instance = None
            provider.status = "unloaded"

        self._requests.put(None)
        self._worker.join(timeout=5.0)
        if self._worker.is_alive() and first_error is None:
            first_error = RuntimeError("resource worker did not stop within timeout")
        if first_error is not None:
            raise first_error

    def _provider(
        self,
        key: ResourceKey[ResourceT],
    ) -> _ResourceProvider[ResourceT]:
        provider = self._providers.get(key.id)
        if provider is None:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        return cast(_ResourceProvider[ResourceT], provider)

    def _request_provider(self, provider: _ResourceProvider[object]) -> None:
        if self._closed:
            raise RuntimeError("resource manager is closed")
        with self._lock:
            if provider.status != "unloaded":
                return
            provider.status = "loading"
            provider.completed.clear()
            self._requests.put(provider)

    def _run_worker(self) -> None:
        try:
            if self._cpu_affinity is not None:
                configure_current_thread(
                    self._cpu_affinity,
                    realtime_priority=0,
                )
        except BaseException as exc:
            self._worker_error = exc
        finally:
            self._worker_ready.set()
        if self._worker_error is not None:
            return

        while True:
            provider = self._requests.get()
            if provider is None:
                return
            try:
                context = ResourceLoadContext(
                    mod_id=provider.owner,
                    mod_root=provider.root,
                )
                instance = provider.factory(context)
                if instance is None:
                    raise RuntimeError("resource factory returned None")
                provider.instance = instance
                provider.status = "ready"
            except BaseException as exc:
                provider.error = exc
                provider.status = "failed"
            finally:
                provider.completed.set()


__all__ = ["ResourceManager"]
