"""Backend registration, automatic selection and shared performance policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import warnings

from .backends import (
    BackendFactory,
    InferenceBackend,
    OnnxBackendFactory,
    OpenVinoBackendFactory,
    RknnBackendFactory,
)
from .model import ModelSpec


class BackendUnavailableError(RuntimeError):
    pass


class BackendRegistry:
    def __init__(self, factories: Iterable[BackendFactory] = ()) -> None:
        self._factories: dict[str, BackendFactory] = {}
        for factory in factories:
            self.register(factory)

    def register(self, factory: BackendFactory) -> None:
        name = factory.backend_name
        if name in self._factories:
            raise ValueError(f"inference backend already registered: {name}")
        self._factories[name] = factory

    def get(self, name: str) -> BackendFactory | None:
        return self._factories.get(name)


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    backend: str = "auto"
    warmup_runs: int = 1
    warn_on_fallback: bool = True


class InferenceRuntime:
    """Small facade shared by policies; policy math remains backend-neutral."""

    def __init__(
        self,
        *,
        registry: BackendRegistry | None = None,
        options: RuntimeOptions | None = None,
    ) -> None:
        self.registry = registry or BackendRegistry(
            (RknnBackendFactory(), OpenVinoBackendFactory(), OnnxBackendFactory())
        )
        self.options = options or RuntimeOptions()
        self._warned_fallbacks: set[tuple[tuple[str, ...], str]] = set()

    def open_backend(
        self,
        spec: ModelSpec,
        *,
        backend: str | None = None,
    ) -> InferenceBackend:
        requested = backend or self.options.backend
        errors: list[str] = []
        skipped: list[str] = []
        matched = False

        for artifact in spec.artifacts:
            if requested != "auto" and artifact.backend != requested:
                continue
            matched = True
            factory = self.registry.get(artifact.backend)
            if factory is None:
                detail = f"{artifact.backend}: backend is not registered"
                errors.append(detail)
                skipped.append(detail)
                continue
            availability = factory.availability(artifact)
            if not availability.available:
                detail = f"{artifact.backend}: {availability.reason}"
                if availability.install_hint:
                    detail += f". Install: {availability.install_hint}"
                errors.append(detail)
                skipped.append(detail)
                continue
            try:
                selected = factory.open(artifact, spec)
            except Exception as exc:
                detail = f"{artifact.backend}: initialization failed: {exc}"
                errors.append(detail)
                skipped.append(detail)
                continue

            if requested == "auto" and skipped and self.options.warn_on_fallback:
                warning_key = (tuple(skipped), selected.backend_name)
                if warning_key not in self._warned_fallbacks:
                    self._warned_fallbacks.add(warning_key)
                    warnings.warn(
                        "inference backend fallback: "
                        + "; ".join(skipped)
                        + f"; selected {selected.backend_name}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return selected

        if not matched:
            errors.append(f"no artifact matches requested backend '{requested}'")
        detail = "; ".join(errors) if errors else "no compatible artifact"
        raise BackendUnavailableError(f"cannot open inference model: {detail}")


_DEFAULT_RUNTIME: InferenceRuntime | None = None


def default_runtime() -> InferenceRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = InferenceRuntime()
    return _DEFAULT_RUNTIME


__all__ = [
    "BackendRegistry",
    "BackendUnavailableError",
    "InferenceRuntime",
    "RuntimeOptions",
    "default_runtime",
]
