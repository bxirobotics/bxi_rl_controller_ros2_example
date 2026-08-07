"""Backend registration, automatic selection and shared performance policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
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


class _LoggerLike(Protocol):
    def info(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...


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
        logger: _LoggerLike | None = None,
    ) -> None:
        self.registry = registry or BackendRegistry(
            (RknnBackendFactory(), OpenVinoBackendFactory(), OnnxBackendFactory())
        )
        self.options = options or RuntimeOptions()
        self._logger = logger
        self._warned_fallbacks: set[tuple[str, tuple[str, ...], str]] = set()

    def set_logger(self, logger: _LoggerLike | None) -> None:
        """Attach the host logger before model resources begin loading."""

        self._logger = logger

    @staticmethod
    def _model_path(spec: ModelSpec) -> Path:
        for artifact in spec.artifacts:
            source = getattr(artifact, "source_onnx", None)
            if source is not None:
                return Path(source).expanduser().resolve()
        for artifact in spec.artifacts:
            path = artifact.resolved_path
            if path.suffix.lower() == ".onnx":
                return path
        return spec.artifacts[0].resolved_path

    def _log_selection(
        self,
        *,
        spec: ModelSpec,
        requested: str,
        selected: InferenceBackend,
        artifact_path: Path,
        skipped: list[str],
    ) -> None:
        model_path = self._model_path(spec)
        prefix = (
            "inference backend selected: "
            f"model={model_path.name}, model_path={model_path}, "
            f"requested={requested}, selected_backend={selected.backend_name}, "
            f"artifact={artifact_path}"
        )
        logger = self._logger
        if requested == "auto" and skipped and self.options.warn_on_fallback:
            message = (
                prefix
                + "; fallback_reasons="
                + " | ".join(skipped)
                + f"; selected {selected.backend_name}"
            )
            warning_key = (str(model_path), tuple(skipped), selected.backend_name)
            if warning_key in self._warned_fallbacks:
                return
            self._warned_fallbacks.add(warning_key)
            if logger is not None:
                logger.warning(message)
            else:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
            return
        if logger is not None:
            logger.info(prefix)

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

            self._log_selection(
                spec=spec,
                requested=requested,
                selected=selected,
                artifact_path=artifact.resolved_path,
                skipped=skipped,
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
