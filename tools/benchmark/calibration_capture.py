"""Tool-side capture of model inputs from any framework inference backend."""

from __future__ import annotations

import atexit
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import sys
import threading
from typing import Any

import numpy as np


ACTIVE_ENV = "BXI_TOOL_CALIBRATION_CAPTURE"
ROOT_ENV = "BXI_TOOL_CALIBRATION_ROOT"
EVERY_ENV = "BXI_TOOL_CALIBRATION_EVERY"
MAX_ENV = "BXI_TOOL_CALIBRATION_MAX"
SKIP_ENV = "BXI_TOOL_CALIBRATION_SKIP"
QUEUE_ENV = "BXI_TOOL_CALIBRATION_QUEUE"
_STOP = object()


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "model"


def _integer_environment(name: str, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"missing calibration capture setting: {name}")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    root: Path
    every: int
    max_samples: int
    skip_calls: int
    queue_size: int

    @classmethod
    def from_environment(cls) -> "CaptureConfig":
        root = os.environ.get(ROOT_ENV, "").strip()
        if not root:
            raise RuntimeError(f"missing calibration capture setting: {ROOT_ENV}")
        return cls(
            root=Path(root).expanduser().resolve(),
            every=_integer_environment(EVERY_ENV, minimum=1),
            max_samples=_integer_environment(MAX_ENV, minimum=1),
            skip_calls=_integer_environment(SKIP_ENV, minimum=0),
            queue_size=_integer_environment(QUEUE_ENV, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    name: str
    source: str
    sha256: str | None
    backend: str


class CalibrationDatasetWriter:
    """Copy inference inputs and serialize them outside the caller's thread."""

    def __init__(
        self,
        config: CaptureConfig,
        identity: ModelIdentity,
    ) -> None:
        self.identity = identity
        self.directory = config.root / _safe_component(identity.name)
        if any(character.isspace() for character in str(self.directory)):
            raise ValueError(
                "calibration directory must not contain whitespace: "
                f"{self.directory}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.directory / "dataset.txt"
        self.every = config.every
        self.max_samples = config.max_samples
        self.skip_calls = config.skip_calls
        self._calls = 0
        self._queued = self._existing_sample_count()
        self._written = self._queued
        self._dropped = 0
        self._error: BaseException | None = None
        self._closed = False
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_size)
        self._validate_existing_identity()
        self._thread = threading.Thread(
            target=self._write_loop,
            name=f"calibration-writer-{_safe_component(identity.name)}",
            daemon=True,
        )
        self._thread.start()
        print(
            "[calibration-tool] capturing "
            f"{identity.name} via {identity.backend}: skip={self.skip_calls}, "
            f"every={self.every}, max={self.max_samples}, "
            f"directory={self.directory}"
        )

    def capture(self, inputs: Mapping[str, Any]) -> bool:
        if self._closed:
            return False
        if self._error is not None:
            raise RuntimeError(
                f"calibration writer failed for {self.identity.name}"
            ) from self._error
        self._calls += 1
        eligible_call = self._calls - self.skip_calls
        if eligible_call <= 0:
            return False
        if (eligible_call - 1) % self.every != 0:
            return False
        if self._queued >= self.max_samples:
            return False
        if not inputs:
            raise ValueError("calibration capture requires at least one model input")

        snapshot = tuple(
            (str(name), np.array(value, order="C", copy=True))
            for name, value in inputs.items()
        )
        sample_index = self._queued
        try:
            self._queue.put_nowait((sample_index, snapshot))
        except queue.Full:
            self._dropped += 1
            return False
        self._queued += 1
        return True

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=timeout)
            except queue.Full as exc:
                raise TimeoutError(
                    f"calibration queue for {self.identity.name} did not drain"
                ) from exc
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"calibration writer for {self.identity.name} did not stop"
            )
        if self._error is not None:
            raise RuntimeError(
                f"calibration writer failed for {self.identity.name}"
            ) from self._error
        print(
            "[calibration-tool] finished "
            f"{self.identity.name}: samples={self._written}, "
            f"dropped={self._dropped}, dataset={self.dataset_path}"
        )

    def _existing_sample_count(self) -> int:
        try:
            lines = self.dataset_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return 0
        return sum(bool(line.strip()) for line in lines)

    def _validate_existing_identity(self) -> None:
        metadata_path = self.directory / "capture.json"
        if not metadata_path.is_file():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_hash = metadata.get("model_sha256")
        if (
            previous_hash
            and self.identity.sha256
            and previous_hash != self.identity.sha256
        ):
            raise ValueError(
                f"capture directory {self.directory} belongs to another model; "
                "use a new output root"
            )

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    sample_index, snapshot = item
                    self._write_sample(sample_index, snapshot)
                    self._written += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            print(
                f"[calibration-tool] writer failed for "
                f"{self.identity.name}: {exc}",
                file=sys.stderr,
            )

    def _write_sample(
        self,
        sample_index: int,
        snapshot: tuple[tuple[str, np.ndarray[Any, Any]], ...],
    ) -> None:
        self._ensure_metadata(snapshot)
        paths: list[Path] = []
        for input_index, (name, value) in enumerate(snapshot):
            if value.dtype.hasobject:
                raise TypeError(
                    f"calibration input {name!r} has object dtype"
                )
            if (
                np.issubdtype(value.dtype, np.inexact)
                and not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"calibration input {name!r} contains NaN or infinity"
                )
            filename = (
                f"{sample_index:06d}_{input_index:02d}_"
                f"{_safe_component(name)}.npy"
            )
            path = self.directory / filename
            temporary = path.with_suffix(".npy.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, value, allow_pickle=False)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            paths.append(path)

        with self.dataset_path.open("a", encoding="utf-8") as dataset:
            dataset.write(" ".join(str(path) for path in paths) + "\n")
            dataset.flush()

    def _ensure_metadata(
        self,
        snapshot: tuple[tuple[str, np.ndarray[Any, Any]], ...],
    ) -> None:
        destination = self.directory / "capture.json"
        inputs = [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in snapshot
        ]
        if destination.is_file():
            metadata = json.loads(destination.read_text(encoding="utf-8"))
            if metadata.get("inputs") != inputs:
                raise ValueError(
                    f"model input contract changed in {self.directory}; "
                    "use a new output root"
                )
            return

        payload = {
            "schema_version": 2,
            "model": self.identity.name,
            "model_source": self.identity.source,
            "model_sha256": self.identity.sha256,
            "capture_backend": self.identity.backend,
            "inputs": inputs,
        }
        temporary = destination.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class RecordingBackend:
    """Transparent tool-only proxy around one selected inference backend."""

    def __init__(self, backend: Any, writer: CalibrationDatasetWriter) -> None:
        self._backend = backend
        self._writer = writer
        self.backend_name = backend.backend_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._backend.metadata

    def input_shape(self, name: str) -> tuple[object, ...]:
        return self._backend.input_shape(name)

    def output_shape(self, name: str) -> tuple[object, ...]:
        return self._backend.output_shape(name)

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        self._writer.capture(inputs)
        return self._backend.run(inputs)

    def warmup(self, inputs: Mapping[str, Any], runs: int = 1) -> None:
        for _ in range(max(0, int(runs))):
            self.run(inputs)

    def close(self) -> None:
        self._backend.close()


class CaptureManager:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._writers: dict[tuple[str, str | None], CalibrationDatasetWriter] = {}
        self._closed = False

    def wrap(self, backend: Any, spec: Any) -> RecordingBackend:
        identity = _model_identity(spec, backend.backend_name)
        key = (identity.name, identity.sha256)
        writer = self._writers.get(key)
        if writer is None:
            writer = CalibrationDatasetWriter(self.config, identity)
            self._writers[key] = writer
        return RecordingBackend(backend, writer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            print(f"[calibration-tool] shutdown failed: {first_error}", file=sys.stderr)


def _model_identity(spec: Any, backend_name: str) -> ModelIdentity:
    artifacts = tuple(getattr(spec, "artifacts", ()))
    artifact = next(
        (item for item in artifacts if getattr(item, "backend", None) == backend_name),
        artifacts[0] if artifacts else None,
    )
    if artifact is None:
        source = backend_name
    else:
        source_value = getattr(artifact, "source_onnx", None) or getattr(
            artifact, "path", backend_name
        )
        source = str(Path(source_value).expanduser().resolve())
    source_path = Path(source)
    return ModelIdentity(
        name=source_path.stem or backend_name,
        source=source,
        sha256=_file_sha256(source_path),
        backend=backend_name,
    )


def install_from_environment() -> CaptureManager | None:
    """Patch the framework only inside a process launched by the capture tool."""
    if os.environ.get(ACTIVE_ENV) != "1":
        return None
    try:
        from bxi_example_py_elf3.framework.inference.runtime import InferenceRuntime
    except ModuleNotFoundError:
        return None

    existing = getattr(InferenceRuntime, "_bxi_capture_manager", None)
    if existing is not None:
        return existing

    manager = CaptureManager(CaptureConfig.from_environment())
    original_open_backend = InferenceRuntime.open_backend

    def open_backend(self: Any, spec: Any, *, backend: str | None = None) -> Any:
        selected = original_open_backend(self, spec, backend=backend)
        return manager.wrap(selected, spec)

    InferenceRuntime.open_backend = open_backend
    InferenceRuntime._bxi_capture_manager = manager
    atexit.register(manager.close)
    return manager


__all__ = [
    "ACTIVE_ENV",
    "CaptureConfig",
    "CaptureManager",
    "CalibrationDatasetWriter",
    "ModelIdentity",
    "RecordingBackend",
    "install_from_environment",
]
