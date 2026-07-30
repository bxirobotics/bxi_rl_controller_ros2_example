"""Non-blocking capture of representative model inputs for RKNN calibration."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import queue
import re
import sys
import threading

import numpy as np
from numpy.typing import NDArray


CALIBRATION_DIR_ENV = "BXI_RKNN_CALIBRATION_DIR"
CALIBRATION_EVERY_ENV = "BXI_RKNN_CALIBRATION_EVERY"
CALIBRATION_MAX_ENV = "BXI_RKNN_CALIBRATION_MAX"
DEFAULT_CAPTURE_EVERY = 5
DEFAULT_MAX_SAMPLES = 500
_STOP = object()


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _safe_filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "tensor"


class CalibrationDatasetRecorder:
    """Copy model inputs quickly and write RKNN calibration files off-thread."""

    def __init__(
        self,
        root: str | Path,
        model_name: str,
        *,
        every: int = DEFAULT_CAPTURE_EVERY,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        queue_size: int = 16,
    ) -> None:
        if every <= 0:
            raise ValueError("calibration capture interval must be positive")
        if max_samples <= 0:
            raise ValueError("calibration sample limit must be positive")
        if queue_size <= 0:
            raise ValueError("calibration writer queue size must be positive")

        self.model_name = _safe_filename_component(model_name)
        self.directory = Path(root).expanduser().resolve() / self.model_name
        if any(character.isspace() for character in str(self.directory)):
            raise ValueError(
                "RKNN calibration directory must not contain whitespace: "
                f"{self.directory}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.directory / "dataset.txt"
        self.every = int(every)
        self.max_samples = int(max_samples)
        self._seen = 0
        self._queued = self._existing_sample_count()
        self._written = self._queued
        self._dropped = 0
        self._error: BaseException | None = None
        self._closed = False
        self._metadata_written = (self.directory / "capture.json").is_file()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._write_loop,
            name=f"rknn-calibration-{self.model_name}",
            daemon=True,
        )
        self._thread.start()
        print(
            "[RKNN calibration] capturing "
            f"{self.model_name}: every={self.every}, max={self.max_samples}, "
            f"directory={self.directory}"
        )

    @classmethod
    def from_environment(
        cls,
        model_name: str,
    ) -> "CalibrationDatasetRecorder | None":
        root = os.environ.get(CALIBRATION_DIR_ENV)
        if root is None or not root.strip():
            return None
        return cls(
            root,
            model_name,
            every=_positive_environment_integer(
                CALIBRATION_EVERY_ENV,
                DEFAULT_CAPTURE_EVERY,
            ),
            max_samples=_positive_environment_integer(
                CALIBRATION_MAX_ENV,
                DEFAULT_MAX_SAMPLES,
            ),
        )

    @property
    def written_samples(self) -> int:
        return self._written

    @property
    def dropped_samples(self) -> int:
        return self._dropped

    @property
    def error(self) -> BaseException | None:
        return self._error

    def capture(self, inputs: Mapping[str, NDArray[np.generic]]) -> bool:
        """Queue one persistent snapshot without waiting for filesystem IO."""
        if self._closed:
            raise RuntimeError("calibration recorder is closed")
        if self._error is not None:
            return False

        self._seen += 1
        if (self._seen - 1) % self.every != 0 or self._queued >= self.max_samples:
            return False
        if not inputs:
            raise ValueError("calibration capture requires at least one model input")

        snapshot = tuple(
            (
                str(name),
                np.array(value, dtype=np.float32, order="C", copy=True),
            )
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
                    f"calibration writer queue for {self.model_name} did not drain "
                    f"in {timeout}s"
                ) from exc
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"calibration writer for {self.model_name} did not stop in {timeout}s"
            )
        if self._error is not None:
            raise RuntimeError("calibration writer failed") from self._error
        print(
            "[RKNN calibration] finished "
            f"{self.model_name}: samples={self._written}, "
            f"dropped={self._dropped}, dataset={self.dataset_path}"
        )

    def _existing_sample_count(self) -> int:
        try:
            lines = self.dataset_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return 0
        return sum(bool(line.strip()) for line in lines)

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
                f"[RKNN calibration] writer failed for {self.model_name}: {exc}",
                file=sys.stderr,
            )

    def _write_sample(
        self,
        sample_index: int,
        snapshot: tuple[tuple[str, NDArray[np.float32]], ...],
    ) -> None:
        if not self._metadata_written:
            self._write_metadata(snapshot)
            self._metadata_written = True

        paths: list[Path] = []
        for input_index, (name, value) in enumerate(snapshot):
            if not np.isfinite(value).all():
                raise ValueError(f"calibration input {name!r} contains NaN or infinity")
            filename = (
                f"{sample_index:06d}_{input_index:02d}_"
                f"{_safe_filename_component(name)}.npy"
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

    def _write_metadata(
        self,
        snapshot: tuple[tuple[str, NDArray[np.float32]], ...],
    ) -> None:
        destination = self.directory / "capture.json"
        temporary = destination.with_suffix(".json.tmp")
        payload = {
            "schema_version": 1,
            "model": self.model_name,
            "inputs": [
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in snapshot
            ],
        }
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "CALIBRATION_DIR_ENV",
    "CALIBRATION_EVERY_ENV",
    "CALIBRATION_MAX_ENV",
    "CalibrationDatasetRecorder",
]
