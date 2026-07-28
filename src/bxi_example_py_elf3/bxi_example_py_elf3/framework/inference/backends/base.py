"""Small backend interface shared by ONNX Runtime, RKNN and future engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..model import ModelArtifact, ModelSpec


TensorMap = Mapping[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    available: bool
    reason: str
    install_hint: str | None = None


class InferenceBackend(ABC):
    """A loaded model with a stable logical input/output interface."""

    backend_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    @property
    def metadata(self) -> Mapping[str, str]:
        return {}

    def input_shape(self, name: str) -> tuple[object, ...]:
        raise NotImplementedError(f"{self.backend_name} does not expose input shapes")

    def output_shape(self, name: str) -> tuple[object, ...]:
        raise NotImplementedError(f"{self.backend_name} does not expose output shapes")

    @abstractmethod
    def run(self, inputs: TensorMap) -> Mapping[str, NDArray[np.generic]]:
        ...

    def warmup(self, inputs: TensorMap, runs: int = 1) -> None:
        for _ in range(max(0, int(runs))):
            self.run(inputs)

    def close(self) -> None:
        pass


class BackendFactory(ABC):
    backend_name: str

    @abstractmethod
    def availability(self, artifact: ModelArtifact) -> BackendAvailability:
        ...

    @abstractmethod
    def open(self, artifact: ModelArtifact, spec: ModelSpec) -> InferenceBackend:
        ...


__all__ = [
    "BackendAvailability",
    "BackendFactory",
    "InferenceBackend",
    "TensorMap",
]
