"""Optional RKNN Lite backend; importing the framework never requires RKNN."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import BackendAvailability, BackendFactory, InferenceBackend, TensorMap
from .rknn_builder import prepare_rknn_artifact
from ..model import ModelArtifact, ModelSpec, RknnArtifact


def _rockchip_compatible() -> str:
    for path in (
        Path("/proc/device-tree/compatible"),
        Path("/sys/firmware/devicetree/base/compatible"),
    ):
        try:
            return (
                path.read_bytes().replace(b"\x00", b",").decode(errors="ignore").lower()
            )
        except OSError:
            continue
    return ""


class RknnBackend(InferenceBackend):
    backend_name = "rknn"

    def __init__(self, artifact: RknnArtifact, spec: ModelSpec) -> None:
        from rknnlite.api import RKNNLite

        if not spec.input_names or not spec.output_names:
            raise ValueError("RKNN models require explicit logical input/output order")

        self._runtime = RKNNLite()
        result = self._runtime.load_rknn(str(artifact.resolved_path))
        if result != 0:
            raise RuntimeError(f"RKNN load_rknn failed with code {result}")

        init_kwargs = {}
        if artifact.core_mask is not None:
            init_kwargs["core_mask"] = artifact.core_mask
        result = self._runtime.init_runtime(**init_kwargs)
        if result != 0:
            self._runtime.release()
            raise RuntimeError(f"RKNN init_runtime failed with code {result}")

        self.input_names = spec.input_names
        self.output_names = spec.output_names
        self._input_shapes = dict(artifact.input_shapes)
        self._output_shapes = dict(artifact.output_shapes)
        self._metadata = dict(artifact.metadata)
        self._ordered_inputs = [None] * len(self.input_names)
        self._outputs: dict[str, NDArray[np.generic]] = {}

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    def input_shape(self, name: str) -> tuple[object, ...]:
        try:
            return self._input_shapes[name]
        except KeyError as exc:
            raise ValueError(
                f"RKNN artifact does not declare input shape for '{name}'"
            ) from exc

    def output_shape(self, name: str) -> tuple[object, ...]:
        try:
            return self._output_shapes[name]
        except KeyError as exc:
            raise ValueError(
                f"RKNN artifact does not declare output shape for '{name}'"
            ) from exc

    def run(self, inputs: TensorMap) -> Mapping[str, NDArray[np.generic]]:
        for index, name in enumerate(self.input_names):
            self._ordered_inputs[index] = inputs[name]
        values = self._runtime.inference(inputs=self._ordered_inputs)
        if len(values) < len(self.output_names):
            raise RuntimeError(
                f"RKNN returned {len(values)} outputs, expected {len(self.output_names)}"
            )
        for name, value in zip(self.output_names, values):
            self._outputs[name] = np.asarray(value)
        return self._outputs

    def close(self) -> None:
        runtime = self._runtime
        self._runtime = None
        self._outputs.clear()
        if runtime is not None:
            runtime.release()


class RknnBackendFactory(BackendFactory):
    backend_name = "rknn"

    def availability(self, artifact: ModelArtifact) -> BackendAvailability:
        if not isinstance(artifact, RknnArtifact):
            return BackendAvailability(False, "artifact is not an RKNN model")
        preparation = prepare_rknn_artifact(artifact)
        if not preparation.ready:
            return BackendAvailability(
                False,
                preparation.reason,
                preparation.install_hint,
            )
        if importlib.util.find_spec("rknnlite") is None:
            return BackendAvailability(
                False,
                "rknnlite is not installed",
                "install the rknn-toolkit-lite2 wheel matching your Python "
                "version and Rockchip SoC from the official RKNN Toolkit2 release",
            )
        compatible = _rockchip_compatible()
        target = preparation.target or artifact.target
        if target and target.lower() not in compatible:
            return BackendAvailability(
                False,
                f"platform is not compatible with RKNN target {target}",
            )
        return BackendAvailability(True, "RKNN Lite and compatible model are available")

    def open(self, artifact: ModelArtifact, spec: ModelSpec) -> InferenceBackend:
        if not isinstance(artifact, RknnArtifact):
            raise TypeError("RKNN backend requires RknnArtifact")
        return RknnBackend(artifact, spec)


__all__ = ["RknnBackend", "RknnBackendFactory"]
