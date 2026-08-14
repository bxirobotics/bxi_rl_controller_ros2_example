"""Output-routed inference assembled from an accelerator and ONNX sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import InferenceBackend, TensorMap
from .onnxruntime import OnnxBackend
from ..model import ModelSpec, OnnxArtifact, _required_onnx_inputs


class CompositeBackend(InferenceBackend):
    """Merge disjoint named outputs while retaining one logical model API."""

    def __init__(
        self,
        primary: InferenceBackend,
        sidecar: InferenceBackend,
        spec: ModelSpec,
    ) -> None:
        primary_outputs = set(primary.output_names)
        sidecar_outputs = set(sidecar.output_names)
        overlap = primary_outputs.intersection(sidecar_outputs)
        if overlap:
            raise ValueError(
                f"composite backends produce duplicate outputs: {sorted(overlap)}"
            )
        missing = set(spec.output_names) - primary_outputs - sidecar_outputs
        unexpected = primary_outputs.union(sidecar_outputs) - set(spec.output_names)
        if missing or unexpected:
            raise ValueError(
                "composite output contract mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

        self._primary = primary
        self._sidecar = sidecar
        self.backend_name = f"{primary.backend_name}+onnxruntime-sidecar"
        self.input_names = spec.input_names
        self.output_names = spec.output_names
        self.output_routes = {
            name: (
                primary.backend_name
                if name in primary_outputs
                else "onnxruntime-sidecar"
            )
            for name in self.output_names
        }
        self._output_owners = {
            name: primary if name in primary_outputs else sidecar
            for name in self.output_names
        }
        self._outputs: dict[str, NDArray[np.generic]] = {}

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._primary.metadata

    def input_shape(self, name: str) -> tuple[object, ...]:
        if name in self._primary.input_names:
            return self._primary.input_shape(name)
        return self._sidecar.input_shape(name)

    def output_shape(self, name: str) -> tuple[object, ...]:
        return self._output_owners[name].output_shape(name)

    @staticmethod
    def _selected_inputs(
        backend: InferenceBackend,
        inputs: TensorMap,
    ) -> dict[str, NDArray[np.generic]]:
        missing = set(backend.input_names) - inputs.keys()
        if missing:
            raise ValueError(
                f"missing {backend.backend_name} input(s): {sorted(missing)}"
            )
        return {name: inputs[name] for name in backend.input_names}

    def run(self, inputs: TensorMap) -> Mapping[str, NDArray[np.generic]]:
        primary_outputs = self._primary.run(
            self._selected_inputs(self._primary, inputs)
        )
        sidecar_outputs = self._sidecar.run(
            self._selected_inputs(self._sidecar, inputs)
        )
        for name in self.output_names:
            values = (
                primary_outputs
                if name in self._primary.output_names
                else sidecar_outputs
            )
            self._outputs[name] = values[name]
        return self._outputs

    def close(self) -> None:
        self._outputs.clear()
        try:
            self._primary.close()
        finally:
            self._sidecar.close()


def create_onnx_output_sidecar(
    source: str | Path,
    output_names: tuple[str, ...],
    *,
    providers: tuple[str, ...] | None = None,
) -> OnnxBackend:
    """Extract exact output branches in memory and execute them with ORT."""

    if not output_names:
        raise ValueError("ONNX sidecar needs at least one output")
    source_path = Path(source).expanduser().resolve()
    try:
        import onnx
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "output-routed RKNN models require the onnx package to extract "
            "their CPU sidecar"
        ) from exc

    model = onnx.load_model(str(source_path), load_external_data=True)
    required_inputs = _required_onnx_inputs(source_path, output_names)
    extractor = onnx.utils.Extractor(model)
    extracted = extractor.extract_model(list(required_inputs), list(output_names))
    artifact = OnnxArtifact(source_path, providers=providers)
    sidecar_spec = ModelSpec(
        artifacts=(artifact,),
        input_names=required_inputs,
        output_names=output_names,
    )
    return OnnxBackend(
        artifact,
        sidecar_spec,
        model_source=extracted.SerializeToString(),
    )


__all__ = ["CompositeBackend", "create_onnx_output_sidecar"]
