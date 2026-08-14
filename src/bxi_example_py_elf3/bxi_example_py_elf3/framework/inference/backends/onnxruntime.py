"""ONNX Runtime backend with lazy dependency loading."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util

import numpy as np
from numpy.typing import NDArray

from .base import BackendAvailability, BackendFactory, InferenceBackend, TensorMap
from ..model import ModelArtifact, ModelSpec, OnnxArtifact


class OnnxBackend(InferenceBackend):
    backend_name = "onnxruntime"

    def __init__(
        self,
        artifact: OnnxArtifact,
        spec: ModelSpec,
        *,
        model_source: str | bytes | None = None,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = list(artifact.providers) if artifact.providers else None
        if providers is None:
            available = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )

        self._session = ort.InferenceSession(
            str(artifact.resolved_path) if model_source is None else model_source,
            providers=providers,
            sess_options=options,
        )
        session_inputs = self._session.get_inputs()
        session_outputs = self._session.get_outputs()
        actual_inputs = {item.name for item in session_inputs}
        actual_outputs = {item.name for item in session_outputs}
        missing_inputs = set(spec.input_names) - actual_inputs
        missing_outputs = set(spec.output_names) - actual_outputs
        if missing_inputs or missing_outputs:
            raise ValueError(
                "model IO mismatch: "
                f"missing inputs={sorted(missing_inputs)}, "
                f"missing outputs={sorted(missing_outputs)}"
            )
        self.input_names = spec.input_names or tuple(
            item.name for item in session_inputs
        )
        self.output_names = spec.output_names or tuple(
            item.name for item in session_outputs
        )
        self._input_shapes = {item.name: tuple(item.shape) for item in session_inputs}
        self._output_shapes = {item.name: tuple(item.shape) for item in session_outputs}
        self._outputs: dict[str, NDArray[np.generic]] = {}
        self._io_binding = None
        self._bound_inputs: tuple[NDArray[np.generic], ...] = ()
        self._output_ortvalues = []
        self._binding_failed = False

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._session.get_modelmeta().custom_metadata_map

    def input_shape(self, name: str) -> tuple[object, ...]:
        return self._input_shapes[name]

    def output_shape(self, name: str) -> tuple[object, ...]:
        return self._output_shapes[name]

    def run(self, inputs: TensorMap) -> Mapping[str, NDArray[np.generic]]:
        if self._io_binding is not None and self._inputs_are_bound(inputs):
            self._io_binding.synchronize_inputs()
            self._session.run_with_iobinding(self._io_binding)
            self._io_binding.synchronize_outputs()
            return self._outputs

        values = self._session.run(list(self.output_names), dict(inputs))
        for name, value in zip(self.output_names, values):
            self._outputs[name] = value
        if not self._binding_failed:
            try:
                self._prepare_io_binding(inputs)
            except Exception:
                # Some execution providers or unusual tensor layouts cannot
                # bind caller-owned CPU buffers. Correct inference remains
                # available through session.run; do not retry every cycle.
                self._binding_failed = True
                self._io_binding = None
                self._output_ortvalues.clear()
        return self._outputs

    def _inputs_are_bound(self, inputs: TensorMap) -> bool:
        for index, name in enumerate(self.input_names):
            if inputs[name] is not self._bound_inputs[index]:
                return False
        return True

    def _prepare_io_binding(self, inputs: TensorMap) -> None:
        import onnxruntime as ort

        binding = self._session.io_binding()
        for name in self.input_names:
            value = inputs[name]
            if not value.flags.c_contiguous:
                raise ValueError(f"input '{name}' must be C-contiguous for IO binding")
            binding.bind_cpu_input(name, value)

        output_ortvalues = []
        for name in self.output_names:
            value = self._outputs[name]
            if not value.flags.c_contiguous:
                raise ValueError(f"output '{name}' must be C-contiguous for IO binding")
            ort_value = ort.OrtValue.ortvalue_from_numpy(value)
            binding.bind_ortvalue_output(name, ort_value)
            output_ortvalues.append(ort_value)

        self._io_binding = binding
        self._bound_inputs = tuple(inputs[name] for name in self.input_names)
        self._output_ortvalues = output_ortvalues

    def close(self) -> None:
        self._outputs.clear()
        self._io_binding = None
        self._bound_inputs = ()
        self._output_ortvalues.clear()
        self._session = None


class OnnxBackendFactory(BackendFactory):
    backend_name = "onnxruntime"

    def availability(self, artifact: ModelArtifact) -> BackendAvailability:
        if not isinstance(artifact, OnnxArtifact):
            return BackendAvailability(False, "artifact is not an ONNX model")
        if importlib.util.find_spec("onnxruntime") is None:
            return BackendAvailability(
                False,
                "onnxruntime is not installed",
                "python3 -m pip install onnxruntime",
            )
        if not artifact.resolved_path.is_file():
            return BackendAvailability(False, f"model does not exist: {artifact.path}")
        return BackendAvailability(True, "ONNX Runtime and model are available")

    def open(self, artifact: ModelArtifact, spec: ModelSpec) -> InferenceBackend:
        if not isinstance(artifact, OnnxArtifact):
            raise TypeError("ONNX backend requires OnnxArtifact")
        return OnnxBackend(artifact, spec)


__all__ = ["OnnxBackend", "OnnxBackendFactory"]
