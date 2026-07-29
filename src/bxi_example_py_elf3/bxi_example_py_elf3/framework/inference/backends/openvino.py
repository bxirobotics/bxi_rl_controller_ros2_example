"""Optional OpenVINO backend with persistent requests and shared input memory."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import warnings

import numpy as np
from numpy.typing import NDArray

from .base import BackendAvailability, BackendFactory, InferenceBackend, TensorMap
from ..model import ModelArtifact, ModelSpec, OpenVinoArtifact


def _openvino_api():
    import openvino as ov

    core_type = getattr(ov, "Core", None)
    tensor_type = getattr(ov, "Tensor", None)
    if core_type is not None and tensor_type is not None:
        return core_type, tensor_type

    # OpenVINO releases before 2023 exposed the runtime API below this module.
    from openvino.runtime import Core, Tensor

    return Core, Tensor


def _port_name(port) -> str:
    try:
        return port.get_any_name()
    except RuntimeError:
        names = tuple(port.get_names())
        if not names:
            raise ValueError("OpenVINO model contains a tensor without a name")
        return names[0]


def _port_shape(port) -> tuple[object, ...]:
    try:
        shape = port.shape
    except RuntimeError:
        shape = port.partial_shape
    return tuple(_dimension_value(dimension) for dimension in shape)


def _dimension_value(dimension) -> object:
    try:
        return int(dimension)
    except (TypeError, ValueError):
        is_static = getattr(dimension, "is_static", False)
        if callable(is_static):
            is_static = is_static()
        if is_static:
            get_length = getattr(dimension, "get_length", None)
            if callable(get_length):
                return int(get_length())
        return str(dimension)


def _device_name(core, device: str) -> str:
    try:
        return str(core.get_property(device, "FULL_DEVICE_NAME"))
    except Exception:
        return device


def _unsupported_gpu_devices(core) -> dict[str, str]:
    """Return OpenCL GPUs enumerated by Intel's plugin that it cannot run."""

    try:
        devices = tuple(core.available_devices)
    except Exception:
        return {}
    unsupported = {}
    for device in devices:
        if device.split(".", 1)[0].upper() != "GPU":
            continue
        full_name = _device_name(core, device)
        if "intel" not in full_name.lower():
            unsupported[device] = full_name
    return unsupported


def _safe_device(core, requested: str) -> str:
    unsupported = _unsupported_gpu_devices(core)
    if not unsupported:
        return requested

    normalized = requested.upper()
    if normalized == "AUTO" or normalized.startswith("AUTO:"):
        try:
            supported = [
                device for device in core.available_devices if device not in unsupported
            ]
        except Exception:
            supported = []
        if not supported:
            names = ", ".join(unsupported.values())
            raise RuntimeError(
                f"OpenVINO found only unsupported non-Intel GPU devices: {names}"
            )
        selected = "CPU" if "CPU" in supported else supported[0]
        names = ", ".join(unsupported.values())
        warnings.warn(
            f"OpenVINO AUTO ignored unsupported non-Intel GPU device(s) "
            f"{names}; selected {selected}",
            RuntimeWarning,
            stacklevel=3,
        )
        return selected

    requested_base = requested.split(".", 1)[0].upper()
    if requested_base == "GPU":
        matched = next(
            (
                name
                for device, name in unsupported.items()
                if device.upper() == normalized
            ),
            None,
        )
        if matched is None and normalized == "GPU" and len(unsupported) == 1:
            matched = next(iter(unsupported.values()))
        if matched is not None:
            raise RuntimeError(
                f"OpenVINO Intel GPU plugin cannot execute on {matched}; "
                "use ONNX Runtime CUDA/TensorRT for NVIDIA GPUs"
            )
    return requested


def _onnx_metadata(path) -> dict[str, str]:
    if path.suffix.lower() != ".onnx" or importlib.util.find_spec("onnx") is None:
        return {}
    import onnx

    model = onnx.load_model(str(path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


class OpenVinoBackend(InferenceBackend):
    backend_name = "openvino"

    def __init__(self, artifact: OpenVinoArtifact, spec: ModelSpec) -> None:
        Core, Tensor = _openvino_api()
        self._tensor_type = Tensor
        self._core = Core()
        model = self._core.read_model(str(artifact.resolved_path))
        device = _safe_device(self._core, artifact.device)
        self._compiled_model = self._core.compile_model(
            model,
            device_name=device,
            config=dict(artifact.config),
        )
        self._request = self._compiled_model.create_infer_request()

        actual_inputs = {_port_name(port): port for port in self._compiled_model.inputs}
        actual_outputs = {
            _port_name(port): port for port in self._compiled_model.outputs
        }
        self.input_names = spec.input_names or tuple(actual_inputs)
        self.output_names = spec.output_names or tuple(actual_outputs)

        missing_inputs = set(self.input_names) - actual_inputs.keys()
        missing_outputs = set(self.output_names) - actual_outputs.keys()
        if missing_inputs or missing_outputs:
            raise ValueError(
                "model IO mismatch: "
                f"missing inputs={sorted(missing_inputs)}, "
                f"missing outputs={sorted(missing_outputs)}"
            )

        self._input_shapes = {
            name: _port_shape(actual_inputs[name]) for name in self.input_names
        }
        self._output_shapes = {
            name: _port_shape(actual_outputs[name]) for name in self.output_names
        }
        self._metadata = dict(artifact.metadata)
        if not self._metadata:
            self._metadata = _onnx_metadata(artifact.resolved_path)
        self._bound_inputs: tuple[NDArray[np.generic], ...] = ()
        self._input_tensors = []
        self._outputs: dict[str, NDArray[np.generic]] = {}

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    def input_shape(self, name: str) -> tuple[object, ...]:
        return self._input_shapes[name]

    def output_shape(self, name: str) -> tuple[object, ...]:
        return self._output_shapes[name]

    def run(self, inputs: TensorMap) -> Mapping[str, NDArray[np.generic]]:
        if not self._inputs_are_bound(inputs):
            self._bind_inputs(inputs)

        self._request.infer()
        if not self._outputs:
            for name in self.output_names:
                self._outputs[name] = np.asarray(self._request.get_tensor(name).data)
        return self._outputs

    def _inputs_are_bound(self, inputs: TensorMap) -> bool:
        if len(self._bound_inputs) != len(self.input_names):
            return False
        for index, name in enumerate(self.input_names):
            if inputs[name] is not self._bound_inputs[index]:
                return False
        return True

    def _bind_inputs(self, inputs: TensorMap) -> None:
        bound_inputs = []
        input_tensors = []
        for name in self.input_names:
            value = inputs[name]
            if not value.flags.c_contiguous:
                raise ValueError(f"input '{name}' must be C-contiguous")
            tensor = self._tensor_type(value, shared_memory=True)
            self._request.set_tensor(name, tensor)
            bound_inputs.append(value)
            input_tensors.append(tensor)
        self._bound_inputs = tuple(bound_inputs)
        self._input_tensors = input_tensors

    def close(self) -> None:
        self._outputs.clear()
        self._bound_inputs = ()
        self._input_tensors.clear()
        self._request = None
        self._compiled_model = None
        self._core = None


class OpenVinoBackendFactory(BackendFactory):
    backend_name = "openvino"

    def availability(self, artifact: ModelArtifact) -> BackendAvailability:
        if not isinstance(artifact, OpenVinoArtifact):
            return BackendAvailability(False, "artifact is not an OpenVINO model")
        if importlib.util.find_spec("openvino") is None:
            return BackendAvailability(
                False,
                "openvino is not installed",
                "python3 -m pip install openvino",
            )
        if not artifact.resolved_path.is_file():
            return BackendAvailability(False, f"model does not exist: {artifact.path}")
        return BackendAvailability(True, "OpenVINO and model are available")

    def open(self, artifact: ModelArtifact, spec: ModelSpec) -> InferenceBackend:
        if not isinstance(artifact, OpenVinoArtifact):
            raise TypeError("OpenVINO backend requires OpenVinoArtifact")
        return OpenVinoBackend(artifact, spec)


__all__ = ["OpenVinoBackend", "OpenVinoBackendFactory"]
