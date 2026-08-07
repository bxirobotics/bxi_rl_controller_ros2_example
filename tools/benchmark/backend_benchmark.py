#!/usr/bin/env python3
"""Discover every model and benchmark every locally usable inference backend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = ROOT / "src/bxi_example_py_elf3"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from bxi_example_py_elf3.framework.inference.backends import (  # noqa: E402
    OnnxBackendFactory,
    OpenVinoBackendFactory,
    RknnBackendFactory,
)
from bxi_example_py_elf3.framework.inference.model import (  # noqa: E402
    ModelSpec,
    OnnxArtifact,
    OpenVinoArtifact,
    RknnArtifact,
)


SCHEMA_VERSION = 1
RKNN_CONVERT_ENV = "BXI_RKNN_CONVERT_ON_LOAD"
FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[object, ...]
    dtype: np.dtype


@dataclass(frozen=True, slots=True)
class ModelInfo:
    path: Path
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    factory: object
    artifact: object
    requested_device: str | None = None
    output_names: tuple[str, ...] | None = None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _openvino_inventory() -> tuple[list[str], dict[str, str], dict[str, str]]:
    if importlib.util.find_spec("openvino") is None:
        return [], {}, {}
    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        names = {}
        unsupported = {}
        supported = []
        for device in devices:
            try:
                names[device] = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception as exc:
                names[device] = f"unavailable: {exc}"
            if (
                device.split(".", 1)[0].upper() == "GPU"
                and "intel" not in names[device].lower()
            ):
                unsupported[device] = (
                    f"{names[device]} is not supported by OpenVINO's Intel GPU plugin; "
                    "use ONNX Runtime CUDA/TensorRT"
                )
            else:
                supported.append(device)
        return supported, names, unsupported
    except Exception as exc:
        return [], {"error": str(exc)}, {}


def _onnxruntime_providers() -> list[str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _environment() -> tuple[dict[str, Any], list[str]]:
    openvino_devices, device_names, unsupported_devices = _openvino_inventory()
    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": _distribution_version("onnx"),
        "onnxruntime": _distribution_version("onnxruntime"),
        "openvino": _distribution_version("openvino"),
        "rknn_toolkit_lite2": _distribution_version("rknn-toolkit-lite2"),
        "rknn_toolkit2": _distribution_version("rknn-toolkit2"),
        "onnxruntime_providers": _onnxruntime_providers(),
        "openvino_devices": device_names,
        "openvino_unsupported_devices": unsupported_devices,
        "rknn_conversion_enabled": _rknn_conversion_enabled(),
    }
    return environment, openvino_devices


def _rknn_conversion_enabled() -> bool:
    value = os.environ.get(RKNN_CONVERT_ENV)
    return value is not None and value.strip().lower() not in FALSE_ENV_VALUES


def _discover_models(paths: list[Path]) -> list[Path]:
    roots = paths or [ROOT / "src"]
    discovered: set[Path] = set()
    for item in roots:
        path = item.expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".onnx":
            discovered.add(path)
        elif path.is_dir():
            discovered.update(candidate.resolve() for candidate in path.rglob("*.onnx"))
        else:
            print(f"warning: model path does not exist or is not ONNX: {item}")
    return sorted(discovered, key=lambda path: _relative_path(path))


def _onnx_tensor_info(value_info) -> TensorInfo:
    import onnx

    tensor_type = value_info.type.tensor_type
    shape: list[object] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param:
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
    return TensorInfo(value_info.name, tuple(shape), dtype)


def _inspect_with_onnx(path: Path) -> ModelInfo:
    import onnx

    model = onnx.load_model(str(path), load_external_data=False)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = tuple(
        _onnx_tensor_info(item)
        for item in model.graph.input
        if item.name not in initializer_names
    )
    outputs = tuple(_onnx_tensor_info(item) for item in model.graph.output)
    return ModelInfo(path, inputs, outputs)


def _ort_dtype(value: str) -> np.dtype:
    types = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int16)": np.dtype(np.int16),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(uint64)": np.dtype(np.uint64),
        "tensor(uint32)": np.dtype(np.uint32),
        "tensor(uint16)": np.dtype(np.uint16),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(bool)": np.dtype(np.bool_),
    }
    try:
        return types[value]
    except KeyError as exc:
        raise TypeError(f"unsupported ONNX Runtime tensor type: {value}") from exc


def _inspect_with_onnxruntime(path: Path) -> ModelInfo:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = tuple(
        TensorInfo(item.name, tuple(item.shape), _ort_dtype(item.type))
        for item in session.get_inputs()
    )
    outputs = tuple(
        TensorInfo(item.name, tuple(item.shape), _ort_dtype(item.type))
        for item in session.get_outputs()
    )
    return ModelInfo(path, inputs, outputs)


def _openvino_tensor_info(port) -> TensorInfo:
    try:
        name = port.get_any_name()
    except RuntimeError:
        names = tuple(port.get_names())
        if not names:
            raise ValueError("OpenVINO tensor has no name")
        name = names[0]
    shape = tuple(
        int(dimension.get_length()) if dimension.is_static else str(dimension)
        for dimension in port.partial_shape
    )
    try:
        dtype = np.dtype(port.element_type.to_dtype())
    except (AttributeError, TypeError):
        dtype = np.dtype(str(port.element_type))
    return TensorInfo(name, shape, dtype)


def _inspect_with_openvino(path: Path) -> ModelInfo:
    import openvino as ov

    model = ov.Core().read_model(str(path))
    inputs = tuple(_openvino_tensor_info(item) for item in model.inputs)
    outputs = tuple(_openvino_tensor_info(item) for item in model.outputs)
    return ModelInfo(path, inputs, outputs)


def _inspect_model(path: Path) -> ModelInfo:
    errors = []
    if importlib.util.find_spec("onnx") is not None:
        try:
            return _inspect_with_onnx(path)
        except Exception as exc:
            errors.append(f"onnx: {exc}")
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            return _inspect_with_onnxruntime(path)
        except Exception as exc:
            errors.append(f"onnxruntime: {exc}")
    if importlib.util.find_spec("openvino") is not None:
        try:
            return _inspect_with_openvino(path)
        except Exception as exc:
            errors.append(f"openvino: {exc}")
    detail = "; ".join(errors) if errors else "install onnx, onnxruntime or openvino"
    raise RuntimeError(f"cannot inspect model I/O: {detail}")


def _required_onnx_inputs(
    path: Path,
    output_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    if importlib.util.find_spec("onnx") is None:
        return None
    try:
        import onnx

        model = onnx.load_model(str(path), load_external_data=False)
    except Exception:
        return None

    graph = model.graph
    initializer_names = {item.name for item in graph.initializer}
    graph_inputs = [
        item.name for item in graph.input if item.name not in initializer_names
    ]
    producers = {}
    for node in graph.node:
        for output_name in node.output:
            if output_name:
                producers[output_name] = node

    required: set[str] = set()
    visited: set[str] = set()
    pending = list(output_names)
    while pending:
        value_name = pending.pop()
        if not value_name or value_name in visited:
            continue
        visited.add(value_name)
        if value_name in initializer_names:
            continue
        if value_name in graph_inputs:
            required.add(value_name)
            continue
        producer = producers.get(value_name)
        if producer is None:
            continue
        pending.extend(item for item in producer.input if item)
    return tuple(name for name in graph_inputs if name in required)


def _parse_shape_overrides(values: list[str]) -> dict[str, tuple[int, ...]]:
    result = {}
    for value in values:
        try:
            name, raw_shape = value.split("=", 1)
            shape = tuple(int(item) for item in raw_shape.split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape override '{value}', expected name=1,2,3"
            ) from exc
        if not name or not shape or any(dimension <= 0 for dimension in shape):
            raise argparse.ArgumentTypeError(
                f"invalid shape override '{value}', dimensions must be positive"
            )
        result[name] = shape
    return result


def _parse_input_ranges(values: list[str]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for value in values:
        try:
            name, raw_range = value.split("=", 1)
            raw_low, raw_high = raw_range.split(",", 1)
            low = float(raw_low)
            high = float(raw_high)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid input range '{value}', expected NAME=LOW,HIGH"
            ) from exc
        if not name or not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise argparse.ArgumentTypeError(
                f"invalid input range '{value}', bounds must be finite and LOW < HIGH"
            )
        if name in result:
            raise argparse.ArgumentTypeError(f"duplicate input range for '{name}'")
        result[name] = (low, high)
    return result


def _concrete_shape(
    tensor: TensorInfo,
    overrides: dict[str, tuple[int, ...]],
    dynamic_dimension: int,
) -> tuple[int, ...]:
    if tensor.name in overrides:
        return overrides[tensor.name]
    return tuple(
        int(dimension)
        if isinstance(dimension, (int, np.integer)) and dimension > 0
        else dynamic_dimension
        for dimension in tensor.shape
    )


def _make_inputs(
    model: ModelInfo,
    overrides: dict[str, tuple[int, ...]],
    dynamic_dimension: int,
    seed: int,
    input_ranges: dict[str, tuple[float, float]],
    input_npz: Path | None,
) -> dict[str, np.ndarray]:
    if input_npz is not None:
        return _load_input_npz(model, overrides, dynamic_dimension, input_npz)

    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for tensor in model.inputs:
        shape = _concrete_shape(tensor, overrides, dynamic_dimension)
        if np.issubdtype(tensor.dtype, np.floating):
            low, high = input_ranges.get(tensor.name, (-1.0, 1.0))
            value = rng.uniform(low, high, size=shape).astype(tensor.dtype)
        elif np.issubdtype(tensor.dtype, np.integer):
            if tensor.name in input_ranges:
                raise ValueError(
                    f"--input-range only supports floating input '{tensor.name}'"
                )
            value = np.zeros(shape, dtype=tensor.dtype)
        elif np.issubdtype(tensor.dtype, np.bool_):
            if tensor.name in input_ranges:
                raise ValueError(
                    f"--input-range only supports floating input '{tensor.name}'"
                )
            value = np.zeros(shape, dtype=np.bool_)
        else:
            raise TypeError(f"unsupported input dtype {tensor.dtype} for {tensor.name}")
        inputs[tensor.name] = np.ascontiguousarray(value)
    return inputs


def _load_input_npz(
    model: ModelInfo,
    overrides: dict[str, tuple[int, ...]],
    dynamic_dimension: int,
    path: Path,
) -> dict[str, np.ndarray]:
    expected_names = tuple(tensor.name for tensor in model.inputs)
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot load benchmark input NPZ {path}: {exc}") from exc
    with archive:
        actual_names = tuple(archive.files)
        missing = set(expected_names) - set(actual_names)
        extra = set(actual_names) - set(expected_names)
        if missing or extra:
            raise ValueError(
                f"benchmark input NPZ names do not match model: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        inputs: dict[str, np.ndarray] = {}
        for tensor in model.inputs:
            value = np.asarray(archive[tensor.name])
            expected_shape = _concrete_shape(tensor, overrides, dynamic_dimension)
            if value.shape != expected_shape:
                raise ValueError(
                    f"benchmark input {tensor.name!r} shape is {value.shape}, "
                    f"expected {expected_shape}"
                )
            if value.dtype != tensor.dtype:
                raise ValueError(
                    f"benchmark input {tensor.name!r} dtype is {value.dtype}, "
                    f"expected {tensor.dtype}"
                )
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(
                    f"benchmark input {tensor.name!r} contains NaN or infinity"
                )
            inputs[tensor.name] = np.ascontiguousarray(value)
    return inputs


def _model_seed(base_seed: int, path: Path) -> int:
    identity = _relative_path(path).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return (base_seed + offset) % (1 << 64)


def _cached_rknn_path(source: Path, cache_root: Path) -> Path:
    try:
        relative = source.relative_to(ROOT)
    except ValueError:
        relative = Path(source.parent.name) / source.name
    return cache_root / relative.with_suffix(".rknn")


def _cases_for_model(
    model: ModelInfo,
    onnxruntime_providers: list[str],
    openvino_devices: list[str],
    include_auto: bool,
    rknn_target: str | None,
    rknn_cache: Path,
    rknn_output_names: tuple[str, ...],
    dynamic_dimension: int,
    shape_overrides: dict[str, tuple[int, ...]],
) -> list[BenchmarkCase]:
    cases = []
    cpu_provider = "CPUExecutionProvider"
    ignored_providers = {"AzureExecutionProvider"}
    if cpu_provider in onnxruntime_providers:
        cases.append(
            BenchmarkCase(
                "onnxruntime:CPU",
                OnnxBackendFactory(),
                OnnxArtifact(model.path, providers=(cpu_provider,)),
                requested_device=cpu_provider,
            )
        )
    acceleration_providers = [
        provider
        for provider in onnxruntime_providers
        if provider != cpu_provider and provider not in ignored_providers
    ]
    if acceleration_providers:
        cases.append(
            BenchmarkCase(
                "onnxruntime:auto",
                OnnxBackendFactory(),
                OnnxArtifact(model.path),
                requested_device="auto",
            )
        )
        automatically_selected = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in onnxruntime_providers
            else cpu_provider
        )
        for provider in acceleration_providers:
            if provider == automatically_selected:
                continue
            label = provider.removesuffix("ExecutionProvider")
            fallback = (cpu_provider,) if cpu_provider in onnxruntime_providers else ()
            cases.append(
                BenchmarkCase(
                    f"onnxruntime:{label}",
                    OnnxBackendFactory(),
                    OnnxArtifact(model.path, providers=(provider, *fallback)),
                    requested_device=provider,
                )
            )
    if not cases:
        cases.append(
            BenchmarkCase(
                "onnxruntime",
                OnnxBackendFactory(),
                OnnxArtifact(model.path),
            )
        )
    for device in openvino_devices:
        cases.append(
            BenchmarkCase(
                f"openvino:{device}",
                OpenVinoBackendFactory(),
                OpenVinoArtifact(model.path, device=device),
                requested_device=device,
            )
        )
    if include_auto:
        cases.append(
            BenchmarkCase(
                "openvino:AUTO",
                OpenVinoBackendFactory(),
                OpenVinoArtifact(model.path, device="AUTO"),
                requested_device="AUTO",
            )
        )

    adjacent_rknn = model.path.with_suffix(".rknn")
    conversion_enabled = _rknn_conversion_enabled()
    if adjacent_rknn.is_file():
        rknn_path = adjacent_rknn
    else:
        rknn_path = _cached_rknn_path(model.path, rknn_cache)
    if adjacent_rknn.is_file() or rknn_path.is_file() or conversion_enabled:
        available_outputs = {item.name for item in model.outputs}
        missing_outputs = set(rknn_output_names) - available_outputs
        if missing_outputs:
            raise ValueError(
                "requested RKNN output(s) do not exist in "
                f"{model.path.name}: {sorted(missing_outputs)}"
            )
        rknn_outputs = rknn_output_names or tuple(
            item.name for item in model.outputs if item.name == "actions"
        )
        if not rknn_outputs and model.outputs:
            rknn_outputs = (model.outputs[0].name,)
        rknn_inputs = _required_onnx_inputs(model.path, rknn_outputs) or tuple(
            item.name for item in model.inputs
        )
        cases.append(
            BenchmarkCase(
                "rknn",
                RknnBackendFactory(),
                RknnArtifact(
                    rknn_path,
                    target=rknn_target,
                    source_onnx=model.path,
                    conversion_output_names=rknn_outputs,
                    runtime_input_names=rknn_inputs,
                    input_shapes=_artifact_input_shapes(
                        model,
                        dynamic_dimension,
                        shape_overrides,
                    ),
                    output_shapes=tuple(
                        (item.name, tuple(_json_shape(item.shape)))
                        for item in model.outputs
                        if item.name in rknn_outputs
                    ),
                ),
                requested_device=rknn_target,
                output_names=rknn_outputs,
            )
        )
    return cases


def _json_shape(shape: tuple[object, ...]) -> list[object]:
    return [
        int(item) if isinstance(item, (int, np.integer)) else str(item)
        for item in shape
    ]


def _artifact_input_shapes(
    model: ModelInfo,
    dynamic_dimension: int,
    shape_overrides: dict[str, tuple[int, ...]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            tensor.name,
            _concrete_shape(tensor, shape_overrides, dynamic_dimension),
        )
        for tensor in model.inputs
    )


def _backend_details(case: BenchmarkCase, backend) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if case.name.startswith("onnxruntime"):
        try:
            details["execution_providers"] = list(backend._session.get_providers())
        except Exception:
            pass
    elif case.name.startswith("openvino:"):
        try:
            devices = backend._compiled_model.get_property("EXECUTION_DEVICES")
            details["execution_devices"] = [str(item) for item in devices]
        except Exception:
            pass
    return details


def _measure(
    backend,
    inputs: dict[str, np.ndarray],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    backend.warmup(inputs, warmup)
    initial_outputs = backend.run(inputs)
    output_ids = {name: id(initial_outputs[name]) for name in backend.output_names}
    samples = np.empty(iterations, dtype=np.int64)
    stable_output = True
    for index in range(iterations):
        started = time.perf_counter_ns()
        outputs = backend.run(inputs)
        samples[index] = time.perf_counter_ns() - started
        for name, expected_id in output_ids.items():
            if id(outputs[name]) != expected_id:
                stable_output = False
    final_outputs = {
        name: np.array(outputs[name], copy=True) for name in backend.output_names
    }
    mean_us = float(np.mean(samples) / 1_000.0)
    metrics = {
        "iterations": iterations,
        "warmup_runs": warmup,
        "p50_us": float(np.percentile(samples, 50) / 1_000.0),
        "p95_us": float(np.percentile(samples, 95) / 1_000.0),
        "p99_us": float(np.percentile(samples, 99) / 1_000.0),
        "mean_us": mean_us,
        "min_us": float(np.min(samples) / 1_000.0),
        "max_us": float(np.max(samples) / 1_000.0),
        "throughput_hz": 1_000_000.0 / mean_us,
        "stable_output_buffers": stable_output,
    }
    return metrics, final_outputs


def _compare_outputs(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    common = sorted(reference.keys() & candidate.keys())
    if not common:
        return {"comparable": False, "reason": "no common output names"}
    maximum = 0.0
    total = 0.0
    squared_total = 0.0
    reference_squared_total = 0.0
    candidate_squared_total = 0.0
    dot_total = 0.0
    count = 0
    allclose = True
    per_output: dict[str, dict[str, Any]] = {}
    for name in common:
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape:
            return {
                "comparable": False,
                "reason": f"output shape mismatch for {name}: {left.shape} != {right.shape}",
            }
        if not (
            np.issubdtype(left.dtype, np.number)
            and np.issubdtype(right.dtype, np.number)
        ):
            allclose &= bool(np.array_equal(left, right))
            continue

        reference_values = left.astype(np.float64, copy=False)
        candidate_values = right.astype(np.float64, copy=False)
        signed_difference = candidate_values - reference_values
        absolute_difference = np.abs(signed_difference)
        output_count = absolute_difference.size
        if output_count:
            output_maximum = float(np.max(absolute_difference))
            output_total = float(np.sum(absolute_difference))
            output_squared_total = float(np.sum(signed_difference * signed_difference))
            output_reference_squared = float(
                np.sum(reference_values * reference_values)
            )
            output_candidate_squared = float(
                np.sum(candidate_values * candidate_values)
            )
            output_dot = float(np.sum(reference_values * candidate_values))
            output_reference_norm = np.sqrt(output_reference_squared)
            output_candidate_norm = np.sqrt(output_candidate_squared)

            maximum = max(maximum, output_maximum)
            total += output_total
            squared_total += output_squared_total
            reference_squared_total += output_reference_squared
            candidate_squared_total += output_candidate_squared
            dot_total += output_dot
            count += output_count
            per_output[name] = {
                "element_count": output_count,
                "max_abs_error": output_maximum,
                "mean_abs_error": output_total / output_count,
                "rmse": np.sqrt(output_squared_total / output_count),
                "relative_l2_error": (
                    np.sqrt(output_squared_total) / output_reference_norm
                    if output_reference_norm > 0.0
                    else None
                ),
                "cosine_similarity": (
                    output_dot / (output_reference_norm * output_candidate_norm)
                    if output_reference_norm > 0.0 and output_candidate_norm > 0.0
                    else None
                ),
            }
        allclose &= bool(np.allclose(left, right, rtol=rtol, atol=atol))

    reference_norm = np.sqrt(reference_squared_total)
    candidate_norm = np.sqrt(candidate_squared_total)
    return {
        "comparable": True,
        "allclose": allclose,
        "max_abs_error": maximum,
        "mean_abs_error": total / count if count else 0.0,
        "rmse": np.sqrt(squared_total / count) if count else 0.0,
        "relative_l2_error": (
            np.sqrt(squared_total) / reference_norm if reference_norm > 0.0 else None
        ),
        "cosine_similarity": (
            dot_total / (reference_norm * candidate_norm)
            if reference_norm > 0.0 and candidate_norm > 0.0
            else None
        ),
        "per_output": per_output,
        "rtol": rtol,
        "atol": atol,
    }


def _run_case(
    case: BenchmarkCase,
    model: ModelInfo,
    inputs: dict[str, np.ndarray],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    result: dict[str, Any] = {
        "backend": case.name,
        "requested_device": case.requested_device,
    }
    setup_started = time.perf_counter_ns()
    availability = case.factory.availability(case.artifact)
    if not availability.available:
        result.update(
            {
                "status": "skipped",
                "setup_ms": (time.perf_counter_ns() - setup_started) / 1_000_000.0,
                "reason": availability.reason,
                "install_hint": availability.install_hint,
            }
        )
        return result, None

    spec = ModelSpec(
        artifacts=(case.artifact,),
        input_names=tuple(item.name for item in model.inputs),
        output_names=case.output_names or tuple(item.name for item in model.outputs),
    )
    backend = None
    try:
        backend = case.factory.open(case.artifact, spec)
        result["setup_ms"] = (time.perf_counter_ns() - setup_started) / 1_000_000.0
        result.update(_backend_details(case, backend))
        metrics, outputs = _measure(backend, inputs, warmup, iterations)
        result.update(metrics)
        result["status"] = "ok"
        return result, outputs
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "setup_ms": (time.perf_counter_ns() - setup_started) / 1_000_000.0,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return result, None
    finally:
        if backend is not None:
            backend.close()


def _worker_case(args: argparse.Namespace, model: ModelInfo) -> BenchmarkCase:
    path = args.worker_artifact_path
    if args.worker_backend.startswith("onnxruntime"):
        return BenchmarkCase(
            args.worker_backend,
            OnnxBackendFactory(),
            OnnxArtifact(
                path,
                providers=(tuple(args.worker_provider) or None),
            ),
            requested_device=args.worker_device or None,
        )
    if args.worker_backend.startswith("openvino:"):
        device = args.worker_device
        return BenchmarkCase(
            args.worker_backend,
            OpenVinoBackendFactory(),
            OpenVinoArtifact(path, device=device),
            requested_device=device,
        )
    if args.worker_backend == "rknn":
        output_names = tuple(args.worker_output) or tuple(
            item.name for item in model.outputs if item.name == "actions"
        )
        if not output_names and model.outputs:
            output_names = (model.outputs[0].name,)
        input_names = _required_onnx_inputs(model.path, output_names) or tuple(
            item.name for item in model.inputs
        )
        return BenchmarkCase(
            "rknn",
            RknnBackendFactory(),
            RknnArtifact(
                path,
                target=args.rknn_target,
                source_onnx=model.path,
                conversion_output_names=output_names,
                runtime_input_names=input_names,
                input_shapes=_artifact_input_shapes(
                    model,
                    args.dynamic_dim,
                    args.shape_overrides,
                ),
                output_shapes=tuple(
                    (item.name, tuple(_json_shape(item.shape)))
                    for item in model.outputs
                    if item.name in output_names
                ),
            ),
            requested_device=args.rknn_target,
            output_names=output_names,
        )
    raise ValueError(f"unknown benchmark worker backend: {args.worker_backend}")


def _worker_main(args: argparse.Namespace) -> int:
    try:
        model = _inspect_model(args.worker_model)
        inputs = _make_inputs(
            model,
            args.shape_overrides,
            args.dynamic_dim,
            args.seed,
            args.input_ranges,
            args.input_npz,
        )
        case = _worker_case(args, model)
        result, outputs = _run_case(
            case,
            model,
            inputs,
            args.warmup,
            args.iterations,
        )
    except Exception as exc:
        result = {
            "backend": args.worker_backend,
            "requested_device": args.worker_device,
            "status": "error",
            "setup_ms": 0.0,
            "reason": f"worker {type(exc).__name__}: {exc}",
        }
        outputs = None

    args.worker_result.parent.mkdir(parents=True, exist_ok=True)
    args.worker_result.write_text(json.dumps(result), encoding="utf-8")
    if outputs is not None:
        np.savez(args.worker_outputs, **outputs)
    return 0


def _worker_failure(
    case: BenchmarkCase,
    returncode: int | None,
    stderr: str,
    reason: str | None = None,
) -> dict[str, Any]:
    if reason is None:
        if returncode is not None and returncode < 0:
            reason = f"worker terminated by signal {-returncode}"
        else:
            reason = f"worker exited with code {returncode}"
    stderr = stderr.strip()
    if stderr:
        reason += ": " + " | ".join(stderr.splitlines()[-8:])
    return {
        "backend": case.name,
        "requested_device": case.requested_device,
        "status": "error",
        "setup_ms": 0.0,
        "reason": reason,
        "worker_returncode": returncode,
    }


def _run_case_isolated(
    case: BenchmarkCase,
    model: ModelInfo,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    with tempfile.TemporaryDirectory(prefix="bxi-inference-benchmark-") as directory:
        temporary = Path(directory)
        result_path = temporary / "result.json"
        outputs_path = temporary / "outputs.npz"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            "--_worker-model",
            str(model.path),
            "--_worker-backend",
            case.name,
            "--_worker-artifact-path",
            str(case.artifact.resolved_path),
            "--_worker-result",
            str(result_path),
            "--_worker-outputs",
            str(outputs_path),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--dynamic-dim",
            str(args.dynamic_dim),
            "--seed",
            str(seed),
        ]
        if case.requested_device:
            command.extend(("--_worker-device", case.requested_device))
        providers = getattr(case.artifact, "providers", None)
        if providers:
            for provider in providers:
                command.extend(("--_worker-provider", provider))
        if args.rknn_target:
            command.extend(("--rknn-target", args.rknn_target))
        if case.output_names:
            for output_name in case.output_names:
                command.extend(("--_worker-output", output_name))
        for name, shape in args.shape_overrides.items():
            command.extend(("--shape", f"{name}=" + ",".join(map(str, shape))))
        for name, (low, high) in args.input_ranges.items():
            command.extend(("--input-range", f"{name}={low!r},{high!r}"))
        if args.input_npz is not None:
            command.extend(("--input-npz", str(args.input_npz)))

        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=args.case_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = (
                exc.stderr.decode()
                if isinstance(exc.stderr, bytes)
                else exc.stderr or ""
            )
            return (
                _worker_failure(
                    case,
                    None,
                    stderr,
                    f"worker timed out after {args.case_timeout:.1f} seconds",
                ),
                None,
            )

        if not result_path.is_file():
            return (
                _worker_failure(case, completed.returncode, completed.stderr),
                None,
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return (
                _worker_failure(
                    case,
                    completed.returncode,
                    completed.stderr,
                    f"invalid worker result: {exc}",
                ),
                None,
            )
        result["worker_returncode"] = completed.returncode
        if completed.stderr.strip():
            result["worker_stderr"] = completed.stderr[-8_000:]
        outputs = None
        if result.get("status") == "ok" and outputs_path.is_file():
            with np.load(outputs_path, allow_pickle=False) as archive:
                outputs = {
                    name: np.array(archive[name], copy=True) for name in archive.files
                }
        return result, outputs


def _print_environment(environment: dict[str, Any]) -> None:
    print("System")
    print(f"  host: {environment['hostname']}")
    print(f"  platform: {environment['platform']}")
    print(f"  CPU: {environment['cpu']} ({environment['logical_cpu_count']} logical)")
    print(
        "  versions: "
        f"Python {environment['python']}, NumPy {environment['numpy']}, "
        f"ORT {environment['onnxruntime'] or '-'}, "
        f"OpenVINO {environment['openvino'] or '-'}, "
        f"RKNN Lite2 {environment['rknn_toolkit_lite2'] or '-'}"
    )
    print(f"  ORT providers: {environment['onnxruntime_providers'] or '-'}")
    print(f"  OpenVINO devices: {environment['openvino_devices'] or '-'}")
    if environment["openvino_unsupported_devices"]:
        print(
            "  OpenVINO skipped devices: "
            f"{environment['openvino_unsupported_devices']}"
        )
    print()


def _input_statistics(value: np.ndarray) -> dict[str, float | int | None]:
    if not value.size or not np.issubdtype(value.dtype, np.number):
        return {"minimum": None, "maximum": None}
    return {
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def _format_input_range(value: np.ndarray) -> str:
    statistics = _input_statistics(value)
    minimum = statistics["minimum"]
    maximum = statistics["maximum"]
    if minimum is None or maximum is None:
        return "n/a"
    return f"{minimum:.3g},{maximum:.3g}"


def _print_model_header(
    model: ModelInfo,
    inputs: dict[str, np.ndarray],
    *,
    input_source: str,
    effective_seed: int,
) -> None:
    input_text = ", ".join(
        f"{name}:{tuple(value.shape)}:{value.dtype}" f"[{_format_input_range(value)}]"
        for name, value in inputs.items()
    )
    print(
        f"Model: {_relative_path(model.path)} ({model.path.stat().st_size / 1e6:.2f} MB)"
    )
    print(f"  input source: {input_source}; effective seed: {effective_seed}")
    print(f"  inputs: {input_text}")
    print(
        f"  {'backend':<28} {'setup':>10} {'p50':>10} {'p95':>10} "
        f"{'p99':>10} {'mean':>10} {'Hz':>10} {'stable':>8} {'match':>7}"
    )
    print("  " + "-" * 117)


def _print_case(result: dict[str, Any]) -> None:
    name = result["backend"]
    if name == "openvino:AUTO" and result.get("execution_devices"):
        name += "[" + ",".join(result["execution_devices"]) + "]"
    if result["status"] != "ok":
        detail = result.get("reason", "unknown")
        if result.get("install_hint"):
            detail += f"; install: {result['install_hint']}"
        print(
            f"  {name:<28} {result['status']}: {detail} "
            f"({result['setup_ms']:.1f} ms)"
        )
        return
    comparison = result.get("comparison", {})
    if comparison.get("reference"):
        match = "ref"
    elif not comparison.get("comparable", False):
        match = "n/a"
    else:
        match = "yes" if comparison.get("allclose") else "NO"
    print(
        f"  {name:<28} {result['setup_ms']:>7.1f} ms "
        f"{result['p50_us']:>7.1f} us {result['p95_us']:>7.1f} us "
        f"{result['p99_us']:>7.1f} us {result['mean_us']:>7.1f} us "
        f"{result['throughput_hz']:>9.1f} {str(result['stable_output_buffers']):>8} "
        f"{match:>7}"
    )
    if comparison.get("reference") or not comparison.get("comparable", False):
        return

    def metric(name: str, values: dict[str, Any] = comparison) -> str:
        value = values.get(name)
        return "n/a" if value is None else f"{value:.3e}"

    print(
        f"    precision vs {comparison.get('reference_backend', 'reference')}: "
        f"max_abs={metric('max_abs_error')}, "
        f"mean_abs={metric('mean_abs_error')}, "
        f"rmse={metric('rmse')}, "
        f"rel_l2={metric('relative_l2_error')}, "
        f"cosine={metric('cosine_similarity')}"
    )
    per_output = comparison.get("per_output", {})
    if len(per_output) > 1:
        for output_name, output_metrics in per_output.items():
            print(
                f"      output {output_name}: "
                f"max_abs={metric('max_abs_error', output_metrics)}, "
                f"mean_abs={metric('mean_abs_error', output_metrics)}, "
                f"rmse={metric('rmse', output_metrics)}, "
                f"rel_l2={metric('relative_l2_error', output_metrics)}, "
                f"cosine={metric('cosine_similarity', output_metrics)}"
            )


def _default_report_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().replace("/", "_")
    return ROOT / "tools/benchmark/results" / f"benchmark-{host}-{stamp}.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "models",
        nargs="*",
        type=Path,
        help="ONNX files or directories; omitted means every ONNX model under src/",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 10 warmups and 100 measurements for a fast smoke test",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="NAME=D1,D2",
        help="override a dynamic input shape; repeat for multiple inputs",
    )
    parser.add_argument(
        "--input-range",
        action="append",
        default=[],
        metavar="NAME=LOW,HIGH",
        help=(
            "override the default [-1,1] uniform range for a floating input; "
            "repeat for multiple inputs"
        ),
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        help=(
            "load named model inputs from one NPZ file; requires exactly one model "
            "and cannot be combined with --input-range"
        ),
    )
    parser.add_argument(
        "--dynamic-dim",
        type=int,
        default=1,
        help="value used for unresolved dynamic dimensions",
    )
    parser.add_argument(
        "--no-openvino-auto",
        action="store_true",
        help="do not benchmark OpenVINO AUTO in addition to concrete devices",
    )
    parser.add_argument("--rknn-target", help="target for an existing RKNN model")
    parser.add_argument(
        "--rknn-cache",
        type=Path,
        default=ROOT / "tools/benchmark/cache/rknn",
        help="output cache used when RKNN conversion is explicitly enabled",
    )
    parser.add_argument(
        "--rknn-output",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "RKNN output to retain; repeat to preserve a multi-output production "
            "contract. By default only actions (or the first model output) is used"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=300.0,
        help="maximum seconds for one isolated model/backend worker",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--json",
        type=Path,
        help="JSON report path; omitted creates a timestamped report",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="do not write a JSON report"
    )
    parser.add_argument(
        "--_worker",
        dest="worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-model",
        dest="worker_model",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-backend", dest="worker_backend", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_worker-device",
        dest="worker_device",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-provider",
        dest="worker_provider",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-artifact-path",
        dest="worker_artifact_path",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-result",
        dest="worker_result",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-outputs",
        dest="worker_outputs",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-output",
        dest="worker_output",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.quick:
        args.warmup = 10
        args.iterations = 100
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.dynamic_dim <= 0
        or args.case_timeout <= 0
        or args.seed < 0
    ):
        parser.error(
            "warmup and seed must be non-negative; iterations, dynamic-dim and "
            "timeout positive"
        )
    try:
        args.shape_overrides = _parse_shape_overrides(args.shape)
        args.input_ranges = _parse_input_ranges(args.input_range)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.input_npz is not None:
        if args.input_range:
            parser.error("--input-npz cannot be combined with --input-range")
        args.input_npz = args.input_npz.expanduser().resolve()
        if not args.input_npz.is_file():
            parser.error(f"input NPZ does not exist: {args.input_npz}")
    return args


def main() -> int:
    args = _arguments()
    if args.worker:
        required = (
            args.worker_model,
            args.worker_backend,
            args.worker_artifact_path,
            args.worker_result,
            args.worker_outputs,
        )
        if any(value is None for value in required):
            print("incomplete internal benchmark worker arguments", file=sys.stderr)
            return 2
        return _worker_main(args)
    models = _discover_models(args.models)
    if not models:
        print("no ONNX models found", file=sys.stderr)
        return 2
    if args.input_npz is not None and len(models) != 1:
        print("--input-npz requires exactly one discovered model", file=sys.stderr)
        return 2

    environment, openvino_devices = _environment()
    _print_environment(environment)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "environment": environment,
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "case_timeout_seconds": args.case_timeout,
            "process_isolation": True,
            "dynamic_dimension": args.dynamic_dim,
            "shape_overrides": {
                name: list(shape) for name, shape in args.shape_overrides.items()
            },
            "input_source": (
                str(args.input_npz) if args.input_npz is not None else "generated"
            ),
            "default_float_distribution": "uniform[-1,1]",
            "input_ranges": {
                name: [low, high] for name, (low, high) in args.input_ranges.items()
            },
            "seed": args.seed,
            "base_seed": args.seed,
            "seed_derivation": "base_seed + first_u64(sha256(model_path))",
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "models": [],
    }
    successful_cases = 0

    used_input_ranges: set[str] = set()
    for path in models:
        effective_seed = _model_seed(args.seed, path)
        try:
            model = _inspect_model(path)
            used_input_ranges.update(
                args.input_ranges.keys() & {tensor.name for tensor in model.inputs}
            )
            inputs = _make_inputs(
                model,
                args.shape_overrides,
                args.dynamic_dim,
                effective_seed,
                args.input_ranges,
                args.input_npz,
            )
        except Exception as exc:
            print(f"Model: {_relative_path(path)}\n  inspect error: {exc}\n")
            report["models"].append(
                {
                    "path": _relative_path(path),
                    "status": "inspect_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        model_result: dict[str, Any] = {
            "path": _relative_path(path),
            "size_bytes": path.stat().st_size,
            "input_source": (
                str(args.input_npz) if args.input_npz is not None else "generated"
            ),
            "effective_seed": effective_seed,
            "inputs": [
                {
                    "name": item.name,
                    "model_shape": _json_shape(item.shape),
                    "benchmark_shape": list(inputs[item.name].shape),
                    "dtype": str(item.dtype),
                    **_input_statistics(inputs[item.name]),
                }
                for item in model.inputs
            ],
            "outputs": [
                {
                    "name": item.name,
                    "model_shape": _json_shape(item.shape),
                    "dtype": str(item.dtype),
                }
                for item in model.outputs
            ],
            "benchmarks": [],
        }
        _print_model_header(
            model,
            inputs,
            input_source=(
                str(args.input_npz) if args.input_npz is not None else "generated"
            ),
            effective_seed=effective_seed,
        )
        reference_outputs = None
        reference_backend = None
        for case in _cases_for_model(
            model,
            environment["onnxruntime_providers"],
            openvino_devices,
            not args.no_openvino_auto,
            args.rknn_target,
            args.rknn_cache.expanduser().resolve(),
            tuple(args.rknn_output),
            args.dynamic_dim,
            args.shape_overrides,
        ):
            result, outputs = _run_case_isolated(
                case,
                model,
                args,
                effective_seed,
            )
            if result["status"] == "ok":
                successful_cases += 1
                if reference_outputs is None:
                    reference_outputs = outputs
                    reference_backend = result["backend"]
                    result["comparison"] = {"reference": True}
                else:
                    result["comparison"] = {
                        "reference_backend": reference_backend,
                        **_compare_outputs(
                            reference_outputs,
                            outputs,
                            args.rtol,
                            args.atol,
                        ),
                    }
            model_result["benchmarks"].append(result)
            _print_case(result)
        good = [item for item in model_result["benchmarks"] if item["status"] == "ok"]
        if good:
            fastest = min(good, key=lambda item: item["mean_us"])
            model_result["fastest_backend"] = fastest["backend"]
            print(
                f"  fastest mean: {fastest['backend']} "
                f"({fastest['mean_us']:.1f} us)"
            )
        print()
        report["models"].append(model_result)

    unused_input_ranges = set(args.input_ranges) - used_input_ranges
    if unused_input_ranges:
        print(
            "warning: input range names did not match any model input: "
            + ", ".join(sorted(unused_input_ranges)),
            file=sys.stderr,
        )

    if not args.no_json:
        report_path = (args.json or _default_report_path()).expanduser().resolve()
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"warning: could not write JSON report {report_path}: {exc}")
        else:
            print(f"JSON report: {report_path}")
    print(f"Completed: {len(models)} models, {successful_cases} successful cases")
    return 0 if successful_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
