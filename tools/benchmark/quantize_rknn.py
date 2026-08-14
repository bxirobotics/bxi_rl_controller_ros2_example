#!/usr/bin/env python3
"""Validate captured model inputs and build INT8 RKNN artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = ROOT / "src/bxi_example_py_elf3"
DEFAULT_CACHE = ROOT / "tools/benchmark/cache/rknn"
RKNN_CONVERT_ENV = "BXI_RKNN_CONVERT_ON_LOAD"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from bxi_example_py_elf3.framework.inference.backends.rknn_builder import (  # noqa: E402
    prepare_rknn_artifact,
)
from bxi_example_py_elf3.framework.inference.model import (  # noqa: E402
    ModelSpec,
    RknnArtifact,
)


def _concrete_shape(value_info) -> tuple[int, ...]:
    dimensions: list[int] = []
    for index, dimension in enumerate(value_info.type.tensor_type.shape.dim):
        value = int(dimension.dim_value)
        if value > 0:
            dimensions.append(value)
        elif index == 0:
            dimensions.append(1)
        else:
            name = dimension.dim_param or "?"
            raise ValueError(
                f"input {value_info.name!r} has unresolved dimension {name!r}"
            )
    return tuple(dimensions)


def _model_inputs(model_path: Path) -> tuple[tuple[str, tuple[int, ...]], ...]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX is required to validate calibration data") from exc

    model = onnx.load_model(str(model_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return tuple(
        (value_info.name, _concrete_shape(value_info))
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    )


def _dataset_lines(dataset_path: Path) -> list[tuple[Path, ...]]:
    try:
        raw_lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"calibration dataset does not exist: {dataset_path}") from exc

    samples: list[tuple[Path, ...]] = []
    for line_number, raw_line in enumerate(raw_lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        resolved_entries: list[Path] = []
        for item in line.split():
            original = Path(item).expanduser()
            candidate = (
                original if original.is_absolute() else dataset_path.parent / original
            )
            if not candidate.is_file() and original.is_absolute():
                relocated = dataset_path.parent / original.name
                if relocated.is_file():
                    candidate = relocated
            resolved_entries.append(candidate.resolve())
        entries = tuple(resolved_entries)
        samples.append(entries)
    return samples


def _resolved_dataset(dataset_path: Path) -> Path:
    """Write a movable capture as local absolute paths for RKNN Toolkit."""
    destination = dataset_path.with_name("dataset.rknn.txt")
    temporary = destination.with_suffix(".txt.tmp")
    samples = _dataset_lines(dataset_path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for paths in samples:
                stream.write(" ".join(str(path) for path in paths) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_dataset(
    model_path: Path,
    dataset_path: Path,
    *,
    min_samples: int,
) -> int:
    inputs = _model_inputs(model_path)
    samples = _dataset_lines(dataset_path)
    if len(samples) < min_samples:
        raise ValueError(
            f"{dataset_path} contains {len(samples)} samples; "
            f"at least {min_samples} are required"
        )

    metadata_path = dataset_path.with_name("capture.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        captured_names = tuple(item.get("name") for item in metadata.get("inputs", ()))
        expected_names = tuple(name for name, _shape in inputs)
        if captured_names != expected_names:
            raise ValueError(
                "captured input order does not match ONNX: "
                f"captured={captured_names}, expected={expected_names}"
            )

    for sample_index, paths in enumerate(samples):
        if len(paths) != len(inputs):
            raise ValueError(
                f"sample {sample_index} contains {len(paths)} inputs; "
                f"model requires {len(inputs)}"
            )
        for input_index, (path, (name, shape)) in enumerate(zip(paths, inputs)):
            if not path.is_file():
                raise ValueError(
                    f"sample {sample_index} input {name!r} is missing: {path}"
                )
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise ValueError(
                    f"cannot load sample {sample_index} input {name!r}: {path}: {exc}"
                ) from exc
            if value.shape != shape:
                raise ValueError(
                    f"sample {sample_index} input {input_index} {name!r} shape "
                    f"is {value.shape}, expected {shape}: {path}"
                )
            if value.dtype != np.float32:
                raise ValueError(
                    f"sample {sample_index} input {name!r} dtype is {value.dtype}, "
                    f"expected float32: {path}"
                )
            if not np.isfinite(value).all():
                raise ValueError(
                    f"sample {sample_index} input {name!r} contains NaN or infinity: "
                    f"{path}"
                )
    return len(samples)


def _cache_path(model_path: Path, cache_root: Path) -> Path:
    try:
        relative = model_path.relative_to(ROOT)
    except ValueError:
        relative = Path(model_path.parent.name) / model_path.name
    return cache_root / relative.with_suffix(".rknn")


def _convert(
    model_path: Path,
    dataset_path: Path,
    destination: Path,
    *,
    target: str,
    output_name: str,
    quantized_dtype: str,
    quantized_algorithm: str,
    quantized_method: str,
) -> Path:
    spec = ModelSpec.portable_onnx(
        model_path,
        input_names=(),
        output_names=(output_name,),
        rknn_path=destination,
        rknn_target=target,
    )
    artifact = spec.artifacts[0]
    if not isinstance(artifact, RknnArtifact):
        raise RuntimeError("portable ONNX spec did not create an RKNN artifact")
    artifact = replace(
        artifact,
        input_shapes=_model_inputs(model_path),
    )

    settings = {
        "target": target,
        "do_quantization": True,
        "dataset": str(dataset_path),
        "outputs": [output_name],
        "config": {
            "quantized_dtype": quantized_dtype,
            "quantized_algorithm": quantized_algorithm,
            "quantized_method": quantized_method,
            "optimization_level": 3,
        },
        "force_rebuild": True,
    }
    previous = os.environ.get(RKNN_CONVERT_ENV)
    os.environ[RKNN_CONVERT_ENV] = json.dumps(settings, separators=(",", ":"))
    try:
        preparation = prepare_rknn_artifact(artifact)
    finally:
        if previous is None:
            os.environ.pop(RKNN_CONVERT_ENV, None)
        else:
            os.environ[RKNN_CONVERT_ENV] = previous
    if not preparation.ready:
        hint = f"; {preparation.install_hint}" if preparation.install_hint else ""
        raise RuntimeError(f"{preparation.reason}{hint}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"RKNN conversion did not create {destination}")
    return destination


def _install(source: Path, model_path: Path) -> Path:
    destination = model_path.with_suffix(".rknn")
    source_manifest = Path(str(source) + ".build.json")
    destination_manifest = Path(str(destination) + ".build.json")
    if not source_manifest.is_file():
        raise RuntimeError(f"RKNN build contract does not exist: {source_manifest}")
    for source_path, destination_path in (
        (source, destination),
        (source_manifest, destination_manifest),
    ):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination_path.name}.",
                suffix=".tmp",
                dir=destination_path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, destination_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return destination


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate calibration tensors captured by the external capture tool and "
            "build INT8 RKNN models."
        )
    )
    parser.add_argument("models", nargs="+", type=Path, help="ONNX model paths")
    parser.add_argument(
        "--calibration-root",
        required=True,
        type=Path,
        help="root written by collect_calibration.py --output",
    )
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--output", default="actions")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--quantized-dtype",
        choices=("w8a8", "w8a16", "w16a16i", "w16a16i_dfp", "w4a16"),
        default="w8a8",
    )
    parser.add_argument(
        "--algorithm",
        choices=("normal", "mmse", "kl_divergence", "gdq"),
        default="normal",
    )
    parser.add_argument("--method", default="channel")
    parser.add_argument(
        "--install",
        action="store_true",
        help="atomically copy each generated RKNN beside its ONNX model",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate captured tensors without importing RKNN Toolkit",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.min_samples <= 0:
        raise SystemExit("--min-samples must be positive")

    calibration_root = args.calibration_root.expanduser().resolve()
    cache_root = args.cache.expanduser().resolve()
    jobs: list[tuple[Path, Path]] = []
    for raw_model_path in args.models:
        model_path = raw_model_path.expanduser().resolve()
        if not model_path.is_file() or model_path.suffix.lower() != ".onnx":
            raise SystemExit(f"model is not an ONNX file: {model_path}")
        dataset_path = calibration_root / model_path.stem / "dataset.txt"
        try:
            sample_count = validate_dataset(
                model_path,
                dataset_path,
                min_samples=args.min_samples,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"calibration validation failed for {model_path}: {exc}")
        print(
            f"VALID {model_path.name}: samples={sample_count}, "
            f"dataset={dataset_path}"
        )
        jobs.append((model_path, dataset_path))

    if args.validate_only:
        return 0

    for model_path, dataset_path in jobs:
        destination = _cache_path(model_path, cache_root)
        toolkit_dataset_path = _resolved_dataset(dataset_path)
        try:
            converted = _convert(
                model_path,
                toolkit_dataset_path,
                destination,
                target=args.target,
                output_name=args.output,
                quantized_dtype=args.quantized_dtype,
                quantized_algorithm=args.algorithm,
                quantized_method=args.method,
            )
        except Exception as exc:
            raise SystemExit(f"RKNN conversion failed for {model_path}: {exc}")
        print(f"BUILT {converted} ({converted.stat().st_size / 1e6:.2f} MB)")
        if args.install:
            installed = _install(converted, model_path)
            print(f"INSTALLED {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
