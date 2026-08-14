"""Backend-neutral model artifacts and logical model descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    path: str | Path
    backend: ClassVar[str] = ""

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class OnnxArtifact(ModelArtifact):
    backend: ClassVar[str] = "onnxruntime"
    providers: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RknnArtifact(ModelArtifact):
    backend: ClassVar[str] = "rknn"
    target: str | None = None
    source_onnx: str | Path | None = None
    do_quantization: bool = False
    dataset: str | Path | None = None
    build_config: tuple[tuple[str, object], ...] = ()
    conversion_output_names: tuple[str, ...] = ()
    core_mask: object | None = None
    runtime_input_names: tuple[str, ...] = ()
    input_shapes: tuple[tuple[str, tuple[object, ...]], ...] = ()
    output_shapes: tuple[tuple[str, tuple[object, ...]], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class OpenVinoArtifact(ModelArtifact):
    backend: ClassVar[str] = "openvino"
    device: str = "AUTO"
    config: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    artifacts: tuple[ModelArtifact, ...]
    input_names: tuple[str, ...] = ("obs",)
    output_names: tuple[str, ...] = ("actions",)

    def __post_init__(self) -> None:
        if not self.artifacts:
            raise ValueError("model spec needs at least one artifact")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("model input names must be unique")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("model output names must be unique")

    @classmethod
    def onnx(
        cls,
        path: str | Path,
        *,
        input_names: tuple[str, ...] = ("obs",),
        output_names: tuple[str, ...] = ("actions",),
        providers: tuple[str, ...] | None = None,
    ) -> "ModelSpec":
        return cls(
            artifacts=(OnnxArtifact(path, providers=providers),),
            input_names=input_names,
            output_names=output_names,
        )

    @classmethod
    def portable_onnx(
        cls,
        path: str | Path,
        *,
        input_names: tuple[str, ...] = ("obs",),
        output_names: tuple[str, ...] = ("actions",),
        rknn_path: str | Path | None = None,
        rknn_target: str | None = None,
        rknn_core_mask: object | None = None,
        rknn_output_names: tuple[str, ...] | None = None,
        rknn_build_config: tuple[tuple[str, object], ...] = (),
        rknn_do_quantization: bool = False,
        rknn_dataset: str | Path | None = None,
        openvino_device: str = "AUTO",
        providers: tuple[str, ...] | None = None,
    ) -> "ModelSpec":
        """Prefer RKNN, then OpenVINO, then ONNX Runtime for an ONNX file.

        ``output_names`` is the stable logical model contract. When
        ``rknn_output_names`` declares a strict subset, the runtime executes
        that subset on RKNN and preserves every missing output with an exact
        ONNX Runtime sidecar extracted from the source graph.
        """

        onnx_path = Path(path)
        rknn_model_path = (
            Path(rknn_path) if rknn_path is not None else onnx_path.with_suffix(".rknn")
        )
        metadata, input_shapes, output_shapes = _read_onnx_description(onnx_path)
        inferred_input_names = tuple(name for name, _shape in input_shapes)
        inferred_output_names = tuple(name for name, _shape in output_shapes)
        logical_input_names = input_names or inferred_input_names
        logical_output_names = output_names or inferred_output_names
        physical_rknn_outputs = (
            logical_output_names
            if rknn_output_names is None
            else tuple(rknn_output_names)
        )
        if not physical_rknn_outputs:
            raise ValueError("RKNN physical output names must not be empty")
        unknown_rknn_outputs = set(physical_rknn_outputs) - set(
            logical_output_names
        )
        if unknown_rknn_outputs:
            raise ValueError(
                "RKNN physical outputs are outside the logical model contract: "
                f"{sorted(unknown_rknn_outputs)}"
            )
        rknn_inputs = (
            _required_onnx_inputs(onnx_path, physical_rknn_outputs)
            or logical_input_names
        )

        return cls(
            artifacts=(
                RknnArtifact(
                    rknn_model_path,
                    target=rknn_target,
                    source_onnx=onnx_path,
                    do_quantization=rknn_do_quantization,
                    dataset=rknn_dataset,
                    build_config=rknn_build_config,
                    conversion_output_names=physical_rknn_outputs,
                    core_mask=rknn_core_mask,
                    runtime_input_names=rknn_inputs,
                    # Keep all logical input descriptions so a composite
                    # backend can expose one stable model contract. Physical
                    # RKNN inputs and outputs are declared independently.
                    input_shapes=input_shapes,
                    output_shapes=tuple(
                        (name, shape)
                        for name, shape in output_shapes
                        if name in physical_rknn_outputs
                    ),
                    metadata=metadata,
                ),
                OpenVinoArtifact(path, device=openvino_device),
                OnnxArtifact(path, providers=providers),
            ),
            input_names=logical_input_names,
            output_names=logical_output_names,
        )


def _dimension_value(dimension) -> object:
    value = getattr(dimension, "dim_value", 0)
    if value:
        return int(value)
    param = getattr(dimension, "dim_param", "")
    return str(param) if param else "?"


def _value_info_shape(value_info) -> tuple[object, ...]:
    tensor_type = value_info.type.tensor_type
    return tuple(_dimension_value(item) for item in tensor_type.shape.dim)


def _read_onnx_description(
    path: Path,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, tuple[object, ...]], ...],
    tuple[tuple[str, tuple[object, ...]], ...],
]:
    if path.suffix.lower() != ".onnx" or importlib.util.find_spec("onnx") is None:
        return (), (), ()
    try:
        import onnx

        model = onnx.load_model(str(path), load_external_data=False)
    except Exception:
        return (), (), ()
    metadata = tuple((item.key, item.value) for item in model.metadata_props)
    input_shapes = tuple(
        (item.name, _value_info_shape(item)) for item in model.graph.input
    )
    output_shapes = tuple(
        (item.name, _value_info_shape(item)) for item in model.graph.output
    )
    return metadata, input_shapes, output_shapes


def _required_onnx_inputs(
    path: Path,
    output_names: tuple[str, ...],
) -> tuple[str, ...]:
    if (
        path.suffix.lower() != ".onnx"
        or not output_names
        or importlib.util.find_spec("onnx") is None
    ):
        return ()
    try:
        import onnx

        model = onnx.load_model(str(path), load_external_data=False)
    except Exception:
        return ()
    graph_inputs = {item.name for item in model.graph.input}
    required_outputs = set(output_names)
    needed_tensors = set(required_outputs)
    needed_inputs: set[str] = set()
    for node in reversed(model.graph.node):
        produced = set(node.output)
        if not produced.intersection(needed_tensors):
            continue
        for name in node.input:
            if not name:
                continue
            if name in graph_inputs:
                needed_inputs.add(name)
            else:
                needed_tensors.add(name)
    ordered = tuple(
        item.name for item in model.graph.input if item.name in needed_inputs
    )
    return ordered


__all__ = [
    "ModelArtifact",
    "ModelSpec",
    "OnnxArtifact",
    "OpenVinoArtifact",
    "RknnArtifact",
]
