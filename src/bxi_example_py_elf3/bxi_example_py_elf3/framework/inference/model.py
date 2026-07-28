"""Backend-neutral model artifacts and logical model descriptions."""

from __future__ import annotations

from dataclasses import dataclass
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
    core_mask: object | None = None
    input_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    output_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
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
        openvino_device: str = "AUTO",
        providers: tuple[str, ...] | None = None,
    ) -> "ModelSpec":
        """Prefer OpenVINO for an ONNX file and fall back to ONNX Runtime."""

        return cls(
            artifacts=(
                OpenVinoArtifact(path, device=openvino_device),
                OnnxArtifact(path, providers=providers),
            ),
            input_names=input_names,
            output_names=output_names,
        )


__all__ = [
    "ModelArtifact",
    "ModelSpec",
    "OnnxArtifact",
    "OpenVinoArtifact",
    "RknnArtifact",
]
