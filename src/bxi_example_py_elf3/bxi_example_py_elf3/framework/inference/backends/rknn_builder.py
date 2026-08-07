"""Opt-in, cached ONNX to RKNN conversion used before RKNN Lite loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import warnings

from ..model import RknnArtifact


RKNN_CONVERT_ENV = "BXI_RKNN_CONVERT_ON_LOAD"
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_KEYS = {
    "target",
    "target_platform",
    "do_quantization",
    "dataset",
    "config",
    "force_rebuild",
    "outputs",
}
_TOOLKIT_INSTALL_HINT = (
    "install the official rknn-toolkit2 wheel matching your Python version "
    "and x86_64 platform from the RKNN Toolkit2 release"
)


@dataclass(frozen=True, slots=True)
class RknnPreparation:
    ready: bool
    converted: bool = False
    reason: str = ""
    install_hint: str | None = None
    target: str | None = None


@dataclass(frozen=True, slots=True)
class _ConversionSettings:
    enabled: bool
    target: str | None = None
    do_quantization: bool = False
    dataset: Path | None = None
    config: tuple[tuple[str, object], ...] = ()
    force_rebuild: bool = False
    output_names: tuple[str, ...] = ()


def _environment_settings(artifact: RknnArtifact) -> _ConversionSettings:
    raw = os.environ.get(RKNN_CONVERT_ENV)
    if raw is None or raw.strip().lower() in _FALSE_VALUES:
        return _ConversionSettings(False)

    value = raw.strip()
    overrides: dict[str, object]
    if value.lower() in _TRUE_VALUES:
        overrides = {}
    elif value.startswith("{"):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{RKNN_CONVERT_ENV} must be a target name or valid JSON: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{RKNN_CONVERT_ENV} JSON must be an object")
        unknown = set(loaded) - _ENV_KEYS
        if unknown:
            raise ValueError(
                f"unsupported {RKNN_CONVERT_ENV} setting(s): "
                + ", ".join(sorted(unknown))
            )
        overrides = loaded
    else:
        overrides = {"target": value}

    target = overrides.get("target", overrides.get("target_platform", artifact.target))
    if target is not None and not isinstance(target, str):
        raise ValueError("RKNN conversion target must be a string")

    quantization = overrides.get("do_quantization", artifact.do_quantization)
    if not isinstance(quantization, bool):
        raise ValueError("RKNN do_quantization must be true or false")

    dataset_value = overrides.get("dataset", artifact.dataset)
    if dataset_value is not None and not isinstance(dataset_value, (str, os.PathLike)):
        raise ValueError("RKNN dataset must be a path string")
    dataset = (
        Path(dataset_value).expanduser().resolve()
        if dataset_value is not None
        else None
    )

    config = dict(artifact.build_config)
    environment_config = overrides.get("config", {})
    if not isinstance(environment_config, dict):
        raise ValueError("RKNN conversion config must be a JSON object")
    config.update(environment_config)
    if "target_platform" in config:
        raise ValueError(
            "put target_platform in the RKNN target setting, not build config"
        )

    force_rebuild = overrides.get("force_rebuild", False)
    if not isinstance(force_rebuild, bool):
        raise ValueError("RKNN force_rebuild must be true or false")

    output_value = overrides.get("outputs", artifact.conversion_output_names)
    if isinstance(output_value, str):
        output_names = (output_value,)
    elif isinstance(output_value, (list, tuple)):
        if not all(isinstance(item, str) and item for item in output_value):
            raise ValueError("RKNN outputs must be non-empty strings")
        output_names = tuple(output_value)
    else:
        raise ValueError("RKNN outputs must be a string or string list")

    return _ConversionSettings(
        True,
        target=target,
        do_quantization=quantization,
        dataset=dataset,
        config=tuple(config.items()),
        force_rebuild=force_rebuild,
        output_names=output_names,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(
    path: Path,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stat = path.stat()
    identity: dict[str, object] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if (
        previous
        and previous.get("path") == identity["path"]
        and previous.get("size") == identity["size"]
        and previous.get("mtime_ns") == identity["mtime_ns"]
        and isinstance(previous.get("sha256"), str)
    ):
        identity["sha256"] = previous["sha256"]
    else:
        identity["sha256"] = _sha256(path)
    return identity


def _dataset_identity(
    path: Path,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    identity = _file_identity(path, previous)
    entries: list[dict[str, object]] = []
    base = path.parent
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = []
    for line in lines:
        for item in line.strip().split():
            candidate = Path(item).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            try:
                stat = candidate.stat()
            except OSError:
                entries.append({"path": str(candidate.resolve()), "missing": True})
            else:
                entries.append(
                    {
                        "path": str(candidate.resolve()),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    identity["entries"] = entries
    return identity


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(
        f"RKNN build config value {value!r} is not JSON-compatible; "
        "use strings, numbers, booleans, lists, tuples or dictionaries"
    )


def _rknn_input_size_list(
    input_shapes: tuple[tuple[str, tuple[object, ...]], ...],
) -> tuple[tuple[int, ...], ...] | None:
    if not input_shapes:
        return None
    resolved: list[tuple[int, ...]] = []
    for _name, shape in input_shapes:
        if not shape:
            return None
        concrete: list[int] = []
        for dimension in shape:
            if not isinstance(dimension, int) or dimension <= 0:
                return None
            concrete.append(dimension)
        resolved.append(tuple(concrete))
    return tuple(resolved)


def _read_manifest(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _installed_toolkit_version() -> str | None:
    try:
        return importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError:
        return None


def _fingerprint(
    source: Path,
    settings: _ConversionSettings,
    previous: Mapping[str, object] | None,
    toolkit_version: str | None,
    input_shapes: tuple[tuple[str, tuple[object, ...]], ...],
) -> dict[str, object]:
    previous_source = previous.get("source") if previous else None
    if not isinstance(previous_source, Mapping):
        previous_source = None
    previous_dataset = previous.get("dataset") if previous else None
    if not isinstance(previous_dataset, Mapping):
        previous_dataset = None
    dataset = (
        _dataset_identity(settings.dataset, previous_dataset)
        if settings.do_quantization and settings.dataset is not None
        else None
    )
    return {
        "source": _file_identity(source, previous_source),
        "target": settings.target,
        "toolkit_version": toolkit_version,
        "do_quantization": settings.do_quantization,
        "dataset": dataset,
        "config": _json_value(dict(settings.config)),
        "outputs": list(settings.output_names),
        "inputs": [
            {"name": name, "shape": _json_value(shape)} for name, shape in input_shapes
        ],
    }


def _runtime_input_shapes(
    artifact: RknnArtifact,
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """Return physical RKNN inputs in runtime order, rejecting incomplete specs."""

    declared = dict(artifact.input_shapes)
    if not artifact.runtime_input_names:
        return artifact.input_shapes
    missing = [name for name in artifact.runtime_input_names if name not in declared]
    if missing:
        raise ValueError(
            "RKNN artifact is missing shapes for runtime inputs: " + ", ".join(missing)
        )
    return tuple((name, declared[name]) for name in artifact.runtime_input_names)


def _manifest_contract_error(
    artifact: RknnArtifact,
    manifest: Mapping[str, object],
    input_shapes: tuple[tuple[str, tuple[object, ...]], ...],
) -> str | None:
    """Check a portable cache sidecar without trusting machine-local paths."""

    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        return "RKNN build manifest has no fingerprint object"

    artifact_identity = manifest.get("artifact")
    if artifact_identity is not None:
        if not isinstance(artifact_identity, Mapping) or not isinstance(
            artifact_identity.get("sha256"), str
        ):
            return "RKNN build manifest has an invalid artifact digest"
        if artifact_identity["sha256"] != _sha256(artifact.resolved_path):
            return "RKNN model does not match its build manifest"

    expected_outputs = artifact.conversion_output_names
    declared_outputs = fingerprint.get("outputs")
    if expected_outputs:
        if not isinstance(declared_outputs, list) or not all(
            isinstance(item, str) for item in declared_outputs
        ):
            return "RKNN build manifest has no valid output contract"
        if tuple(declared_outputs) != expected_outputs:
            return (
                "RKNN output contract mismatch: cache="
                f"{tuple(declared_outputs)}, required={expected_outputs}"
            )

    declared_inputs = fingerprint.get("inputs")
    if declared_inputs is not None:
        if not isinstance(declared_inputs, list) or not all(
            isinstance(item, Mapping)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("shape"), list)
            for item in declared_inputs
        ):
            return "RKNN build manifest has an invalid input contract"
        cached_inputs = tuple(
            (str(item["name"]), tuple(item["shape"])) for item in declared_inputs
        )
        if cached_inputs != input_shapes:
            return (
                "RKNN input contract mismatch: cache="
                f"{cached_inputs}, required={input_shapes}"
            )

    source_identity = fingerprint.get("source")
    source_path = (
        Path(artifact.source_onnx).expanduser().resolve()
        if artifact.source_onnx is not None
        else None
    )
    if source_path is not None and source_path.is_file():
        if not isinstance(source_identity, Mapping) or not isinstance(
            source_identity.get("sha256"), str
        ):
            return "RKNN build manifest has no ONNX source digest"
        if source_identity["sha256"] != _sha256(source_path):
            return "RKNN cache was built from a different ONNX model"

    cached_target = fingerprint.get("target")
    if artifact.target and cached_target != artifact.target:
        return (
            f"RKNN target mismatch: cache={cached_target!r}, "
            f"required={artifact.target!r}"
        )
    return None


def _rknn_api():
    from rknn.api import RKNN

    return RKNN


def _api_version(rknn_type: type, fallback: str | None) -> str:
    module = __import__(rknn_type.__module__.split(".", 1)[0])
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else fallback or "unknown"


def _check_result(operation: str, result: object) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {operation} failed with code {result}")


def _write_manifest(
    path: Path, fingerprint: Mapping[str, object], artifact_path: Path
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema": 1,
                    "fingerprint": fingerprint,
                    "artifact": {
                        "size": artifact_path.stat().st_size,
                        "sha256": _sha256(artifact_path),
                    },
                },
                stream,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _convert(
    source: Path,
    destination: Path,
    manifest_path: Path,
    input_shapes: tuple[tuple[str, tuple[object, ...]], ...],
    settings: _ConversionSettings,
    previous_fingerprint: Mapping[str, object] | None,
    installed_version: str | None,
) -> None:
    try:
        rknn_type = _rknn_api()
    except ModuleNotFoundError as exc:
        if exc.name == "rknn" or (exc.name and exc.name.startswith("rknn.")):
            raise ModuleNotFoundError(_TOOLKIT_INSTALL_HINT) from exc
        raise RuntimeError(
            f"RKNN Toolkit dependency is missing: {exc.name}; install it into "
            "the active Python environment"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(f"RKNN Toolkit import failed: {exc}") from exc

    fingerprint = _fingerprint(
        source,
        settings,
        previous_fingerprint,
        _api_version(rknn_type, installed_version),
        input_shapes,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.stem + ".", suffix=".rknn", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    converter = None
    try:
        converter = rknn_type()
        _check_result(
            "config",
            converter.config(
                target_platform=settings.target,
                **dict(settings.config),
            ),
        )
        load_arguments: dict[str, object] = {"model": str(source)}
        input_names = [name for name, _shape in input_shapes]
        input_size_list = _rknn_input_size_list(input_shapes)
        if input_names:
            load_arguments["inputs"] = input_names
        if input_size_list is not None:
            load_arguments["input_size_list"] = [
                list(shape) for shape in input_size_list
            ]
        if settings.output_names:
            load_arguments["outputs"] = list(settings.output_names)
        _check_result("load_onnx", converter.load_onnx(**load_arguments))
        build_arguments: dict[str, object] = {
            "do_quantization": settings.do_quantization
        }
        if settings.do_quantization and settings.dataset is not None:
            build_arguments["dataset"] = str(settings.dataset)
        _check_result("build", converter.build(**build_arguments))
        _check_result("export_rknn", converter.export_rknn(str(temporary)))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("RKNN export_rknn did not create a non-empty model")
        os.replace(temporary, destination)
        _write_manifest(manifest_path, fingerprint, destination)
    finally:
        try:
            if converter is not None:
                converter.release()
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def prepare_rknn_artifact(artifact: RknnArtifact) -> RknnPreparation:
    """Prepare an RKNN cache only when explicitly authorized by the environment."""

    destination = artifact.resolved_path
    manifest_path = Path(str(destination) + ".build.json")
    try:
        input_shapes = _runtime_input_shapes(artifact)
    except ValueError as exc:
        return RknnPreparation(False, reason=str(exc))
    try:
        settings = _environment_settings(artifact)
    except (TypeError, ValueError) as exc:
        return RknnPreparation(False, reason=str(exc))
    if not settings.enabled:
        if destination.is_file():
            if manifest_path.exists():
                manifest = _read_manifest(manifest_path)
                if manifest is None:
                    return RknnPreparation(
                        False,
                        reason=f"invalid RKNN build manifest: {manifest_path}",
                    )
                contract_error = _manifest_contract_error(
                    artifact, manifest, input_shapes
                )
                if contract_error:
                    return RknnPreparation(False, reason=contract_error)
            return RknnPreparation(
                True,
                reason=(
                    "RKNN model and build contract are compatible"
                    if manifest_path.exists()
                    else "legacy RKNN model exists without a build contract"
                ),
                target=artifact.target,
            )
        return RknnPreparation(
            False,
            reason=(
                f"model does not exist: {artifact.path}; set {RKNN_CONVERT_ENV} "
                "to an RKNN target or JSON conversion settings to build it"
            ),
        )
    if artifact.source_onnx is None:
        return RknnPreparation(
            False,
            reason=f"{RKNN_CONVERT_ENV} is set but source_onnx is not configured",
        )
    if not settings.target:
        return RknnPreparation(
            False,
            reason=f"{RKNN_CONVERT_ENV} is set but no RKNN target is configured",
        )

    source = Path(artifact.source_onnx).expanduser().resolve()
    if not source.is_file():
        return RknnPreparation(False, reason=f"ONNX source does not exist: {source}")
    if settings.do_quantization and settings.dataset is None:
        return RknnPreparation(
            False,
            reason="RKNN quantization requires a calibration dataset",
        )
    if (
        settings.do_quantization
        and settings.dataset is not None
        and not settings.dataset.is_file()
    ):
        return RknnPreparation(
            False,
            reason=f"RKNN calibration dataset does not exist: {settings.dataset}",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(destination) + ".build.lock")
    try:
        import fcntl

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            manifest = _read_manifest(manifest_path)
            previous = manifest.get("fingerprint") if manifest else None
            if not isinstance(previous, Mapping):
                previous = None
            installed_version = _installed_toolkit_version()
            comparison_version = installed_version
            if comparison_version is None and previous is not None:
                saved_version = previous.get("toolkit_version")
                if isinstance(saved_version, str):
                    comparison_version = saved_version
            current = _fingerprint(
                source,
                settings,
                previous,
                comparison_version,
                input_shapes,
            )
            if not settings.force_rebuild and destination.is_file():
                if previous == current:
                    return RknnPreparation(
                        True,
                        reason="RKNN cache is current",
                        target=settings.target,
                    )
                if (
                    manifest is None
                    and not manifest_path.exists()
                    and (
                        artifact.target is None
                        or artifact.target.lower() == settings.target.lower()
                    )
                    and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
                ):
                    return RknnPreparation(
                        True,
                        reason="existing RKNN model is newer than its ONNX source",
                        target=settings.target,
                    )

            warnings.warn(
                f"converting ONNX to RKNN for {settings.target}; this can make "
                f"startup significantly slower: {source} -> {destination}",
                RuntimeWarning,
                stacklevel=2,
            )
            _convert(
                source,
                destination,
                manifest_path,
                input_shapes,
                settings,
                previous,
                installed_version,
            )
            warnings.warn(
                f"RKNN conversion completed and cached: {destination}",
                RuntimeWarning,
                stacklevel=2,
            )
            return RknnPreparation(
                True,
                converted=True,
                reason="RKNN model was converted and cached",
                target=settings.target,
            )
    except ModuleNotFoundError as exc:
        return RknnPreparation(
            False,
            reason=str(exc),
            install_hint=_TOOLKIT_INSTALL_HINT,
        )
    except Exception as exc:
        return RknnPreparation(False, reason=f"RKNN conversion failed: {exc}")


__all__ = [
    "RKNN_CONVERT_ENV",
    "RknnPreparation",
    "prepare_rknn_artifact",
]
