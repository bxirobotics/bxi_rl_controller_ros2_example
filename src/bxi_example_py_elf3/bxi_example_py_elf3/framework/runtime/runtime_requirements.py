"""Runtime dependency declarations and availability checks for Mods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import ctypes.util
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
from typing import cast

from ament_index_python.packages import PackageNotFoundError, get_package_prefix


@dataclass(frozen=True)
class PythonRequirement:
    import_name: str


@dataclass(frozen=True)
class RosRequirement:
    package: str


@dataclass(frozen=True)
class SystemRequirement:
    library: str


@dataclass(frozen=True)
class RuntimeRequirements:
    python: tuple[PythonRequirement, ...]
    ros: tuple[RosRequirement, ...]
    system: tuple[SystemRequirement, ...]


@dataclass(frozen=True)
class RuntimeRequirementReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    vendor_python_paths: tuple[Path, ...]
    vendor_libraries: tuple[Path, ...]

    @property
    def available(self) -> bool:
        return not self.errors

    @property
    def uses_vendor(self) -> bool:
        return bool(self.vendor_python_paths or self.vendor_libraries)

    @property
    def vendor_python(self) -> bool:
        return bool(self.vendor_python_paths)


def read_runtime_requirements(
    value: object,
    context: str,
) -> RuntimeRequirements:
    """Parse the explicit runtime_requirements manifest block."""

    requirements = _mapping(value, context)
    expected_fields = {"python", "ros", "system"}
    missing = expected_fields - set(requirements)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    unknown = set(requirements) - expected_fields
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")

    return RuntimeRequirements(
        python=tuple(
            PythonRequirement(item)
            for item in _read_named_entries(
                requirements["python"],
                context=f"{context}.python",
                field="import",
                pattern=r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            )
        ),
        ros=tuple(
            RosRequirement(item)
            for item in _read_named_entries(
                requirements["ros"],
                context=f"{context}.ros",
                field="package",
                pattern=r"[a-z][a-z0-9_]*",
            )
        ),
        system=tuple(
            SystemRequirement(item)
            for item in _read_named_entries(
                requirements["system"],
                context=f"{context}.system",
                field="library",
                pattern=r"[A-Za-z0-9][A-Za-z0-9_.+-]*",
            )
        ),
    )


def check_runtime_requirements(
    requirements: RuntimeRequirements,
    mod_root: Path,
    *,
    python_executable: Path | None = None,
    python_paths: Sequence[Path] | None = None,
    library_paths: Sequence[Path] | None = None,
    environment: Mapping[str, str] | None = None,
) -> RuntimeRequirementReport:
    """Check vendor-first, then host availability without installing anything."""

    selected_python_paths = (
        vendor_python_paths(mod_root) if python_paths is None else tuple(python_paths)
    )
    selected_library_paths = (
        vendor_library_paths(mod_root)
        if library_paths is None
        else tuple(library_paths)
    )
    errors: list[str] = []
    warnings: list[str] = []
    used_python_paths: list[Path] = []
    vendor_libraries: list[Path] = []

    for requirement in requirements.python:
        if python_executable is not None:
            bundled_python = any(
                _vendor_python_module_exists(root, requirement.import_name)
                for root in selected_python_paths
            )
            probe_error = _probe_python_import(
                requirement.import_name,
                selected_python_paths,
                selected_library_paths,
                executable=python_executable,
                environment=environment,
            )
            if probe_error is not None:
                errors.append(
                    f"Python module '{requirement.import_name}' is not importable "
                    f"with '{python_executable}': {probe_error}"
                )
            elif bundled_python:
                used_python_paths.extend(selected_python_paths)
            continue
        bundled_python = any(
            _vendor_python_module_exists(root, requirement.import_name)
            for root in selected_python_paths
        )
        if bundled_python:
            probe_error = _probe_python_import(
                requirement.import_name,
                selected_python_paths,
                selected_library_paths,
            )
            if probe_error is not None:
                errors.append(
                    f"bundled Python module '{requirement.import_name}' is not "
                    f"importable for target '{runtime_python_tag()}': {probe_error}"
                )
                continue
            used_python_paths.extend(selected_python_paths)
            continue

        incompatible_targets = _incompatible_python_targets(
            mod_root,
            requirement.import_name,
        )
        legacy_vendor = _vendor_python_module_exists(
            mod_root / "vendor" / "python",
            requirement.import_name,
        )
        if incompatible_targets:
            warnings.append(
                f"ignored bundled Python module '{requirement.import_name}' "
                f"for incompatible targets {list(incompatible_targets)}"
            )
        if legacy_vendor:
            warnings.append(
                f"ignored legacy flat vendor Python module "
                f"'{requirement.import_name}'; move it under "
                f"vendor/python/{runtime_python_tag()} or vendor/python/common"
            )
        if _host_python_module_exists(requirement.import_name):
            probe_error = _probe_python_import(requirement.import_name, ())
            if probe_error is not None:
                errors.append(
                    f"host Python module '{requirement.import_name}' is not "
                    f"importable: {probe_error}"
                )
                continue
            continue
        errors.append(f"missing Python module '{requirement.import_name}'")

    for requirement in requirements.ros:
        try:
            get_package_prefix(requirement.package)
        except PackageNotFoundError:
            errors.append(f"missing ROS package '{requirement.package}'")

    for requirement in requirements.system:
        vendor_library = next(
            (
                library
                for root in selected_library_paths
                if (library := _find_vendor_library(root, requirement.library))
                is not None
            ),
            None,
        )
        if vendor_library is not None:
            vendor_libraries.append(vendor_library)
            continue
        incompatible_targets = _incompatible_library_targets(
            mod_root,
            requirement.library,
        )
        if incompatible_targets:
            warnings.append(
                f"ignored bundled system library '{requirement.library}' "
                f"for incompatible targets {list(incompatible_targets)}"
            )
        if _host_library_exists(requirement.library):
            continue
        errors.append(f"missing system library '{requirement.library}'")

    return RuntimeRequirementReport(
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
        vendor_python_paths=tuple(dict.fromkeys(used_python_paths)),
        vendor_libraries=tuple(dict.fromkeys(vendor_libraries)),
    )


def runtime_platform_tag() -> str:
    return _safe_tag(sysconfig.get_platform())


def runtime_python_tag() -> str:
    cache_tag = sys.implementation.cache_tag
    if not cache_tag:
        cache_tag = f"python-{sys.version_info.major}{sys.version_info.minor}"
    return f"{runtime_platform_tag()}-{_safe_tag(cache_tag)}"


def vendor_python_paths(mod_root: Path) -> tuple[Path, ...]:
    root = mod_root / "vendor" / "python"
    candidates = (root / runtime_python_tag(), root / "common")
    return tuple(path.resolve() for path in candidates if path.is_dir())


def vendor_library_paths(mod_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []

    # Native Python wheels commonly keep their private dependency libraries
    # beside the extension module. Put those directories before the generic
    # vendor/lib directory so a process cannot accidentally bind the extension
    # to an ABI-incompatible host library with the same SONAME.
    for python_root in vendor_python_paths(mod_root):
        private_library_directories: set[Path] = set()
        for candidate in sorted(python_root.rglob("*")):
            if candidate.is_file() and _is_private_shared_library(candidate.name):
                private_library_directories.add(candidate.parent.resolve())
        paths.extend(
            sorted(
                private_library_directories,
                key=lambda path: (
                    len(path.relative_to(python_root).parts),
                    str(path),
                ),
            )
        )

    root = mod_root / "vendor" / "lib"
    candidate = root / runtime_platform_tag()
    if candidate.is_dir():
        paths.append(candidate.resolve())
    return tuple(dict.fromkeys(paths))


def _is_private_shared_library(filename: str) -> bool:
    lower = filename.lower()
    return (
        lower.endswith(".dll")
        or (lower.startswith("lib") and ".so" in lower)
        or (lower.startswith("lib") and ".dylib" in lower)
    )


def _read_named_entries(
    value: object,
    *,
    context: str,
    field: str,
    pattern: str,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a list")
    result: list[str] = []
    for index, raw_entry in enumerate(value):
        entry = _mapping(raw_entry, f"{context}[{index}]")
        if set(entry) != {field}:
            raise ValueError(
                f"{context}[{index}] must contain only the '{field}' field"
            )
        name = entry[field]
        if not isinstance(name, str) or not re.fullmatch(pattern, name):
            raise ValueError(f"{context}[{index}].{field} is invalid: {name!r}")
        if name in result:
            raise ValueError(f"{context} contains duplicate '{name}'")
        result.append(name)
    return result


def _vendor_python_module_exists(root: Path, import_name: str) -> bool:
    if not root.is_dir():
        return False
    relative = Path(*import_name.split("."))
    module_path = root / relative
    if module_path.is_dir():
        return True
    return any(
        module_path.with_suffix(suffix).is_file()
        for suffix in importlib.machinery.all_suffixes()
    )


def _host_python_module_exists(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _probe_python_import(
    import_name: str,
    extra_paths: Sequence[Path],
    library_paths: Sequence[Path] = (),
    *,
    executable: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    script = (
        "import importlib, sys\n"
        "for path in reversed(sys.argv[2:]):\n"
        "    sys.path.insert(0, path)\n"
        "importlib.import_module(sys.argv[1])\n"
    )
    child_environment = dict(environment or os.environ)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if library_paths:
        inherited = child_environment.get("LD_LIBRARY_PATH")
        values = [*(str(path) for path in library_paths)]
        if inherited:
            values.extend(inherited.split(os.pathsep))
        child_environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(values))
    try:
        completed = subprocess.run(
            [
                str(executable or sys.executable),
                "-c",
                script,
                import_name,
                *(str(path) for path in extra_paths),
            ],
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if completed.returncode == 0:
        return None
    output = completed.stderr.strip() or completed.stdout.strip()
    detail = output.splitlines()[-1] if output else f"exit code {completed.returncode}"
    return detail[:500]


def _incompatible_python_targets(
    mod_root: Path,
    import_name: str,
) -> tuple[str, ...]:
    root = mod_root / "vendor" / "python"
    if not root.is_dir():
        return ()
    current = runtime_python_tag()
    return tuple(
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir()
        and child.name not in (current, "common")
        and _is_python_target_tag(child.name)
        and _vendor_python_module_exists(child, import_name)
    )


def _incompatible_library_targets(
    mod_root: Path,
    library: str,
) -> tuple[str, ...]:
    root = mod_root / "vendor" / "lib"
    if not root.is_dir():
        return ()
    current = runtime_platform_tag()
    return tuple(
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir()
        and child.name != current
        and _find_vendor_library(child, library) is not None
    )


def _is_python_target_tag(value: str) -> bool:
    return re.fullmatch(r".+-(?:cpython|pypy)-\d+", value) is not None


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())


def _find_vendor_library(root: Path, library: str) -> Path | None:
    if not root.is_dir():
        return None
    exact_names = {
        library,
        f"lib{library}.so",
        f"lib{library}.dylib",
        f"{library}.dll",
    }
    versioned_prefixes = (f"lib{library}.so.", f"lib{library}.dylib.")
    for candidate in sorted(root.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.name in exact_names or candidate.name.startswith(
            versioned_prefixes
        ):
            return candidate.resolve()
    return None


def _host_library_exists(library: str) -> bool:
    if ctypes.util.find_library(library) is not None:
        return True
    search_roots = tuple(
        Path(item)
        for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if item
    )
    return any(_find_vendor_library(root, library) is not None for root in search_roots)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed map")
    return cast(Mapping[str, object], value)


__all__ = [
    "RuntimeRequirementReport",
    "RuntimeRequirements",
    "check_runtime_requirements",
    "read_runtime_requirements",
    "runtime_platform_tag",
    "runtime_python_tag",
    "vendor_library_paths",
    "vendor_python_paths",
]
