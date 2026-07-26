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
    vendor_python: bool
    vendor_libraries: tuple[Path, ...]

    @property
    def available(self) -> bool:
        return not self.errors

    @property
    def uses_vendor(self) -> bool:
        return self.vendor_python or bool(self.vendor_libraries)


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
) -> RuntimeRequirementReport:
    """Check vendor-first, then host availability without installing anything."""

    vendor_python_root = mod_root / "vendor" / "python"
    vendor_library_root = mod_root / "vendor" / "lib"
    errors: list[str] = []
    vendor_python = False
    vendor_libraries: list[Path] = []

    for requirement in requirements.python:
        if _vendor_python_module_exists(vendor_python_root, requirement.import_name):
            vendor_python = True
            continue
        if _host_python_module_exists(requirement.import_name):
            continue
        errors.append(f"missing Python module '{requirement.import_name}'")

    for requirement in requirements.ros:
        try:
            get_package_prefix(requirement.package)
        except PackageNotFoundError:
            errors.append(f"missing ROS package '{requirement.package}'")

    for requirement in requirements.system:
        vendor_library = _find_vendor_library(vendor_library_root, requirement.library)
        if vendor_library is not None:
            vendor_libraries.append(vendor_library)
            continue
        if _host_library_exists(requirement.library):
            continue
        errors.append(f"missing system library '{requirement.library}'")

    return RuntimeRequirementReport(
        errors=tuple(errors),
        vendor_python=vendor_python,
        vendor_libraries=tuple(dict.fromkeys(vendor_libraries)),
    )


def vendor_python_path(mod_root: Path) -> Path:
    return mod_root / "vendor" / "python"


def vendor_library_path(mod_root: Path) -> Path:
    return mod_root / "vendor" / "lib"


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
    "vendor_library_path",
    "vendor_python_path",
]
