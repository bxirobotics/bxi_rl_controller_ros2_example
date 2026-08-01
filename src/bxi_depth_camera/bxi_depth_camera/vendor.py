from __future__ import annotations

import importlib
import importlib.machinery
import os
from pathlib import Path
import re
import sys
import sysconfig

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - only used outside a ROS environment
    get_package_share_directory = None


_FALLBACK_ENV = "BXI_DEPTH_CAMERA_VENDOR_MODULES"
_BOOTSTRAPPED_ENV = "BXI_DEPTH_CAMERA_BOOTSTRAPPED"


def runtime_tag() -> str:
    platform = re.sub(r"[^A-Za-z0-9_.-]+", "_", sysconfig.get_platform().lower())
    cache_tag = sys.implementation.cache_tag or (
        f"python-{sys.version_info.major}{sys.version_info.minor}"
    )
    cache_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_tag.lower())
    return f"{platform}-{cache_tag}"


def vendor_root() -> Path:
    if get_package_share_directory is not None:
        try:
            installed = Path(get_package_share_directory("bxi_depth_camera")) / "vendor"
            if installed.is_dir():
                return installed
        except Exception:
            pass
    return Path(__file__).resolve().parents[1] / "vendor"


def vendor_python_root() -> Path:
    return vendor_root() / "python" / runtime_tag()


def vendor_module_exists(module_name: str) -> bool:
    root = vendor_python_root()
    relative = root.joinpath(*module_name.split("."))
    return relative.is_dir() or any(
        relative.with_suffix(suffix).is_file()
        for suffix in importlib.machinery.all_suffixes()
    )


def fallback_modules() -> frozenset[str]:
    return frozenset(
        name for name in os.environ.get(_FALLBACK_ENV, "").split(",") if name
    )


def probe_system_modules(module_names: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception:
            for loaded in tuple(sys.modules):
                if loaded == name or loaded.startswith(name + "."):
                    sys.modules.pop(loaded, None)
            if vendor_module_exists(name):
                missing.append(name)
    return tuple(missing)


def vendor_library_paths(module_names: tuple[str, ...]) -> tuple[Path, ...]:
    root = vendor_python_root()
    paths: list[Path] = []
    for module_name in module_names:
        module_root = root.joinpath(*module_name.split("."))
        if not module_root.is_dir():
            continue
        candidates = [
            module_root,
            *(path for path in module_root.rglob("*") if path.is_dir()),
        ]
        for candidate in candidates:
            try:
                contains_library = any(
                    child.is_file()
                    and child.name.startswith("lib")
                    and ".so" in child.name
                    for child in candidate.iterdir()
                )
            except OSError:
                continue
            if contains_library:
                paths.append(candidate.resolve())
    return tuple(dict.fromkeys(paths))


def bootstrap_environment(module_names: tuple[str, ...]) -> dict[str, str] | None:
    if os.environ.get(_BOOTSTRAPPED_ENV) == "1":
        return None
    fallback = probe_system_modules(module_names)
    if not fallback:
        return None
    environment = dict(os.environ)
    environment[_BOOTSTRAPPED_ENV] = "1"
    environment[_FALLBACK_ENV] = ",".join(fallback)
    library_paths = [str(path) for path in vendor_library_paths(fallback)]
    inherited = environment.get("LD_LIBRARY_PATH", "")
    if inherited:
        library_paths.extend(path for path in inherited.split(os.pathsep) if path)
    if library_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(library_paths))
    return environment


def load_sdk(module_name: str):
    if module_name not in fallback_modules():
        try:
            return importlib.import_module(module_name), None
        except Exception as exc:
            return None, f"system import failed: {exc}"

    root = vendor_python_root()
    if not vendor_module_exists(module_name):
        return None, f"no bundled runtime for {runtime_tag()}"
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"bundled import failed from {root}: {exc}"
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


__all__ = [
    "bootstrap_environment",
    "fallback_modules",
    "load_sdk",
    "runtime_tag",
    "vendor_python_root",
]
