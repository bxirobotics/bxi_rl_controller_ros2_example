"""Resolve the XRoboToolkit PC Service shipped with or installed for this Mod."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import sysconfig


_SYSTEM_SERVICE_ROOT = Path("/opt/apps/roboticsservice")
_SERVICE_SDK_DIRECTORIES = {
    "linux-x86_64": "x64",
    "linux-aarch64": "arm64",
}


def runtime_platform_tag() -> str:
    """Use the same stable platform spelling as the framework runtime profiles."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sysconfig.get_platform().lower())


def runtime_python_tag() -> str:
    """Match the framework's platform-and-interpreter vendor directory tag."""
    cache_tag = sys.implementation.cache_tag
    if not cache_tag:
        cache_tag = f"python-{sys.version_info.major}{sys.version_info.minor}"
    safe_cache_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_tag.lower())
    return f"{runtime_platform_tag()}-{safe_cache_tag}"


def service_sdk_directory_name(platform_tag: str | None = None) -> str:
    """Map the framework platform tag to the vendor SDK directory spelling."""
    tag = platform_tag or runtime_platform_tag()
    try:
        return _SERVICE_SDK_DIRECTORIES[tag]
    except KeyError as exc:
        raise RuntimeError(
            f"XRoboToolkit is unsupported on platform {tag!r}; "
            f"supported platforms: {', '.join(sorted(_SERVICE_SDK_DIRECTORIES))}"
        ) from exc


def mod_root() -> Path:
    explicit = os.environ.get("BXI_MOD_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    # Keep the logical path when a symlink install points at the source Mod.
    return Path(__file__).absolute().parents[1]


def bundled_service_root() -> Path:
    return mod_root() / "runtime" / runtime_platform_tag() / "roboticsservice"


def resolve_service_root() -> Path:
    """Select explicit or user-installed service, then the bundled fallback.

    An existing user installation is authoritative even when incomplete. This
    makes a damaged or incompatible user runtime fail visibly instead of
    silently changing ABI by selecting the bundled copy.
    """
    explicit = os.environ.get("SONIC_XRT_SERVICE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    if _SYSTEM_SERVICE_ROOT.exists():
        return _SYSTEM_SERVICE_ROOT
    bundled = bundled_service_root()
    if bundled.exists():
        return bundled
    return _SYSTEM_SERVICE_ROOT


def service_library_paths(
    root: Path, platform_tag: str | None = None
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    tag = platform_tag or runtime_platform_tag()
    sdk_directory = _SERVICE_SDK_DIRECTORIES.get(tag)
    if sdk_directory is not None:
        candidates.append(root / "SDK" / sdk_directory)

    # External service builds may use an architecture spelling unknown to the
    # Mod. Discover their SDK directory by the library contract instead of
    # rejecting an otherwise valid user-managed platform.
    sdk_root = root / "SDK"
    if sdk_root.is_dir():
        candidates.extend(
            child
            for child in sorted(sdk_root.iterdir())
            if child.is_dir() and (child / "libPXREARobotSDK.so").is_file()
        )
    candidates.extend((root, root / "lib"))
    return tuple(dict.fromkeys(path for path in candidates if path.is_dir()))


def prepare_service_environment() -> Path:
    """Prepare a child re-exec to load the SDK from the selected runtime."""
    root = resolve_service_root()
    paths = [str(path) for path in service_library_paths(root)]
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    if inherited:
        paths.extend(value for value in inherited.split(os.pathsep) if value)
    if paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    os.environ["SONIC_XRT_SERVICE_DIR"] = str(root)
    return root


__all__ = [
    "bundled_service_root",
    "mod_root",
    "prepare_service_environment",
    "resolve_service_root",
    "runtime_platform_tag",
    "runtime_python_tag",
    "service_sdk_directory_name",
    "service_library_paths",
]
