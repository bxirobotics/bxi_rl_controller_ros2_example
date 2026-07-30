"""Language-neutral process runtime profiles declared by Mods.

Profiles are deliberately resolved from paths inside the Mod directory.  A
portable Mod can therefore be moved without rewriting its manifest, while a
simple Mod can continue to use the host process environment without declaring
any profile at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Literal, cast

from .runtime_requirements import (
    runtime_platform_tag,
    runtime_python_tag,
    vendor_library_paths,
    vendor_python_paths,
)


RuntimeMode = Literal["host", "vendor", "portable"]


@dataclass(frozen=True)
class RuntimeCandidate:
    mode: RuntimeMode
    root: str | None = None
    python: str | None = None
    executable_paths: tuple[str, ...] = ()
    library_paths: tuple[str, ...] = ()
    isolated: bool = False


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    candidates: tuple[RuntimeCandidate, ...]


@dataclass(frozen=True)
class ResolvedRuntime:
    """One concrete environment used to start a process node."""

    name: str
    mode: RuntimeMode
    root: Path | None = None
    python_executable: Path | None = None
    python_home: Path | None = None
    python_paths: tuple[Path, ...] = ()
    executable_paths: tuple[Path, ...] = ()
    library_paths: tuple[Path, ...] = ()
    isolated: bool = False

    def apply_environment(self, environment: dict[str, str]) -> None:
        """Apply this runtime to a fresh child-process environment."""

        if self.isolated:
            for name in (
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONEXECUTABLE",
                "__PYVENV_LAUNCHER__",
            ):
                environment.pop(name, None)
            environment["PYTHONNOUSERSITE"] = "1"

        environment["BXI_RUNTIME_MODE"] = self.mode
        if self.root is not None:
            environment["BXI_RUNTIME_ROOT"] = str(self.root)
        else:
            environment.pop("BXI_RUNTIME_ROOT", None)
        if self.python_executable is not None:
            environment["BXI_PYTHON_EXECUTABLE"] = str(self.python_executable)
        else:
            environment.pop("BXI_PYTHON_EXECUTABLE", None)
        if self.python_home is not None:
            environment["PYTHONHOME"] = str(self.python_home)

        _prepend_path(environment, "PATH", self.executable_paths)
        _prepend_path(environment, "PYTHONPATH", self.python_paths)
        _prepend_path(environment, "LD_LIBRARY_PATH", self.library_paths)


@dataclass(frozen=True)
class RuntimeSelection:
    runtime: ResolvedRuntime | None
    error: str | None = None
    warnings: tuple[str, ...] = ()


def read_runtime_profiles(value: object, context: str) -> dict[str, RuntimeProfile]:
    """Parse optional top-level ``runtime_profiles`` declarations."""

    profiles = _mapping(value, context)
    result: dict[str, RuntimeProfile] = {}
    for name, raw_profile in profiles.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name):
            raise ValueError(f"{context} has invalid profile name: {name!r}")
        profile_context = f"{context}.{name}"
        profile = _mapping(raw_profile, profile_context)
        if "candidates" in profile:
            if set(profile) != {"candidates"}:
                raise ValueError(
                    f"{profile_context} cannot combine candidates with "
                    "single-candidate fields"
                )
            raw_candidates = profile["candidates"]
            if not isinstance(raw_candidates, Sequence) or isinstance(
                raw_candidates, (str, bytes)
            ):
                raise ValueError(f"{profile_context}.candidates must be a list")
            if not raw_candidates:
                raise ValueError(f"{profile_context}.candidates must not be empty")
            candidates = tuple(
                _read_candidate(item, f"{profile_context}.candidates[{index}]")
                for index, item in enumerate(raw_candidates)
            )
        else:
            candidates = (_read_candidate(profile, profile_context),)
        result[name] = RuntimeProfile(name=name, candidates=candidates)
    return result


def resolve_runtime_profile(
    profiles: Mapping[str, RuntimeProfile],
    profile_name: str | None,
    mod_root: Path,
    *,
    context: str,
) -> RuntimeSelection:
    """Resolve a profile without installing or modifying anything.

    A missing portable/vendor candidate may fall through to the next declared
    candidate.  Once a candidate directory exists, corruption is a hard error
    and is never hidden by a host fallback.
    """

    if profile_name is None:
        python_paths = vendor_python_paths(mod_root)
        library_paths = vendor_library_paths(mod_root)
        if python_paths or library_paths:
            return RuntimeSelection(
                ResolvedRuntime(
                    name="legacy-vendor",
                    mode="vendor",
                    python_paths=python_paths,
                    library_paths=library_paths,
                )
            )
        return RuntimeSelection(ResolvedRuntime(name="host", mode="host"))

    profile = profiles.get(profile_name)
    if profile is None:
        raise ValueError(
            f"{context} references unknown runtime profile {profile_name!r}"
        )

    skipped: list[str] = []
    for candidate in profile.candidates:
        if candidate.mode == "host":
            return RuntimeSelection(
                ResolvedRuntime(name=profile.name, mode="host"),
                warnings=tuple(skipped),
            )
        if candidate.mode == "vendor":
            python_paths = vendor_python_paths(mod_root)
            library_paths = vendor_library_paths(mod_root)
            if not python_paths and not library_paths:
                skipped.append(
                    f"runtime profile '{profile.name}' skipped vendor candidate: "
                    "no compatible vendor directories are present"
                )
                continue
            return RuntimeSelection(
                ResolvedRuntime(
                    name=profile.name,
                    mode="vendor",
                    python_paths=python_paths,
                    library_paths=library_paths,
                    isolated=candidate.isolated,
                ),
                warnings=tuple(skipped),
            )

        root_text = _expand_tags(cast(str, candidate.root))
        root = _resolve_inside(mod_root, root_text, f"{context}.root")
        if not root.exists():
            skipped.append(
                f"runtime profile '{profile.name}' skipped portable candidate: "
                f"directory does not exist: {root_text}"
            )
            continue
        if not root.is_dir():
            return RuntimeSelection(
                None,
                f"portable runtime root is not a directory: {root_text}",
                tuple(skipped),
            )

        try:
            runtime = _resolve_portable(profile.name, candidate, root)
        except ValueError as exc:
            return RuntimeSelection(None, str(exc), tuple(skipped))
        return RuntimeSelection(runtime, warnings=tuple(skipped))

    detail = "; ".join(skipped) or "profile has no usable candidates"
    return RuntimeSelection(
        None,
        f"runtime profile '{profile.name}' is unavailable: {detail}",
        tuple(skipped),
    )


def _read_candidate(value: object, context: str) -> RuntimeCandidate:
    candidate = _mapping(value, context)
    allowed = {
        "mode",
        "root",
        "python",
        "executable_paths",
        "library_paths",
        "isolated",
    }
    unknown = set(candidate) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")
    mode = candidate.get("mode")
    if mode not in ("host", "vendor", "portable"):
        raise ValueError(f"{context}.mode must be 'host', 'vendor' or 'portable'")

    root = candidate.get("root")
    python = candidate.get("python")
    if mode == "portable":
        if not isinstance(root, str) or not root:
            raise ValueError(f"{context}.root is required for portable mode")
        if python is not None and (not isinstance(python, str) or not python):
            raise ValueError(f"{context}.python must be a non-empty relative path")
    elif root is not None or python is not None:
        raise ValueError(f"{context}.root/python are only valid for portable mode")

    executable_paths = _string_list(
        candidate.get("executable_paths", ()), f"{context}.executable_paths"
    )
    library_paths = _string_list(
        candidate.get("library_paths", ()), f"{context}.library_paths"
    )
    if mode != "portable" and (executable_paths or library_paths):
        raise ValueError(
            f"{context}.executable_paths/library_paths are only valid for portable mode"
        )
    isolated = candidate.get("isolated", mode == "portable")
    if not isinstance(isolated, bool):
        raise ValueError(f"{context}.isolated must be a boolean")
    return RuntimeCandidate(
        mode=cast(RuntimeMode, mode),
        root=cast(str | None, root),
        python=cast(str | None, python),
        executable_paths=executable_paths,
        library_paths=library_paths,
        isolated=isolated,
    )


def _resolve_portable(
    name: str,
    candidate: RuntimeCandidate,
    root: Path,
) -> ResolvedRuntime:
    python_executable: Path | None = None
    python_home: Path | None = None
    if candidate.python is not None:
        python_executable = _resolve_inside(
            root, candidate.python, f"runtime profile '{name}' python"
        )
        if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
            raise ValueError(
                f"portable runtime '{name}' Python is not executable: "
                f"{candidate.python}"
            )
        default_home = root / "python"
        if default_home.is_dir() and default_home in python_executable.parents:
            python_home = default_home.resolve()

    executable_paths = _resolve_directories(
        root,
        candidate.executable_paths,
        defaults=("bin", "python/bin"),
        context=f"runtime profile '{name}' executable_paths",
    )
    library_paths = _resolve_directories(
        root,
        candidate.library_paths,
        defaults=("lib", "python/lib"),
        context=f"runtime profile '{name}' library_paths",
    )
    return ResolvedRuntime(
        name=name,
        mode="portable",
        root=root.resolve(),
        python_executable=python_executable,
        python_home=python_home,
        executable_paths=executable_paths,
        library_paths=library_paths,
        isolated=candidate.isolated,
    )


def _resolve_directories(
    root: Path,
    values: Sequence[str],
    *,
    defaults: Sequence[str],
    context: str,
) -> tuple[Path, ...]:
    if values:
        result: list[Path] = []
        for value in values:
            path = _resolve_inside(root, value, context)
            if not path.is_dir():
                raise ValueError(f"{context} directory does not exist: {value}")
            result.append(path)
        return tuple(dict.fromkeys(result))
    return tuple(
        path.resolve() for value in defaults if (path := root / value).is_dir()
    )


def _resolve_inside(root: Path, value: str, context: str) -> Path:
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"{context} must be a safe relative path")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"{context} escapes its runtime root")
    return candidate


def _expand_tags(value: str) -> str:
    try:
        return value.format(
            platform=runtime_platform_tag(),
            python_tag=runtime_python_tag(),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid runtime path template {value!r}: {exc}") from exc


def _prepend_path(
    environment: dict[str, str],
    name: str,
    paths: Sequence[Path],
) -> None:
    if not paths:
        return
    values = [str(path) for path in paths]
    current = environment.get(name)
    if current:
        values.extend(current.split(os.pathsep))
    environment[name] = os.pathsep.join(
        dict.fromkeys(value for value in values if value)
    )


def _string_list(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context}[{index}] must be a non-empty string")
        # Validate now; resolution later also rejects escaping symlinks.
        relative = Path(item)
        if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise ValueError(f"{context}[{index}] must be a safe relative path")
        result.append(item)
    return tuple(result)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed map")
    return cast(Mapping[str, object], value)


__all__ = [
    "ResolvedRuntime",
    "RuntimeCandidate",
    "RuntimeProfile",
    "RuntimeSelection",
    "read_runtime_profiles",
    "resolve_runtime_profile",
]
