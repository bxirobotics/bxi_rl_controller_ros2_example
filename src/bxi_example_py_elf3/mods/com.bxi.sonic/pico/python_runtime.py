"""Select a Python interpreter that can actually run a SONIC subprocess."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

from service_runtime import runtime_platform_tag, runtime_python_tag


_PROBE_TIMEOUT_SECONDS = 15.0
_SELECTED_PYTHON_ENV = "BXI_SONIC_SELECTED_PYTHON"
_BINDING_SOURCE_ENV = "BXI_SONIC_XRT_BINDING_SOURCE"
_ENVIRONMENT_BINDING = "environment"
_BUNDLED_BINDING = "bundled"
XRT_REQUIRED_CALLS = (
    "init",
    "close",
    "is_body_data_available",
    "get_body_joints_pose",
    "get_time_stamp_ns",
    "get_left_trigger",
    "get_right_trigger",
    "get_left_grip",
    "get_right_grip",
    "get_left_axis",
    "get_right_axis",
    "get_left_menu_button",
    "get_A_button",
    "get_B_button",
    "get_X_button",
    "get_Y_button",
)
_SCRIPT_BOOTSTRAP = (
    "import os, runpy, sys\n"
    "script = os.path.abspath(sys.argv[1])\n"
    "sys.argv = sys.argv[1:]\n"
    "script_dir = os.path.dirname(script)\n"
    "sys.path.insert(0, script_dir)\n"
    "runpy.run_path(script, run_name='__main__')\n"
)


@dataclass(frozen=True)
class PythonSelection:
    executable: Path
    binding_source: str


def _mod_root() -> Path:
    return Path(
        os.environ.get("BXI_MOD_ROOT", Path(__file__).absolute().parents[1])
    ).expanduser().absolute()


def vendor_python_directory() -> Path:
    """Return the packaged binary-extension directory for this platform."""
    return _mod_root() / "vendor" / "python" / runtime_python_tag()


def activate_vendor_python() -> Path:
    """Make packaged vendor bindings importable without PYTHONPATH."""
    directory = vendor_python_directory()
    if not directory.is_dir():
        raise RuntimeError(
            "bundled xrobotoolkit_sdk is unavailable for "
            f"{runtime_python_tag()}: {directory}"
        )
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)
    return directory


def _clean_python_environment() -> dict[str, str]:
    """Remove every parent-Python control variable from the vendor process."""
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTHON") or name == "__PYVENV_LAUNCHER__":
            environment.pop(name, None)
    # This is an environment assembled by SONIC, not inherited Python state.
    # Keep descendants from trying to create __pycache__ in a read-only Mod
    # installation.  The selected interpreter also receives ``-B`` because
    # its isolation flag (``-E``) intentionally ignores PYTHON* variables.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _resolve_python(value: str) -> Path | None:
    """Return an executable path without resolving its final symlink.

    A standard virtual environment normally exposes ``bin/python`` as a
    symlink to the base interpreter.  Python discovers the adjacent
    ``pyvenv.cfg`` from the *launcher path*, so canonicalising that symlink
    silently turns the virtual-environment interpreter back into the system
    interpreter and loses all of the environment's installed packages.
    """
    value = value.strip()
    if not value:
        return None
    if os.sep not in value:
        resolved = shutil.which(value)
        if resolved is None:
            return None
        candidate = Path(os.path.abspath(resolved))
        return (
            candidate
            if candidate.is_file() and os.access(candidate, os.X_OK)
            else None
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        # ``abspath`` normalises ``.`` and ``..`` while deliberately
        # preserving the final symlink that gives a venv its identity.
        candidate = Path(os.path.abspath(os.fspath(candidate)))
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _unique_paths(values: Iterable[str | Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        candidate = _resolve_python(str(value))
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return tuple(result)


def _common_candidates() -> tuple[Path, ...]:
    home = Path.home()
    mod_runtime = _mod_root() / ".runtime" / runtime_platform_tag() / "pico"
    values: list[str | Path] = [mod_runtime / "bin" / "python", sys.executable]
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(variable)
        if root:
            values.append(Path(root) / "bin" / "python")
    values.extend(
        (
            home / "miniconda3" / "bin" / "python",
            home / "anaconda3" / "bin" / "python",
            "python3",
            "python",
        )
    )
    return _unique_paths(values)


def _conda_candidates() -> tuple[Path, ...]:
    conda = shutil.which("conda")
    if conda is None:
        for candidate in (
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                conda = str(candidate)
                break
    if conda is None:
        return ()
    try:
        completed = subprocess.run(
            (conda, "env", "list", "--json"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        roots = json.loads(completed.stdout).get("envs", ())
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return ()
    if not isinstance(roots, list):
        return ()
    return _unique_paths(Path(root) / "bin" / "python" for root in roots)


def _probe(
    interpreter: Path,
    imports: Sequence[str],
    *,
    vendor_directory: Path | None = None,
) -> tuple[bool, str]:
    vendor_path = str(vendor_directory) if vendor_directory is not None else ""
    code = (
        "import importlib, sys\n"
        f"vendor_path = {vendor_path!r}\n"
        "if vendor_path:\n"
        "    sys.path.insert(0, vendor_path)\n"
        f"names = {tuple(imports)!r}\n"
        "modules = {}\n"
        "for name in names:\n"
        "    modules[name] = importlib.import_module(name)\n"
        "if 'xrobotoolkit_sdk' in modules:\n"
        f"    required = {XRT_REQUIRED_CALLS!r}\n"
        "    missing = [name for name in required "
        "if not callable(getattr(modules['xrobotoolkit_sdk'], name, None))]\n"
        "    if missing:\n"
        "        raise RuntimeError('missing callable API: ' + ', '.join(missing))\n"
    )
    try:
        completed = subprocess.run(
            (str(interpreter), "-B", "-E", "-s", "-c", code),
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            env=_clean_python_environment(),
        )
    except subprocess.TimeoutExpired:
        return False, f"dependency probe timed out after {_PROBE_TIMEOUT_SECONDS:.0f}s"
    except OSError as exc:
        return False, str(exc)
    if completed.returncode == 0:
        return True, ""
    output = (completed.stderr or completed.stdout).strip().splitlines()
    return False, output[-1] if output else f"probe exited {completed.returncode}"


def select_python(
    component: str,
    imports: Sequence[str],
) -> PythonSelection:
    """Select a user environment first, then retry with the bundled binding."""
    explicit = os.environ.get("SONIC_PICO_PYTHON", "").strip()
    if explicit:
        candidate = _resolve_python(explicit)
        if candidate is None:
            raise RuntimeError(
                f"SONIC_PICO_PYTHON is not an executable Python: {explicit}"
            )
        candidates = (candidate,)
    else:
        candidates = _common_candidates()

    failures: list[str] = []
    checked_candidates: set[tuple[Path, str]] = set()

    def try_candidates(
        values: Sequence[Path],
        binding_source: str,
    ) -> PythonSelection | None:
        vendor_directory = (
            vendor_python_directory()
            if binding_source == _BUNDLED_BINDING
            else None
        )
        for candidate in values:
            key = (candidate, binding_source)
            if key in checked_candidates:
                continue
            checked_candidates.add(key)
            available, reason = _probe(
                candidate,
                imports,
                vendor_directory=vendor_directory,
            )
            if available:
                print(
                    f"[sonic-python] {component}: selected {candidate}; "
                    f"xrobotoolkit_sdk={binding_source}",
                    flush=True,
                )
                return PythonSelection(candidate, binding_source)
            failures.append(
                f"  - {candidate} [{binding_source} binding]: {reason}"
            )
        return None

    # Search every user-managed environment before considering the bundled
    # extension. This prevents a working pip/conda installation from being
    # shadowed merely because an earlier interpreter can use the fallback.
    selected = try_candidates(candidates, _ENVIRONMENT_BINDING)
    if selected is not None:
        return selected
    conda_candidates: tuple[Path, ...] = ()
    if not explicit:
        conda_candidates = _conda_candidates()
        selected = try_candidates(conda_candidates, _ENVIRONMENT_BINDING)
        if selected is not None:
            return selected

    bundled_directory = vendor_python_directory()
    if bundled_directory.is_dir():
        selected = try_candidates(candidates, _BUNDLED_BINDING)
        if selected is not None:
            return selected
        if not explicit:
            selected = try_candidates(conda_candidates, _BUNDLED_BINDING)
            if selected is not None:
                return selected
    else:
        failures.append(
            "  - bundled binding unavailable for target "
            f"{runtime_python_tag()}: {bundled_directory}"
        )

    checked = "\n".join(failures) if failures else "  - no Python candidates found"
    requirements = Path(__file__).resolve().parents[1] / "requirements-pico.txt"
    raise RuntimeError(
        f"no Python interpreter can run SONIC {component}; required imports: "
        f"{', '.join(imports)}\nchecked:\n{checked}\n"
        "Install the dependencies into one environment:\n"
        f"  <python> -m pip install -r {requirements}\n"
        "Install xrobotoolkit_sdk into the selected environment, or use a target "
        "for which the Mod supplies the fallback binding at "
        f"{vendor_python_directory()}. Then set SONIC_PICO_PYTHON=<python> or "
        "restart the Mod for auto-discovery."
    )


def reexec_if_needed(component: str, imports: Sequence[str]) -> None:
    current = _resolve_python(sys.executable)
    if current is None:
        current = Path(os.path.abspath(sys.executable))
    previously_selected = _resolve_python(os.environ.get(_SELECTED_PYTHON_ENV, ""))
    if previously_selected == current:
        if os.environ.get(_BINDING_SOURCE_ENV) == _BUNDLED_BINDING:
            activate_vendor_python()
        return
    selected = select_python(component, imports)
    environment = _clean_python_environment()
    environment[_SELECTED_PYTHON_ENV] = str(selected.executable)
    environment[_BINDING_SOURCE_ENV] = selected.binding_source
    os.execve(
        str(selected.executable),
        (
            str(selected.executable),
            "-u",
            "-B",
            "-E",
            "-s",
            "-c",
            _SCRIPT_BOOTSTRAP,
            str(Path(sys.argv[0]).resolve()),
            *sys.argv[1:],
        ),
        environment,
    )


__all__ = [
    "PythonSelection",
    "XRT_REQUIRED_CALLS",
    "activate_vendor_python",
    "reexec_if_needed",
    "select_python",
    "vendor_python_directory",
]
