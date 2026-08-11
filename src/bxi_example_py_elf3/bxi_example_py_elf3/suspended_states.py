"""Compatibility imports for the suspended motor-test Mod."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_path,
    )
except ModuleNotFoundError:
    PackageNotFoundError = Exception
    get_package_share_path = None


_MOD_ID = "com.bxi.suspended_tests"
_MODULE_NAME = "bxi_example_py_elf3._suspended_tests_mod_state"


def _candidate_state_files():
    source_file = (
        Path(__file__).resolve().parents[1] / "mods" / _MOD_ID / "state.py"
    )
    yield source_file
    if get_package_share_path is None:
        return
    try:
        yield (
            get_package_share_path("bxi_example_py_elf3")
            / "mods"
            / _MOD_ID
            / "state.py"
        )
    except PackageNotFoundError:
        return


def _load_state_module():
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    for state_file in _candidate_state_files():
        if not state_file.is_file():
            continue
        spec = spec_from_file_location(_MODULE_NAME, state_file)
        if spec is None or spec.loader is None:
            continue
        module = module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(_MODULE_NAME, None)
            raise
        return module
    raise ImportError("cannot find suspended tests Mod state.py")


_state_module = _load_state_module()
SuspendedRunningState = _state_module.SuspendedRunningState
SuspendedVibrationState = _state_module.SuspendedVibrationState
SuspendedLimbTestState = _state_module.SuspendedLimbTestState


def create_button_states():
    return (
        SuspendedRunningState(),
        SuspendedVibrationState(),
        SuspendedLimbTestState(),
    )


__all__ = [
    "create_button_states",
    "SuspendedLimbTestState",
    "SuspendedRunningState",
    "SuspendedVibrationState",
]
