"""Launch the packaged GEAR-SONIC PICO manager in an isolated selected Python.

User-installed XR dependencies are preferred; packaged binaries are fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

from runtime_config import PICO_PORT
from python_runtime import XRT_REQUIRED_CALLS, reexec_if_needed
from service_runtime import prepare_service_environment, resolve_service_root


MANAGER_IMPORTS = (
    "msgpack",
    "numpy",
    "scipy",
    "zmq",
    "pinocchio",
    "xrobotoolkit_sdk",
)
CONFIG_ERROR_EXIT_CODE = getattr(os, "EX_CONFIG", 78)
def _vendor_root() -> Path:
    root = Path(__file__).resolve().parent
    manager = root / "gear_sonic" / "scripts" / "pico_manager_thread_server.py"
    if not manager.is_file():
        raise FileNotFoundError(f"Packaged PICO manager is missing: {manager}")
    return root


def _validate_manager_runtime() -> None:
    import xrobotoolkit_sdk as xrt

    missing = tuple(
        name for name in XRT_REQUIRED_CALLS if not callable(getattr(xrt, name, None))
    )
    if missing:
        raise RuntimeError(
            "xrobotoolkit_sdk is incompatible; missing callable API: "
            + ", ".join(missing)
        )
    service_root = resolve_service_root()
    service_executable = service_root / "RoboticsServiceProcess"
    if not service_executable.is_file() or not os.access(service_executable, os.X_OK):
        raise RuntimeError(
            "RoboticsServiceProcess is not executable: "
            f"{service_executable}; set SONIC_XRT_SERVICE_DIR to its directory"
        )


def main() -> int:
    try:
        prepare_service_environment()
        reexec_if_needed("pico_manager", MANAGER_IMPORTS)
        vendor_root = _vendor_root()
        _validate_manager_runtime()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[sonic-pico] configuration error: {exc}", file=sys.stderr, flush=True)
        return CONFIG_ERROR_EXIT_CODE
    manager_script = (
        vendor_root / "gear_sonic" / "scripts" / "pico_manager_thread_server.py"
    )
    sys.path.insert(0, str(vendor_root))
    sys.argv[0] = str(manager_script)
    if not any(arg == "--port" or arg.startswith("--port=") for arg in sys.argv[1:]):
        sys.argv.extend(("--port", str(PICO_PORT)))
    runpy.run_path(str(manager_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
