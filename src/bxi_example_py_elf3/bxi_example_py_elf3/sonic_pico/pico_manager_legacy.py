"""Run the packaged GEAR-SONIC PICO manager compatibility implementation.

Only the headless runtime dependency set required by ELF3 is vendored.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _vendor_root() -> Path:
    root = Path(__file__).resolve().parent / "vendor"
    manager = root / "gear_sonic" / "scripts" / "pico_manager_thread_server.py"
    if not manager.is_file():
        raise FileNotFoundError(f"Packaged PICO manager is missing: {manager}")
    return root


def main() -> int:
    vendor_root = _vendor_root()
    legacy_script = vendor_root / "gear_sonic" / "scripts" / "pico_manager_thread_server.py"
    sys.path.insert(0, str(vendor_root))
    sys.argv[0] = str(legacy_script)
    runpy.run_path(str(legacy_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
