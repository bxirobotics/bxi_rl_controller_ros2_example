"""Vendored runtime subset used by the SONIC PICO bridge.

The contained GEAR-SONIC files retain their upstream license and are kept in a
separate namespace so they do not shadow a system installation of
``gear_sonic``.  See ``THIRD_PARTY_NOTICES.md`` at the repository root.
"""

import sys
from pathlib import Path


_VENDOR_ROOT = str(Path(__file__).resolve().parent)
if _VENDOR_ROOT not in sys.path:
    sys.path.insert(0, _VENDOR_ROOT)
