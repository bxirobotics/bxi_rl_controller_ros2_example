#!/usr/bin/env python3
"""Locate and exec MediaMTX with the Mod-owned RTSP configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import sys


def _platform_tag() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if machine in {"aarch64", "arm64"}:
        return "linux-aarch64"
    return f"linux-{machine}"


def _resolve_binary(mod_root: Path, explicit: str | None) -> Path | None:
    candidates: list[Path] = []
    configured = explicit or os.environ.get("SONIC_MEDIAMTX_BIN")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(mod_root / "runtime" / _platform_tag() / "mediamtx")
    host_binary = shutil.which("mediamtx")
    if host_binary:
        candidates.append(Path(host_binary))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary")
    parser.add_argument("--config", default="runtime/mediamtx.yml")
    args = parser.parse_args()
    mod_root = Path(__file__).resolve().parent
    binary = _resolve_binary(mod_root, args.binary)
    if binary is None:
        print(
            "MediaMTX executable was not found. Install it in PATH, set "
            "SONIC_MEDIAMTX_BIN, or place it at "
            f"{mod_root / 'runtime' / _platform_tag() / 'mediamtx'}",
            file=sys.stderr,
            flush=True,
        )
        return 78
    config = Path(args.config)
    if not config.is_absolute():
        config = mod_root / config
    config = config.resolve()
    if not config.is_file():
        print(f"MediaMTX config does not exist: {config}", file=sys.stderr)
        return 78
    print(f"Starting MediaMTX: {binary} {config}", flush=True)
    os.execv(str(binary), (str(binary), str(config)))
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
