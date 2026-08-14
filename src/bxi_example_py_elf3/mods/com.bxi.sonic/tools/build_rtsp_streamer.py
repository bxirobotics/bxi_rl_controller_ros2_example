#!/usr/bin/env python3
"""Build SONIC's Mod-local ROS/FFmpeg RTSP streamer for this platform."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def _platform_tag() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if machine in {"aarch64", "arm64"}:
        return "linux-aarch64"
    raise RuntimeError(f"unsupported architecture: {machine}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument(
        "--ros-prefix",
        type=Path,
        default=Path(os.environ.get("ROS_DISTRO_PREFIX", "/opt/ros/humble")),
    )
    parser.add_argument("--python", type=Path, default=Path("/usr/bin/python3"))
    args = parser.parse_args()
    mod_root = Path(__file__).resolve().parents[1]
    source = mod_root / "native" / "rtsp_streamer"
    output = mod_root / "bin" / _platform_tag()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sonic-rtsp-build-") as temporary:
        build = Path(temporary)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DCMAKE_BUILD_TYPE={args.build_type}",
                f"-DCMAKE_PREFIX_PATH={args.ros_prefix}",
                f"-DPython3_EXECUTABLE={args.python}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output}",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--parallel"],
            check=True,
        )
    binary = output / "head_camera_rtsp_node"
    if not binary.is_file():
        raise RuntimeError(f"build succeeded but executable is missing: {binary}")
    binary.chmod(0o755)
    print(binary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
