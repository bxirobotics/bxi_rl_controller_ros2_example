#!/usr/bin/env python3
"""Launch any framework application with external inference-input capture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from calibration_capture import (
    ACTIVE_ENV,
    EVERY_ENV,
    MAX_ENV,
    QUEUE_ENV,
    ROOT_ENV,
    SKIP_ENV,
)


HERE = Path(__file__).resolve().parent
HOOK = HERE / "calibration_sitecustomize"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command while externally recording inputs to every inference "
            "backend opened through the BXI framework."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=10,
        help="ignore initial backend calls such as model warmup",
    )
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to launch, normally after --",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.every <= 0 or args.max_samples <= 0 or args.queue_size <= 0:
        raise SystemExit("--every, --max-samples and --queue-size must be positive")
    if args.skip_first < 0:
        raise SystemExit("--skip-first must be non-negative")
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    python_paths = [str(HOOK), str(HERE)]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment.update(
        {
            ACTIVE_ENV: "1",
            ROOT_ENV: str(output),
            EVERY_ENV: str(args.every),
            MAX_ENV: str(args.max_samples),
            SKIP_ENV: str(args.skip_first),
            QUEUE_ENV: str(args.queue_size),
            "PYTHONPATH": os.pathsep.join(python_paths),
        }
    )
    print(f"[calibration-tool] launching: {' '.join(command)}")
    print(f"[calibration-tool] output root: {output}")
    try:
        completed = subprocess.run(command, env=environment, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"cannot launch {command[0]!r}: {exc}") from exc
    except KeyboardInterrupt:
        return 130
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
