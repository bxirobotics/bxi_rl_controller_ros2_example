#!/usr/bin/env python3
"""Microbenchmark the name-driven N-joint input and command hot paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = ROOT / "src/bxi_example_py_elf3"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from bxi_example_py_elf3.framework.inference import (  # noqa: E402
    JointInputBinding,
    PolicyJointContract,
)
from bxi_example_py_elf3.framework.joints import (  # noqa: E402
    JointCommandDefaults,
    JointCommandResolver,
    JointDefault,
    JointLayout,
    JointStateBuffer,
)
from bxi_example_py_elf3.framework.mod_api import MotorFrame  # noqa: E402


def _percentile(samples: np.ndarray, value: float) -> float:
    return float(np.percentile(samples, value))


def _measure(name: str, operation, warmup: int, iterations: int) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    samples = np.empty(iterations, dtype=np.int64)
    for index in range(iterations):
        started = time.perf_counter_ns()
        operation()
        samples[index] = time.perf_counter_ns() - started
    mean_ns = float(statistics.fmean(samples))
    return {
        "name": name,
        "p50_ns": _percentile(samples, 50),
        "p95_ns": _percentile(samples, 95),
        "p99_ns": _percentile(samples, 99),
        "mean_ns": mean_ns,
        "hz": 1_000_000_000.0 / mean_ns,
    }


def run(warmup: int, iterations: int) -> dict[str, object]:
    policy29 = JointLayout(tuple(f"joint_{index}" for index in range(29)), label="policy29")
    robot31 = JointLayout((*policy29.names, "tool_l", "tool_r"), label="robot31")
    state = JointStateBuffer(robot31, dtype=np.float32)
    state.position[:] = np.arange(31, dtype=np.float32)
    input_binding = JointInputBinding(PolicyJointContract(policy29, policy29))

    partial = MotorFrame.create(
        policy29,
        np.arange(29, dtype=np.float32),
        np.full(29, 20.0, dtype=np.float32),
        np.full(29, 0.5, dtype=np.float32),
    )
    defaults = JointCommandDefaults(
        {
            "tool_l": JointDefault(0.2, 5.0, 0.2),
            "tool_r": JointDefault(-0.2, 5.0, 0.2),
        }
    )
    resolver = JointCommandResolver(robot31, defaults)
    resolved = MotorFrame.empty(robot31)

    reordered_layout = JointLayout(tuple(reversed(robot31.names)), label="reordered31")
    reordered = MotorFrame.create(
        reordered_layout,
        np.arange(31, dtype=np.float32),
        np.full(31, 20.0, dtype=np.float32),
        np.full(31, 0.5, dtype=np.float32),
    )
    reordered_output = MotorFrame.empty(robot31)

    exact = MotorFrame.empty(robot31)

    cases = [
        _measure(
            "observation 31 -> policy 29",
            lambda: input_binding.bind(state.view),
            warmup,
            iterations,
        ),
        _measure(
            "command 29 -> robot 31 + defaults",
            lambda: resolver.resolve_into(partial, resolved),
            warmup,
            iterations,
        ),
        _measure(
            "command reordered 31 -> robot 31",
            lambda: resolver.resolve_into(reordered, reordered_output),
            warmup,
            iterations,
        ),
        _measure(
            "exact full-layout dispatch check",
            lambda: exact.layout.names == robot31.names,
            warmup,
            iterations,
        ),
    ]
    np.testing.assert_array_equal(input_binding.joints.position, partial.qpos)
    np.testing.assert_allclose(resolved.qpos[-2:], [0.2, -0.2])
    return {
        "schema": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "warmup": warmup,
        "iterations": iterations,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")

    report = run(args.warmup, args.iterations)
    print(f"Joint mapping benchmark ({args.iterations} iterations)")
    print(f"{'case':42} {'p50':>10} {'p95':>10} {'p99':>10} {'mean':>10} {'Hz':>12}")
    for case in report["cases"]:
        print(
            f"{case['name']:42} "
            f"{case['p50_ns'] / 1000:9.2f}us "
            f"{case['p95_ns'] / 1000:9.2f}us "
            f"{case['p99_ns'] / 1000:9.2f}us "
            f"{case['mean_ns'] / 1000:9.2f}us "
            f"{case['hz']:11.0f}"
        )
    if args.json is not None:
        output = args.json if args.json.is_absolute() else ROOT / args.json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
