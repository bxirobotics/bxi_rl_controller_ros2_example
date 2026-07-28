#!/usr/bin/env python3
"""Compare the current inference framework with the repository HEAD baseline.

Each case/version runs in a separate process so ONNX Runtime sessions, thread
pools and allocator caches cannot leak from one result into another.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src/bxi_example_py_elf3"
ASSETS = PACKAGE_ROOT / "mods/com.bxi.basic_actions/assets"
DEPTH_ASSETS = PACKAGE_ROOT / "mods/com.bxi.normal_depth/assets"

CASES = ("normal", "amp", "motion_legacy", "motion_v3", "depth_cached", "depth_fresh")
BASELINE_SOURCE_CANDIDATES = {
    "normal": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/normal.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/normal.py",
    ),
    "amp": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/amp.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/amp.py",
    ),
    "motion_legacy": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/beyondmimic.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/beyondmimic.py",
    ),
    "motion_v3": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/beyondmimic.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/beyondmimic.py",
    ),
    "depth_cached": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/depth.py",
        "src/bxi_example_py_elf3/mods/com.bxi.normal_depth/amp_depth.py",
    ),
    "depth_fresh": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/depth.py",
        "src/bxi_example_py_elf3/mods/com.bxi.normal_depth/amp_depth.py",
    ),
}


def _load_baseline(case: str) -> types.ModuleType:
    # Baseline source uses its historical flat import paths. Alias those names
    # only inside this isolated benchmark worker; the installed package keeps
    # no compatibility modules.
    for old_name, current_name in (
        ("history", "history"),
        ("model", "model"),
        ("runtime", "runtime"),
    ):
        qualified = f"bxi_example_py_elf3.inference.{old_name}"
        sys.modules.setdefault(
            qualified,
            importlib.import_module(
                f"bxi_example_py_elf3.framework.inference.{current_name}"
            ),
        )
    sys.modules.setdefault(
        "bxi_example_py_elf3.mod_api",
        importlib.import_module("bxi_example_py_elf3.framework.mod_api"),
    )
    sys.modules.setdefault(
        "bxi_example_py_elf3.mod_api.geometry",
        importlib.import_module("bxi_example_py_elf3.framework.mod_api.geometry"),
    )
    sys.modules.setdefault(
        "bxi_example_py_elf3.joints",
        importlib.import_module("bxi_example_py_elf3.policies.joints"),
    )
    baseline_ref = os.environ.get("INFERENCE_BENCHMARK_BASELINE_REF", "HEAD")
    source_path = _find_baseline_source(case, baseline_ref)
    result = subprocess.run(
        ["git", "show", f"{baseline_ref}:{source_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if "/bxi_example_py_elf3/inference/" in source_path:
        module_name = f"bxi_example_py_elf3.framework.inference._baseline_{case}"
        package = "bxi_example_py_elf3.framework.inference"
    elif "/bxi_example_py_elf3/policies/" in source_path:
        module_name = f"bxi_example_py_elf3.policies._baseline_{case}"
        package = "bxi_example_py_elf3.policies"
    else:
        module_name = f"_baseline_{case}"
        package = ""
    module = types.ModuleType(module_name)
    module.__file__ = str(ROOT / source_path)
    module.__package__ = package
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    return module


def _find_baseline_source(case: str, baseline_ref: str) -> str:
    for source_path in BASELINE_SOURCE_CANDIDATES[case]:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{baseline_ref}:{source_path}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if exists.returncode == 0:
            return source_path
    raise RuntimeError(
        f"cannot find baseline source for case '{case}' at {baseline_ref}"
    )


def _inputs():
    return {
        "q": np.zeros(29, dtype=np.float32),
        "dq": np.zeros(29, dtype=np.float32),
        "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "omega": np.zeros(3, dtype=np.float32),
        "command": np.array([0.2, -0.1, 0.15], dtype=np.float32),
    }


def _make_runner(case: str, version: str):
    state = _inputs()
    q, dq = state["q"], state["dq"]
    wxyz, xyzw = state["quat_wxyz"], state["quat_xyzw"]
    omega, command = state["omega"], state["command"]

    def inference_frame_for(policy):
        from bxi_example_py_elf3.framework.inference import InferenceFrame
        from bxi_example_py_elf3.framework.joints import JointStateView

        layout = policy.joint_contract.observation
        return InferenceFrame(
            joints=JointStateView(layout, q, dq),
            quat_wxyz=wxyz,
            angular_velocity=omega,
            command=command,
        )

    if version == "baseline":
        module = _load_baseline(case)
        if case == "normal":
            policy = module.NormalMotionPolicyMjlab(
                str((ASSETS / "model_normal.onnx").resolve()), backend="onnxruntime"
            )
            infer = getattr(policy, "infer_step", None)
            if callable(infer):
                action = getattr(policy, "action", getattr(policy, "_action", None))
                if action is not None:
                    action.fill(0.0)
                return policy, lambda: infer(q, dq, xyzw, omega, command)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02, advance=False)
        if case == "amp":
            policy = module.HumanoidGaitPolicyLiteIsaaclab(
                str((ASSETS / "amp_run.onnx").resolve()), backend="onnxruntime"
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                policy.reset_runtime_state(q, dq, wxyz, omega, command)
                return policy, lambda: infer(q, dq, wxyz, omega, command)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02)
        if case == "motion_legacy":
            policy = module.DanceMotionPolicyMjlab(
                str((ASSETS / "recover.npz").resolve()),
                str((ASSETS / "recover.onnx").resolve()),
                backend="onnxruntime",
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                return policy, lambda: infer(q, dq, wxyz, omega)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02, advance=False)
        if case == "motion_v3":
            policy = module.DanceMotionPolicyGravityIsaaclabV3(
                str((ASSETS / "shuishou.npz").resolve()),
                str((ASSETS / "shuishou.onnx").resolve()),
                backend="onnxruntime",
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                return policy, lambda: infer(q, dq, wxyz, omega)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02, advance=False)
        policy = module.HumanoidGaitDepthPolicyIsaaclab(
            str((DEPTH_ASSETS / "normal_depth.onnx").resolve()),
            backend="onnxruntime",
        )
        depth = np.ones((policy.depth_h, policy.depth_w), dtype=np.float32)
        frame = [0]
        infer = getattr(policy, "inference_step", None)
        if callable(infer):
            if case == "depth_cached":
                return policy, lambda: infer(
                    q, dq, wxyz, omega, command, depth, depth_frame_id=1
                )

            def baseline_depth_fresh():
                frame[0] += 1
                return infer(
                    q,
                    dq,
                    wxyz,
                    omega,
                    command,
                    depth,
                    depth_frame_id=frame[0],
                )

            return policy, baseline_depth_fresh

        inference_frame = inference_frame_for(policy)
        inference_frame.depth = depth
        policy.reset(inference_frame)
        if case == "depth_cached":
            inference_frame.depth_frame_id = 1
            return policy, lambda: policy.step(inference_frame, 0.02)

        def baseline_depth_fresh():
            frame[0] += 1
            inference_frame.depth_frame_id = frame[0]
            return policy.step(inference_frame, 0.02)

        return policy, baseline_depth_fresh

    from bxi_example_py_elf3.framework.inference import (
        InferenceFrame,
        InferenceRuntime,
        RuntimeOptions,
    )
    from bxi_example_py_elf3.policies import (
        DanceMotionPolicyGravityIsaaclabV3,
        DanceMotionPolicyMjlab,
        HumanoidGaitDepthPolicyIsaaclab,
        HumanoidGaitPolicyLiteIsaaclab,
        NormalMotionPolicyMjlab,
    )
    from bxi_example_py_elf3.framework.joints import JointStateView
    from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

    runtime = InferenceRuntime(options=RuntimeOptions(monitor_enabled=False))
    inference_frame = InferenceFrame(
        joints=JointStateView(ELF3_POLICY_JOINTS, q, dq),
        quat_wxyz=wxyz,
        angular_velocity=omega,
        command=command,
    )
    if case == "normal":
        policy = NormalMotionPolicyMjlab(
            str((ASSETS / "model_normal.onnx").resolve()),
            runtime=runtime,
            backend="onnxruntime",
        )
        policy.reset(inference_frame)
        return policy, lambda: policy.step(inference_frame, 0.02, advance=False)
    if case == "amp":
        policy = HumanoidGaitPolicyLiteIsaaclab(
            str((ASSETS / "amp_run.onnx").resolve()),
            runtime=runtime,
            backend="onnxruntime",
        )
        policy.reset(inference_frame)
        return policy, lambda: policy.step(inference_frame, 0.02)
    if case == "motion_legacy":
        policy = DanceMotionPolicyMjlab(
            str((ASSETS / "recover.npz").resolve()),
            str((ASSETS / "recover.onnx").resolve()),
            runtime=runtime,
            backend="onnxruntime",
        )
        policy.reset(inference_frame)
        return policy, lambda: policy.step(inference_frame, 0.02, advance=False)
    if case == "motion_v3":
        policy = DanceMotionPolicyGravityIsaaclabV3(
            str((ASSETS / "shuishou.npz").resolve()),
            str((ASSETS / "shuishou.onnx").resolve()),
            runtime=runtime,
            backend="onnxruntime",
        )
        policy.reset(inference_frame)
        return policy, lambda: policy.step(inference_frame, 0.02, advance=False)

    policy = HumanoidGaitDepthPolicyIsaaclab(
        str((DEPTH_ASSETS / "normal_depth.onnx").resolve()),
        runtime=runtime,
        backend="onnxruntime",
    )
    depth = np.ones((policy.depth_h, policy.depth_w), dtype=np.float32)
    frame = [0]
    inference_frame.depth = depth
    policy.reset(inference_frame)
    if case == "depth_cached":
        inference_frame.depth_frame_id = 1
        return policy, lambda: policy.step(inference_frame, 0.02)

    def current_depth_fresh():
        frame[0] += 1
        inference_frame.depth_frame_id = frame[0]
        return policy.step(inference_frame, 0.02)

    return policy, current_depth_fresh


def _close(policy) -> None:
    close = getattr(policy, "close", None)
    if callable(close):
        close()


def _worker(
    case: str,
    version: str,
    warmup: int,
    iterations: int,
    repeats: int,
    alloc_iterations: int,
):
    repeat_results = []
    checksum = 0.0
    for _ in range(repeats):
        policy, step = _make_runner(case, version)
        first_output = step()
        if hasattr(first_output, "joints"):
            checksum_array = first_output.joints.position
        else:
            checksum_array = (
                first_output[0] if isinstance(first_output, tuple) else first_output
            )
        checksum += float(np.asarray(checksum_array).reshape(-1)[0])
        for _ in range(warmup):
            step()
        samples = np.empty(iterations, dtype=np.int64)
        for index in range(iterations):
            started = time.perf_counter_ns()
            output = step()
            samples[index] = time.perf_counter_ns() - started
        repeat_results.append(
            {
                "p50_us": float(np.percentile(samples, 50) / 1_000.0),
                "p95_us": float(np.percentile(samples, 95) / 1_000.0),
                "p99_us": float(np.percentile(samples, 99) / 1_000.0),
                "mean_us": float(np.mean(samples) / 1_000.0),
                "max_us": float(np.max(samples) / 1_000.0),
            }
        )
        _close(policy)

    policy, step = _make_runner(case, version)
    for _ in range(warmup):
        step()
    tracemalloc.start()
    tracemalloc.reset_peak()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(alloc_iterations):
        step()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _close(policy)

    result = {
        "case": case,
        "version": version,
        "p50_us": statistics.median(item["p50_us"] for item in repeat_results),
        "p95_us": statistics.median(item["p95_us"] for item in repeat_results),
        "p99_us": statistics.median(item["p99_us"] for item in repeat_results),
        "mean_us": statistics.median(item["mean_us"] for item in repeat_results),
        "max_us": statistics.median(item["max_us"] for item in repeat_results),
        "traced_net_bytes": current - before,
        "traced_peak_bytes": peak - before,
        "checksum": checksum,
    }
    sys.__stdout__.write(json.dumps(result) + "\n")


def _run_child(args, case: str, version: str):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        case,
        "--version",
        version,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--repeats",
        str(args.repeats),
        "--alloc-iterations",
        str(args.alloc_iterations),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    env.setdefault("PYTHONHASHSEED", "0")
    env["INFERENCE_BENCHMARK_BASELINE_REF"] = args.baseline_ref
    completed = subprocess.run(
        command, cwd=ROOT, env=env, check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _print_results(results):
    mismatches = []
    print(
        f"{'case':<16} {'old p50':>9} {'new p50':>9} {'speedup':>8} "
        f"{'old p99':>9} {'new p99':>9} {'old peak':>10} {'new peak':>10}"
    )
    print("-" * 100)
    for case in dict.fromkeys(item["case"] for item in results):
        old = next(
            item
            for item in results
            if item["case"] == case and item["version"] == "baseline"
        )
        new = next(
            item
            for item in results
            if item["case"] == case and item["version"] == "current"
        )
        speedup = old["p50_us"] / new["p50_us"]
        if abs(old["checksum"] - new["checksum"]) > 1e-4:
            mismatches.append(case)
        print(
            f"{case:<16} {old['p50_us']:>8.1f}µ {new['p50_us']:>8.1f}µ "
            f"{speedup:>7.2f}x {old['p99_us']:>8.1f}µ {new['p99_us']:>8.1f}µ "
            f"{old['traced_peak_bytes']:>9}B {new['traced_peak_bytes']:>9}B"
        )
    if mismatches:
        raise RuntimeError(f"output checksum mismatch: {', '.join(mismatches)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--alloc-iterations", type=int, default=1000)
    parser.add_argument(
        "--baseline-ref",
        default="HEAD",
        help="Git revision containing the implementation used as the baseline",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--version", choices=("baseline", "current"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.worker:
        with contextlib.redirect_stdout(io.StringIO()):
            _worker(
                args.case,
                args.version,
                args.warmup,
                args.iterations,
                args.repeats,
                args.alloc_iterations,
            )
        return

    results = []
    for case in args.cases:
        for version in ("baseline", "current"):
            print(f"running {case} / {version}...", flush=True)
            results.append(_run_child(args, case, version))
    _print_results(results)
    print("\nJSON:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
