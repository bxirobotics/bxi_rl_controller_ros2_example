#!/usr/bin/env python3
"""Compare the current inference framework with a selected Git baseline.

Each case/version runs in a separate process so ONNX Runtime sessions, thread
pools and allocator caches cannot leak from one result into another.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import inspect
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
BACK_FLIP_ASSETS = PACKAGE_ROOT / "mods/com.bxi.back_flip/assets"

CASES = (
    "normal",
    "amp",
    "amp_noarm",
    "motion_legacy",
    "motion_isaac",
    "motion_v3",
    "depth_cached",
    "depth_fresh",
)


class _TimedBackend:
    """Benchmark-only proxy; production policies keep the original backend."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self.elapsed_ns = 0
        self.calls = 0

    def begin_sample(self) -> None:
        self.elapsed_ns = 0
        self.calls = 0

    def run(self, *args, **kwargs):
        started = time.perf_counter_ns()
        try:
            return self._backend.run(*args, **kwargs)
        finally:
            self.elapsed_ns += time.perf_counter_ns() - started
            self.calls += 1

    def __getattr__(self, name: str):
        return getattr(self._backend, name)


def _install_backend_timer(policy) -> _TimedBackend | None:
    for attribute in ("backend", "_backend"):
        backend = getattr(policy, attribute, None)
        if backend is None or not callable(getattr(backend, "run", None)):
            continue
        timer = _TimedBackend(backend)
        setattr(policy, attribute, timer)
        return timer
    return None


BASELINE_SOURCE_CANDIDATES = {
    "normal": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/normal.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/normal.py",
    ),
    "amp": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/amp.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/amp.py",
    ),
    "amp_noarm": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/amp.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/amp.py",
    ),
    "motion_legacy": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/beyondmimic.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/beyondmimic.py",
    ),
    "motion_isaac": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/beyondmimic.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/beyondmimic.py",
    ),
    "motion_v3": (
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/beyondmimic.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/inference/beyondmimic.py",
    ),
    "depth_cached": (
        "src/bxi_example_py_elf3/mods/com.bxi.normal_depth/depth.py",
        "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/depth.py",
        "src/bxi_example_py_elf3/mods/com.bxi.normal_depth/amp_depth.py",
    ),
    "depth_fresh": (
        "src/bxi_example_py_elf3/mods/com.bxi.normal_depth/depth.py",
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


def _load_current_depth_policy() -> type:
    source = DEPTH_ASSETS.parent / "depth.py"
    module_name = "_bxi_benchmark_normal_depth_policy"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load current depth policy: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.HumanoidGaitDepthPolicyIsaaclab


def _construct_baseline(cls, *args, **kwargs):
    """Construct a historical policy without assuming today's keywords.

    Pre-framework policies accepted only asset paths, while newer revisions
    also accept backend selection.  Filter optional benchmark keywords against
    the constructor stored in the selected Git revision.
    """
    parameters = inspect.signature(cls).parameters
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported = (
        kwargs
        if accepts_keywords
        else {key: value for key, value in kwargs.items() if key in parameters}
    )
    return cls(*args, **supported)


def _baseline_runtime():
    """Supply removed options only to historical code in the benchmark worker."""

    from bxi_example_py_elf3.framework.inference import InferenceRuntime

    runtime = InferenceRuntime()
    runtime.options = types.SimpleNamespace(
        backend="auto",
        warmup_runs=1,
        warn_on_fallback=True,
        monitor_enabled=False,
    )
    return runtime


def _inputs():
    return {
        # Distinct values make a joint-order regression visible in model output.
        "q": np.linspace(-0.14, 0.14, 29, dtype=np.float32),
        "dq": np.linspace(0.28, -0.28, 29, dtype=np.float32),
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
        from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

        return InferenceFrame(
            joints=JointStateView(ELF3_POLICY_JOINTS, q, dq),
            quat_wxyz=wxyz,
            angular_velocity=omega,
            command=command,
        )

    if version == "baseline":
        module = _load_baseline(case)
        if case == "normal":
            policy = _construct_baseline(
                module.NormalMotionPolicyMjlab,
                str((ASSETS / "model_normal.onnx").resolve()),
                runtime=_baseline_runtime(),
                backend="onnxruntime",
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
        if case in ("amp", "amp_noarm"):
            model_name = "amp_run.onnx" if case == "amp" else "withoutarm.onnx"
            policy = _construct_baseline(
                module.HumanoidGaitPolicyLiteIsaaclab,
                str((ASSETS / model_name).resolve()),
                runtime=_baseline_runtime(),
                backend="onnxruntime",
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                policy.reset_runtime_state(q, dq, wxyz, omega, command)
                return policy, lambda: infer(q, dq, wxyz, omega, command)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02)
        if case == "motion_legacy":
            policy = _construct_baseline(
                module.DanceMotionPolicyMjlab,
                str((ASSETS / "recover.npz").resolve()),
                str((ASSETS / "recover.onnx").resolve()),
                runtime=_baseline_runtime(),
                backend="onnxruntime",
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                return policy, lambda: infer(q, dq, wxyz, omega)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02, advance=False)
        if case == "motion_isaac":
            policy = _construct_baseline(
                module.DanceMotionPolicyGravityIsaaclab,
                str((BACK_FLIP_ASSETS / "back_flip.npz").resolve()),
                str((BACK_FLIP_ASSETS / "back_flip.onnx").resolve()),
                runtime=_baseline_runtime(),
                backend="onnxruntime",
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                return policy, lambda: infer(q, dq, wxyz, omega)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02, advance=False)
        if case == "motion_v3":
            policy = _construct_baseline(
                module.DanceMotionPolicyGravityIsaaclabV3,
                str((ASSETS / "shuishou.npz").resolve()),
                str((ASSETS / "shuishou.onnx").resolve()),
                runtime=_baseline_runtime(),
                backend="onnxruntime",
            )
            infer = getattr(policy, "inference_step", None)
            if callable(infer):
                return policy, lambda: infer(q, dq, wxyz, omega)
            frame = inference_frame_for(policy)
            policy.reset(frame)
            return policy, lambda: policy.step(frame, 0.02, advance=False)
        policy = _construct_baseline(
            module.HumanoidGaitDepthPolicyIsaaclab,
            str((DEPTH_ASSETS / "normal_depth.onnx").resolve()),
            runtime=_baseline_runtime(),
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
    )
    from bxi_example_py_elf3.policies import (
        DanceMotionPolicyGravityIsaaclab,
        DanceMotionPolicyGravityIsaaclabV3,
        DanceMotionPolicyMjlab,
        HumanoidGaitPolicyLiteIsaaclab,
        NormalMotionPolicyMjlab,
    )
    from bxi_example_py_elf3.framework.joints import JointStateView
    from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

    runtime = InferenceRuntime()
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
    if case in ("amp", "amp_noarm"):
        model_name = "amp_run.onnx" if case == "amp" else "withoutarm.onnx"
        policy = HumanoidGaitPolicyLiteIsaaclab(
            str((ASSETS / model_name).resolve()),
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
    if case == "motion_isaac":
        policy = DanceMotionPolicyGravityIsaaclab(
            str((BACK_FLIP_ASSETS / "back_flip.npz").resolve()),
            str((BACK_FLIP_ASSETS / "back_flip.onnx").resolve()),
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

    depth_policy_type = _load_current_depth_policy()
    policy = depth_policy_type(
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


def _semantic_output(output) -> np.ndarray:
    if hasattr(output, "joints"):
        joints = output.joints
        from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

        return np.asarray(
            [
                joints.position[joints.layout.index(name)]
                for name in ELF3_POLICY_JOINTS.names
            ],
            dtype=np.float32,
        )
    values = output[0] if isinstance(output, tuple) else output
    return np.asarray(values, dtype=np.float32).reshape(-1)[:29]


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
    first_semantic_output = None
    final_semantic_output = None
    for _ in range(repeats):
        policy, step = _make_runner(case, version)
        first_output = step()
        semantic_output = _semantic_output(first_output)
        if first_semantic_output is None:
            first_semantic_output = semantic_output.copy()
        checksum += float(semantic_output[0])
        for _ in range(warmup):
            step()
        samples = np.empty(iterations, dtype=np.int64)
        for index in range(iterations):
            started = time.perf_counter_ns()
            output = step()
            samples[index] = time.perf_counter_ns() - started

        backend_timer = _install_backend_timer(policy)
        profiled_samples = np.empty(iterations, dtype=np.int64)
        backend_samples = np.zeros(iterations, dtype=np.int64)
        backend_measured = backend_timer is not None
        for index in range(iterations):
            if backend_timer is not None:
                backend_timer.begin_sample()
            started = time.perf_counter_ns()
            step()
            profiled_samples[index] = time.perf_counter_ns() - started
            if backend_timer is not None:
                backend_samples[index] = backend_timer.elapsed_ns
                backend_measured = backend_measured and backend_timer.calls > 0
        policy_samples = np.maximum(profiled_samples - backend_samples, 0)
        if final_semantic_output is None:
            final_semantic_output = _semantic_output(output).copy()
        repeat_results.append(
            {
                "p50_us": float(np.percentile(samples, 50) / 1_000.0),
                "p95_us": float(np.percentile(samples, 95) / 1_000.0),
                "p99_us": float(np.percentile(samples, 99) / 1_000.0),
                "mean_us": float(np.mean(samples) / 1_000.0),
                "max_us": float(np.max(samples) / 1_000.0),
                "backend_measured": backend_measured,
                "backend_p50_us": float(
                    np.percentile(backend_samples, 50) / 1_000.0
                ),
                "backend_p99_us": float(
                    np.percentile(backend_samples, 99) / 1_000.0
                ),
                "policy_overhead_p50_us": float(
                    np.percentile(policy_samples, 50) / 1_000.0
                ),
                "policy_overhead_p99_us": float(
                    np.percentile(policy_samples, 99) / 1_000.0
                ),
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
        "backend_measured": all(
            item["backend_measured"] for item in repeat_results
        ),
        "backend_p50_us": statistics.median(
            item["backend_p50_us"] for item in repeat_results
        ),
        "backend_p99_us": statistics.median(
            item["backend_p99_us"] for item in repeat_results
        ),
        "policy_overhead_p50_us": statistics.median(
            item["policy_overhead_p50_us"] for item in repeat_results
        ),
        "policy_overhead_p99_us": statistics.median(
            item["policy_overhead_p99_us"] for item in repeat_results
        ),
        "traced_net_bytes": current - before,
        "traced_peak_bytes": peak - before,
        "checksum": checksum,
        "first_output": first_semantic_output.tolist(),
        "final_output": final_semantic_output.tolist(),
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
        command, cwd=ROOT, env=env, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"benchmark worker failed for {case}/{version} "
            f"(exit {completed.returncode}):\n{details}"
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
        if not (
            np.allclose(old["first_output"], new["first_output"], atol=1e-4)
            and np.allclose(old["final_output"], new["final_output"], atol=1e-4)
        ):
            mismatches.append(case)
        print(
            f"{case:<16} {old['p50_us']:>8.1f}µ {new['p50_us']:>8.1f}µ "
            f"{speedup:>7.2f}x {old['p99_us']:>8.1f}µ {new['p99_us']:>8.1f}µ "
            f"{old['traced_peak_bytes']:>9}B {new['traced_peak_bytes']:>9}B"
        )
    if mismatches:
        raise RuntimeError(f"semantic joint output mismatch: {', '.join(mismatches)}")

    print()
    print(
        f"{'current case':<16} {'total p50':>11} {'backend p50':>13} "
        f"{'policy p50':>12} {'backend p99':>13} {'policy p99':>12}"
    )
    print("-" * 84)
    for result in results:
        if result["version"] != "current":
            continue
        if not result["backend_measured"]:
            print(f"{result['case']:<16} backend timing unavailable")
            continue
        print(
            f"{result['case']:<16} "
            f"{result['p50_us']:>10.1f}µ "
            f"{result['backend_p50_us']:>12.1f}µ "
            f"{result['policy_overhead_p50_us']:>11.1f}µ "
            f"{result['backend_p99_us']:>12.1f}µ "
            f"{result['policy_overhead_p99_us']:>11.1f}µ"
        )


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
