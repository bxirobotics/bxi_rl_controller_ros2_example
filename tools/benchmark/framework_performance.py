#!/usr/bin/env python3
"""Measure the real framework cycle boundaries without instrumenting the runtime.

The benchmark creates a temporary API-4 Mod with one allocation-free hold state,
then invokes the existing ``RobotControlRuntime._run_control_cycle`` boundary.
Its fake platform timestamps calls at the outside of the production runtime, so
no benchmark fields, hooks or conditionals are added to framework code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Sequence, cast

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = ROOT / "src/bxi_example_py_elf3"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

try:
    from bxi_example_py_elf3.framework.joints import (
        JointCommandDefaults,
        JointLayout,
        JointStateBuffer,
    )
    from bxi_example_py_elf3.framework.platform import (
        CpuAffinityPlan,
        CpuAffinityRole,
        RobotControlRuntime,
        RobotObservation,
    )
    from bxi_example_py_elf3.framework.platform.cpu_affinity import (
        bootstrap_process_scheduling,
        format_cpu_set,
    )
    from bxi_example_py_elf3.framework.mod_api import ResourceKey
    from bxi_example_py_elf3.framework.runtime.logging import (
        SubprocessLogRouter,
        SubprocessLoggingConfig,
    )
except ModuleNotFoundError as exc:
    if exc.name == "ament_index_python":
        raise SystemExit(
            "ROS 2 Python packages are unavailable. Source the ROS environment first, "
            "for example: source /opt/ros/humble/setup.bash"
        ) from exc
    raise


_PLUGIN = '''\
from __future__ import annotations

import numpy as np

from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    MotorFrame,
    RobotControlState,
)


class BenchmarkHoldState(RobotControlState):
    def on_update(self, ctx, dt):
        frame = self._motor_frame_buffer
        if frame is None or frame.layout != ctx.robot_layout:
            frame = MotorFrame.empty(ctx.robot_layout)
            frame.kp.fill(20.0)
            frame.kd.fill(0.5)
            self._motor_frame_buffer = frame
        np.copyto(frame.qpos, ctx.robot_joints.position, casting="same_kind")
        ctx.set_motor_target(frame)


def create_mod(context: ModLoadContext) -> ModDefinition:
    return ModDefinition(
        state_factories={
            "hold": lambda state: BenchmarkHoldState(state.name, state.state_id),
        }
    )
'''


_MANIFEST = '''\
schema: 1
id: com.bxi.framework_benchmark
name: Framework benchmark fixture
version: 1.0.0
api: ">=4,<5"
enable: true
entrypoint: plugin:create_mod
visibility: public
requires: []
conflicts: []
python_exports: []
runtime_requirements:
  python: []
  ros: []
  system: []
events: {}
states:
  hold:
    manifest:
      label: Framework benchmark
      group: Diagnostics
routes: []
actions: []
'''


class _NullLogger:
    def __init__(self, name: str = "framework_performance") -> None:
        self.name = name

    def get_child(self, suffix: str) -> "_NullLogger":
        return _NullLogger(f"{self.name}.{suffix}")

    def set_level(self, _level: int) -> None:
        pass

    def debug(self, _message: str) -> None:
        pass

    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass

    def fatal(self, _message: str) -> None:
        pass


class _FakeNode:
    def __init__(self) -> None:
        self._logger = _NullLogger()

    def get_logger(self) -> _NullLogger:
        return self._logger


class _BenchmarkPlatform:
    def __init__(self, dof: int) -> None:
        names = tuple(f"joint_{index}" for index in range(dof))
        self.layout = JointLayout(names, label=f"benchmark-{dof}")
        self.joints = JointStateBuffer(self.layout, dtype=np.float64)
        self.quat_xyzw = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        self.quat_wxyz = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float32)
        self.observation = RobotObservation(
            joints=self.joints.view,
            quat_xyzw=self.quat_xyzw,
            quat_wxyz=self.quat_wxyz,
            omega=self.omega,
            raw_cmd_vel=self.command,
        )
        self._snapshot = (self.observation, ())
        self.published_position = np.zeros(dof, dtype=np.float64)
        self.published_kp = np.zeros(dof, dtype=np.float64)
        self.published_kd = np.zeros(dof, dtype=np.float64)
        self.snapshot_started_ns = 0
        self.snapshot_finished_ns = 0
        self.publish_started_ns = 0
        self.publish_finished_ns = 0

    def reset_timings(self) -> None:
        self.snapshot_started_ns = 0
        self.snapshot_finished_ns = 0
        self.publish_started_ns = 0
        self.publish_finished_ns = 0

    def startup_step(self, _now: float) -> bool:
        return True

    def snapshot_control_inputs(self):
        self.snapshot_started_ns = time.perf_counter_ns()
        try:
            self.joints.view.timestamp_ns = time.monotonic_ns()
            return self._snapshot
        finally:
            self.snapshot_finished_ns = time.perf_counter_ns()

    def publish_motor_frame(self, frame) -> None:
        self.publish_started_ns = time.perf_counter_ns()
        try:
            np.copyto(self.published_position, frame.qpos, casting="same_kind")
            np.copyto(self.published_kp, frame.kp, casting="same_kind")
            np.copyto(self.published_kd, frame.kd, casting="same_kind")
        finally:
            self.publish_finished_ns = time.perf_counter_ns()


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _summary(values_ns: Sequence[int]) -> dict[str, float]:
    values = np.asarray(values_ns, dtype=np.float64) / 1_000.0
    mean_us = float(statistics.fmean(values))
    return {
        "p50_us": _percentile(values, 50),
        "p95_us": _percentile(values, 95),
        "p99_us": _percentile(values, 99),
        "mean_us": mean_us,
        "max_us": float(np.max(values)),
        "hz": 1_000_000.0 / mean_us if mean_us > 0.0 else float("inf"),
    }


def _write_fixture(root: Path) -> Path:
    mod_root = root / "com.bxi.framework_benchmark"
    mod_root.mkdir()
    (mod_root / "plugin.py").write_text(_PLUGIN, encoding="utf-8")
    (mod_root / "mod.yaml").write_text(_MANIFEST, encoding="utf-8")
    return root


def _runtime_config() -> dict[str, object]:
    return {
        "initial_state": "com.bxi.framework_benchmark/hold",
        "mod_paths": [],
        "logging": {
            "default_level": "error",
            "levels": {},
            "subprocess": {
                "max_line_bytes": 16_384,
                "max_lines_per_sec": 200,
            },
        },
        "control_runtime": {
            "period_sec": 0.02,
            "compute_budget_sec": 0.01,
            "deadline_tolerance_sec": 0.001,
            "maintenance_hz": 5.0,
            "maintenance_guard_sec": 0.005,
            "statistics_interval_sec": 60.0,
            "deadline_warning_interval_sec": 1.0,
            "spin_wait_us": -1,
            "cpu_affinity": "control",
            "realtime_priority": 0,
            "python_switch_interval_sec": 0.001,
        },
        "default_transition": "instant",
        "transition_profiles": {
            "instant": {"type": "instant"},
        },
    }


def _measure_cycles(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    dof: int,
) -> dict[str, object]:
    repeat_results: list[dict[str, object]] = []
    setup_samples: list[int] = []
    cpu_affinity_plan = CpuAffinityPlan.discover()
    with tempfile.TemporaryDirectory(prefix="bxi-framework-benchmark-") as temp:
        mod_root = _write_fixture(Path(temp))
        for _repeat in range(repeats):
            benchmark_platform = _BenchmarkPlatform(dof)
            setup_started = time.perf_counter_ns()
            runtime = RobotControlRuntime(
                _runtime_config(),
                built_in_mod_root=mod_root,
                command_defaults=JointCommandDefaults(),
                ros_node=_FakeNode(),
                platform=benchmark_platform,
                cpu_affinity_plan=cpu_affinity_plan,
            )
            setup_samples.append(time.perf_counter_ns() - setup_started)
            try:
                for _ in range(warmup):
                    runtime._run_control_cycle()

                snapshot_ns: list[int] = []
                framework_ns: list[int] = []
                publish_ns: list[int] = []
                accounted_ns: list[int] = []
                wall_ns: list[int] = []
                for _ in range(iterations):
                    benchmark_platform.reset_timings()
                    started_ns = time.perf_counter_ns()
                    runtime._run_control_cycle()
                    finished_ns = time.perf_counter_ns()
                    snapshot = (
                        benchmark_platform.snapshot_finished_ns
                        - benchmark_platform.snapshot_started_ns
                    )
                    framework = (
                        benchmark_platform.publish_started_ns
                        - benchmark_platform.snapshot_finished_ns
                    )
                    publish = (
                        benchmark_platform.publish_finished_ns
                        - benchmark_platform.publish_started_ns
                    )
                    if min(snapshot, framework, publish) < 0:
                        raise RuntimeError("incomplete framework timing boundary")
                    snapshot_ns.append(snapshot)
                    framework_ns.append(framework)
                    publish_ns.append(publish)
                    accounted_ns.append(snapshot + framework + publish)
                    wall_ns.append(finished_ns - started_ns)
                repeat_results.append(
                    {
                        "snapshot": _summary(snapshot_ns),
                        "framework": _summary(framework_ns),
                        "publish": _summary(publish_ns),
                        "accounted_cycle": _summary(accounted_ns),
                        "wall_cycle": _summary(wall_ns),
                    }
                )
            finally:
                runtime.close()

    aggregate: dict[str, object] = {"setup": _summary(setup_samples)}
    for component in (
        "snapshot",
        "framework",
        "publish",
        "accounted_cycle",
        "wall_cycle",
    ):
        aggregate[component] = {
            metric: float(
                statistics.median(
                    float(result[component][metric]) for result in repeat_results
                )
            )
            for metric in ("p50_us", "p95_us", "p99_us", "mean_us", "max_us", "hz")
        }
    aggregate["repeats"] = repeat_results
    return aggregate


def _measure_router_idle(seconds: float, repeats: int) -> dict[str, object]:
    baseline_ms: list[float] = []
    routed_ms: list[float] = []
    affinity = frozenset(os.sched_getaffinity(0))
    for _ in range(repeats):
        started = time.process_time_ns()
        time.sleep(seconds)
        baseline_ms.append((time.process_time_ns() - started) / 1_000_000.0)

        router = SubprocessLogRouter(
            SubprocessLoggingConfig(),
            cpu_affinity=affinity,
        )
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(3600)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        router.register(child, mod_id="com.bxi.benchmark", node_name="idle")
        try:
            started = time.process_time_ns()
            time.sleep(seconds)
            routed_ms.append((time.process_time_ns() - started) / 1_000_000.0)
        finally:
            child.terminate()
            child.wait(timeout=5.0)
            router.close()

    baseline = float(statistics.median(baseline_ms))
    routed = float(statistics.median(routed_ms))
    return {
        "sample_seconds": seconds,
        "repeats": repeats,
        "baseline_process_cpu_ms": baseline,
        "router_process_cpu_ms": routed,
        "estimated_router_cpu_ms_per_second": max(
            0.0,
            (routed - baseline) / seconds,
        ),
        "raw_baseline_ms": baseline_ms,
        "raw_router_ms": routed_ms,
    }


def _thread_scheduling(tid: int) -> dict[str, object] | None:
    task_root = Path("/proc/self/task") / str(tid)
    try:
        name = (task_root / "comm").read_text(encoding="utf-8").strip()
        status = (task_root / "status").read_text(encoding="utf-8")
        affinity = next(
            line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.startswith("Cpus_allowed_list:")
        )
        scheduler = os.sched_getscheduler(tid)
        priority = os.sched_getparam(tid).sched_priority
        affinity_cpus = sorted(os.sched_getaffinity(tid))
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {
        "tid": tid,
        "name": name,
        "cpu_affinity": affinity,
        "cpu_affinity_cpus": affinity_cpus,
        "scheduler": scheduler,
        "scheduler_priority": priority,
    }


def _task_ids() -> set[int]:
    return {
        int(path.name)
        for path in Path("/proc/self/task").iterdir()
        if path.name.isdigit()
    }


def _measure_resource_loading(
    model_path: Path,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    baseline_seconds: float,
    after_seconds: float,
    timeout_seconds: float,
    realtime_priority: int,
    dof: int,
) -> dict[str, object]:
    """Measure real 50 Hz control timing while the resource worker opens ONNX."""

    resolved_model = model_path.expanduser().resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"resource-load model does not exist: {resolved_model}")

    cpu_affinity_plan = bootstrap_process_scheduling()
    with tempfile.TemporaryDirectory(prefix="bxi-resource-load-benchmark-") as temp:
        mod_root = _write_fixture(Path(temp))
        benchmark_platform = _BenchmarkPlatform(dof)
        config = _runtime_config()
        control_config = cast(dict[str, object], config["control_runtime"])
        control_config["realtime_priority"] = realtime_priority
        runtime = RobotControlRuntime(
            config,
            built_in_mod_root=mod_root,
            command_defaults=JointCommandDefaults(),
            ros_node=_FakeNode(),
            platform=benchmark_platform,
            cpu_affinity_plan=cpu_affinity_plan,
        )
        key = ResourceKey[object]("com.bxi.framework_benchmark/load_stress")

        def load_backend(_context):
            from bxi_example_py_elf3.framework.inference import (
                InferenceRuntime,
                ModelSpec,
                RuntimeOptions,
            )

            inference = InferenceRuntime(
                options=RuntimeOptions(
                    backend="onnxruntime",
                    warmup_runs=0,
                    warn_on_fallback=False,
                )
            )
            return inference.open_backend(
                ModelSpec.onnx(
                    resolved_model,
                    input_names=input_names,
                    output_names=output_names,
                ),
                backend="onnxruntime",
            )

        runtime.framework.resources.register(
            key,
            owner="com.bxi.framework_benchmark",
            root=mod_root,
            factory=load_backend,
            policy="on_demand",
        )
        handle = runtime.framework.resources.handle(key)
        try:
            runtime.start()
            control_thread = runtime.scheduler._thread
            control_tid = (
                control_thread.native_id if control_thread is not None else None
            )
            resource_tid = runtime.framework.resources._worker.native_id
            control_scheduling = (
                _thread_scheduling(control_tid) if control_tid is not None else None
            )
            resource_scheduling = (
                _thread_scheduling(resource_tid) if resource_tid is not None else None
            )
            time.sleep(baseline_seconds)
            baseline = runtime.scheduler.timing_snapshot(reset_window=True)
            threads_before = _task_ids()

            load_started_ns = time.monotonic_ns()
            handle.request()
            deadline = time.monotonic() + timeout_seconds
            while handle.status == "loading" and time.monotonic() < deadline:
                time.sleep(0.001)
            load_finished_ns = time.monotonic_ns()
            loading = runtime.scheduler.timing_snapshot(reset_window=True)
            if handle.status != "ready":
                detail = str(handle.error) if handle.error is not None else "timeout"
                raise RuntimeError(
                    f"resource load did not complete: status={handle.status}: {detail}"
                )

            threads_after = _task_ids()
            new_threads = []
            for tid in sorted(threads_after - threads_before):
                scheduling = _thread_scheduling(tid)
                if scheduling is not None:
                    new_threads.append(scheduling)

            time.sleep(after_seconds)
            after = runtime.scheduler.timing_snapshot(reset_window=True)
            control_cpus = cpu_affinity_plan.roles[CpuAffinityRole.CONTROL]
            reserved = cpu_affinity_plan.reserved_control_core
            return {
                "model": str(resolved_model),
                "model_bytes": resolved_model.stat().st_size,
                "load_ms": (load_finished_ns - load_started_ns) / 1_000_000.0,
                "resource_status": handle.status,
                "requested_realtime_priority": realtime_priority,
                "control_scheduling_applied": bool(
                    control_scheduling is not None
                    and control_scheduling["scheduler"]
                    == (os.SCHED_FIFO if realtime_priority else os.SCHED_OTHER)
                    and control_scheduling["scheduler_priority"]
                    == realtime_priority
                ),
                "control_cpus": sorted(control_cpus),
                "reserved_control_core": sorted(reserved),
                "compute_cpus": sorted(
                    cpu_affinity_plan.roles[CpuAffinityRole.COMPUTE]
                ),
                "control_thread": control_scheduling,
                "resource_thread": resource_scheduling,
                "baseline": baseline,
                "loading": loading,
                "after": after,
                "new_threads": new_threads,
                "new_thread_control_overlap": [
                    thread["tid"]
                    for thread in new_threads
                    if set(thread["cpu_affinity_cpus"]) & reserved
                ],
            }
        finally:
            runtime.close()


def _system_info() -> dict[str, object]:
    scheduler = os.sched_getscheduler(0)
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "scheduler": scheduler,
        "scheduler_priority": os.sched_getparam(0).sched_priority,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }


def _print_report(report: dict[str, object]) -> None:
    settings = report["settings"]
    print(
        "Framework cycle benchmark "
        f"({settings['dof']} joints, {settings['iterations']} iterations x "
        f"{settings['repeats']} repeats)"
    )
    print(
        f"{'component':<20} {'p50':>10} {'p95':>10} {'p99':>10} "
        f"{'mean':>10} {'max':>10} {'Hz':>12}"
    )
    print("-" * 96)
    timings = report["timings"]
    for component in (
        "snapshot",
        "framework",
        "publish",
        "accounted_cycle",
        "wall_cycle",
    ):
        values = timings[component]
        print(
            f"{component:<20} "
            f"{values['p50_us']:>8.2f} us "
            f"{values['p95_us']:>8.2f} us "
            f"{values['p99_us']:>8.2f} us "
            f"{values['mean_us']:>8.2f} us "
            f"{values['max_us']:>8.2f} us "
            f"{values['hz']:>12.1f}"
        )
    router = report["subprocess_log_router_idle"]
    print()
    print(
        "Subprocess log router idle CPU: "
        f"{router['estimated_router_cpu_ms_per_second']:.3f} ms CPU / s wall "
        f"(median of {router['repeats']} repeats)"
    )
    resource = report.get("resource_loading")
    if resource is not None:
        print()
        print(
            "Concurrent resource load: "
            f"{resource['load_ms']:.2f} ms, "
            f"model={resource['model_bytes'] / (1024 * 1024):.2f} MiB, "
            f"control={format_cpu_set(resource['control_cpus'])}, "
            f"reserved-core={format_cpu_set(resource['reserved_control_core'])}, "
            f"compute={format_cpu_set(resource['compute_cpus'])}"
        )
        for label in ("control_thread", "resource_thread"):
            thread = resource[label]
            if thread is not None:
                print(
                    f"  {label}: TID {thread['tid']}, "
                    f"affinity={thread['cpu_affinity']}, "
                    f"policy={thread['scheduler']}, "
                    f"priority={thread['scheduler_priority']}"
                )
        if not resource["control_scheduling_applied"]:
            print(
                "  WARNING: requested control scheduling was not applied; "
                f"requested priority={resource['requested_realtime_priority']}"
            )
        print(
            f"{'phase':<12} {'cycles':>8} {'wake p99':>12} {'wake max':>12} "
            f"{'cycle p99':>12} {'cycle max':>12} {'misses':>9} {'skipped':>9}"
        )
        print("-" * 94)
        for phase in ("baseline", "loading", "after"):
            values = resource[phase]
            print(
                f"{phase:<12} "
                f"{values['cycles']:>8} "
                f"{values['wake_late_ms']['p99']:>10.3f} ms "
                f"{values['wake_late_ms']['max']:>10.3f} ms "
                f"{values['cycle_ms']['p99']:>10.3f} ms "
                f"{values['cycle_ms']['max']:>10.3f} ms "
                f"{values['deadline_misses']:>9} "
                f"{values['skipped_periods']:>9}"
            )
        print("New threads created while loading:")
        if resource["new_threads"]:
            for thread in resource["new_threads"]:
                print(
                    f"  TID {thread['tid']} {thread['name']}: "
                    f"affinity={thread['cpu_affinity']}, "
                    f"policy={thread['scheduler']}, "
                    f"priority={thread['scheduler_priority']}"
                )
        else:
            print("  none observed")
        overlap = resource["new_thread_control_overlap"]
        print(
            "Control-core overlap from new load threads: "
            + (", ".join(str(tid) for tid in overlap) if overlap else "none")
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure production framework cycle boundaries with a temporary "
            "minimal Mod; the framework source is not instrumented or modified."
        )
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dof", type=int, default=31)
    parser.add_argument("--router-idle-seconds", type=float, default=1.0)
    parser.add_argument("--router-idle-repeats", type=int, default=3)
    parser.add_argument(
        "--resource-load-model",
        type=Path,
        help="also measure 50 Hz control timing while loading this ONNX model",
    )
    parser.add_argument(
        "--resource-load-input-name",
        action="append",
        default=[],
        help="logical ONNX input name; repeat for multiple inputs",
    )
    parser.add_argument(
        "--resource-load-output-name",
        action="append",
        default=[],
        help="logical ONNX output name; repeat for multiple outputs",
    )
    parser.add_argument("--resource-baseline-seconds", type=float, default=2.0)
    parser.add_argument("--resource-after-seconds", type=float, default=2.0)
    parser.add_argument("--resource-load-timeout", type=float, default=60.0)
    parser.add_argument(
        "--resource-load-realtime-priority",
        type=int,
        default=0,
        help="control thread FIFO priority during the resource load test",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0 or args.repeats <= 0 or args.dof <= 0:
        parser.error("--iterations, --repeats and --dof must be positive")
    if args.router_idle_seconds <= 0.0 or args.router_idle_repeats <= 0:
        parser.error("router idle duration and repeats must be positive")
    if (
        args.resource_baseline_seconds <= 0.0
        or args.resource_after_seconds <= 0.0
        or args.resource_load_timeout <= 0.0
    ):
        parser.error("resource load durations must be positive")
    if not 0 <= args.resource_load_realtime_priority <= 99:
        parser.error("resource load realtime priority must be in [0, 99]")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": _system_info(),
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "dof": args.dof,
        },
        "timings": _measure_cycles(
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            dof=args.dof,
        ),
        "subprocess_log_router_idle": _measure_router_idle(
            args.router_idle_seconds,
            args.router_idle_repeats,
        ),
    }
    if args.resource_load_model is not None:
        report["resource_loading"] = _measure_resource_loading(
            args.resource_load_model,
            input_names=tuple(args.resource_load_input_name) or ("obs",),
            output_names=tuple(args.resource_load_output_name) or ("actions",),
            baseline_seconds=args.resource_baseline_seconds,
            after_seconds=args.resource_after_seconds,
            timeout_seconds=args.resource_load_timeout,
            realtime_priority=args.resource_load_realtime_priority,
            dof=args.dof,
        )
    _print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json.resolve()}")


if __name__ == "__main__":
    main()
