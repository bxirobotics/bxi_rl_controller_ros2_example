"""Framework-owned control-loop orchestration independent of ROS timers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
from threading import current_thread, Event, Lock, Thread
import time
from typing import Callable

from rclpy.logging import get_logger

from bxi_example_py_elf3.framework.joints import JointCommandDefaults
from bxi_example_py_elf3.framework.mod_api import TransitionSpec

from bxi_example_py_elf3.framework.runtime.control_scheduler import (
    ControlCycleResult,
    ControlScheduler,
)
from bxi_example_py_elf3.framework.runtime.controller import RobotControlFramework
from bxi_example_py_elf3.framework.runtime.logging import LoggingConfig, ScopedLoggers
from bxi_example_py_elf3.framework.runtime.mod_nodes import ExecutorLike
from .api import ControlPlatformAdapter
from .cpu_affinity import (
    CpuAffinityPlan,
    CpuAffinityRole,
    CpuAffinitySpec,
    read_cpu_affinity,
)


@dataclass(frozen=True)
class ControlRuntimeConfig:
    """Validated scheduler and supervision settings."""

    period_sec: float = 0.02
    compute_budget_sec: float = 0.002
    deadline_tolerance_sec: float = 0.001
    maintenance_hz: float = 5.0
    statistics_interval_sec: float = 30.0
    deadline_warning_interval_sec: float = 1.0
    maintenance_guard_sec: float = 0.005
    python_switch_interval_sec: float = 0.001
    spin_wait_us: int = -1
    cpu_affinity: CpuAffinitySpec = CpuAffinitySpec(
        role=CpuAffinityRole.CONTROL
    )
    realtime_priority: int = 0

    @classmethod
    def from_mapping(cls, raw: object) -> "ControlRuntimeConfig":
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("control_runtime must be a YAML map")

        allowed = {
            "period_sec",
            "compute_budget_sec",
            "deadline_tolerance_sec",
            "maintenance_hz",
            "statistics_interval_sec",
            "deadline_warning_interval_sec",
            "maintenance_guard_sec",
            "python_switch_interval_sec",
            "spin_wait_us",
            "cpu_affinity",
            "realtime_priority",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"control_runtime contains unknown fields: {sorted(unknown)}"
            )

        defaults = cls()
        config = cls(
            period_sec=_number(raw, "period_sec", defaults.period_sec),
            compute_budget_sec=_number(
                raw, "compute_budget_sec", defaults.compute_budget_sec
            ),
            deadline_tolerance_sec=_number(
                raw,
                "deadline_tolerance_sec",
                defaults.deadline_tolerance_sec,
            ),
            maintenance_hz=_number(raw, "maintenance_hz", defaults.maintenance_hz),
            statistics_interval_sec=_number(
                raw,
                "statistics_interval_sec",
                defaults.statistics_interval_sec,
            ),
            deadline_warning_interval_sec=_number(
                raw,
                "deadline_warning_interval_sec",
                defaults.deadline_warning_interval_sec,
            ),
            maintenance_guard_sec=_number(
                raw,
                "maintenance_guard_sec",
                defaults.maintenance_guard_sec,
            ),
            python_switch_interval_sec=_number(
                raw,
                "python_switch_interval_sec",
                defaults.python_switch_interval_sec,
            ),
            spin_wait_us=_integer(raw, "spin_wait_us", defaults.spin_wait_us),
            cpu_affinity=read_cpu_affinity(
                raw.get("cpu_affinity"),
                "control_runtime.cpu_affinity",
                default=CpuAffinityRole.CONTROL,
            ),
            realtime_priority=_integer(
                raw, "realtime_priority", defaults.realtime_priority
            ),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if self.period_sec <= 0.0:
            raise ValueError("control_runtime.period_sec must be greater than zero")
        if not 0.0 <= self.compute_budget_sec < self.period_sec:
            raise ValueError(
                "control_runtime.compute_budget_sec must be non-negative and "
                "less than period_sec"
            )
        if self.deadline_tolerance_sec < 0.0:
            raise ValueError(
                "control_runtime.deadline_tolerance_sec must be non-negative"
            )
        if self.maintenance_hz <= 0.0:
            raise ValueError("control_runtime.maintenance_hz must be greater than zero")
        if self.statistics_interval_sec <= 0.0:
            raise ValueError(
                "control_runtime.statistics_interval_sec must be greater than zero"
            )
        if self.deadline_warning_interval_sec <= 0.0:
            raise ValueError(
                "control_runtime.deadline_warning_interval_sec must be greater "
                "than zero"
            )
        if not 0.0 <= self.maintenance_guard_sec < self.period_sec:
            raise ValueError(
                "control_runtime.maintenance_guard_sec must be non-negative and "
                "less than period_sec"
            )
        if self.python_switch_interval_sec <= 0.0:
            raise ValueError(
                "control_runtime.python_switch_interval_sec must be greater than zero"
            )
        period_us = int(round(self.period_sec * 1_000_000.0))
        if self.spin_wait_us < -1 or self.spin_wait_us >= period_us:
            raise ValueError(
                "control_runtime.spin_wait_us must be -1 or in "
                f"[0, {period_us})"
            )
        if not 0 <= self.realtime_priority <= 99:
            raise ValueError("control_runtime.realtime_priority must be in [0, 99]")


class RobotControlRuntime:
    """Own the framework, deterministic scheduler and background supervision."""

    def __init__(
        self,
        system_config: Mapping[str, object],
        *,
        built_in_mod_root: Path,
        command_defaults: JointCommandDefaults,
        ros_node: object,
        platform: ControlPlatformAdapter,
        cpu_affinity_plan: CpuAffinityPlan,
        fatal_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = ControlRuntimeConfig.from_mapping(
            system_config.get("control_runtime")
        )
        self.logging_config = LoggingConfig.from_mapping(system_config.get("logging"))
        framework_config = {
            key: value
            for key, value in system_config.items()
            if key not in {"control_runtime", "logging"}
        }
        self._platform = platform
        root_logger = ros_node.get_logger()
        self.loggers = ScopedLoggers(
            root_logger,
            self.logging_config,
            logger_factory=get_logger,
        )
        self._logger = self.loggers.framework("runtime")
        self._external_fatal_callback = fatal_callback
        self._framework_lock = Lock()
        self._stop_event = Event()
        self._maintenance_thread: Thread | None = None
        self._deadline_miss_queue: SimpleQueue[dict[str, object]] = SimpleQueue()
        self._pending_deadline_summary: dict[str, object] | None = None
        self._last_deadline_warning_at = 0.0
        self._started = False
        self._closed = False
        self._original_python_switch_interval: float | None = None
        self._last_control_events: tuple[str, ...] = ()
        self._last_reported_total_cycles = 0
        self._last_reported_total_budget_overruns = 0
        self._last_reported_total_deadline_misses = 0
        self._last_reported_total_skipped_periods = 0

        self.cpu_affinity_plan = cpu_affinity_plan
        control_cpus = self.cpu_affinity_plan.resolve(
            self.config.cpu_affinity,
            context="control_runtime.cpu_affinity",
        )

        self.framework = RobotControlFramework(
            framework_config,
            built_in_mod_root=built_in_mod_root,
            command_defaults=command_defaults,
            ros_node=ros_node,
            loggers=self.loggers,
            control_period=self.config.period_sec,
            cpu_affinity_plan=self.cpu_affinity_plan,
        )
        self.scheduler = ControlScheduler(
            self._run_control_cycle,
            period_sec=self.config.period_sec,
            compute_budget_sec=self.config.compute_budget_sec,
            deadline_tolerance_sec=self.config.deadline_tolerance_sec,
            spin_wait_us=self.config.spin_wait_us,
            cpu_affinity=control_cpus,
            realtime_priority=self.config.realtime_priority,
            logger=self.loggers.framework("scheduler"),
            fatal_callback=self._on_control_fatal,
            deadline_miss_callback=self._enqueue_deadline_miss,
        )
        self._log("info", self.cpu_affinity_plan.describe())
        self.framework.log_startup()

    @property
    def is_running(self) -> bool:
        return self._started and self.scheduler.is_running

    @property
    def period_sec(self) -> float:
        return self.scheduler.period_sec

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("RobotControlRuntime is closed")
        if self._started:
            return

        self._stop_event.clear()
        original_interval = sys.getswitchinterval()
        self._original_python_switch_interval = original_interval
        sys.setswitchinterval(self.config.python_switch_interval_sec)
        maintenance_thread = Thread(
            target=self._maintenance_loop,
            name="bxi-maintenance",
            daemon=False,
        )
        self._maintenance_thread = maintenance_thread
        try:
            maintenance_thread.start()
            self.scheduler.start()
            self._started = True
        except BaseException:
            self._stop_event.set()
            self.scheduler.stop()
            self._join_maintenance_thread()
            self._restore_python_switch_interval()
            raise

    def stop(self) -> None:
        self._stop_event.set()
        first_error: Exception | None = None
        try:
            self.scheduler.stop()
        except Exception as exc:
            first_error = exc

        background_error = self._join_maintenance_thread()
        if first_error is None:
            first_error = background_error
        self._started = False
        self._restore_python_switch_interval()
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        try:
            self.stop()
        except Exception as exc:
            first_error = exc
        try:
            with self._framework_lock:
                self.framework.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                self._log("warning", f"control framework cleanup also failed: {exc}")
        if first_error is not None:
            raise first_error

    def attach_executor(self, executor: ExecutorLike) -> None:
        with self._framework_lock:
            self.framework.attach_executor(executor)

    def detach_executor(self) -> None:
        with self._framework_lock:
            self.framework.detach_executor()

    def extract_remote_events(
        self,
        values: object,
        *,
        sync_only: bool = False,
    ) -> list[str]:
        with self._framework_lock:
            return self.framework.extract_remote_events(values, sync_only=sync_only)

    def request_state(
        self,
        state_name: str,
        *,
        trigger: str,
        transition: TransitionSpec = None,
        delay: float = 0.0,
        force: bool = False,
    ) -> bool:
        with self._framework_lock:
            return self.framework.request_state(
                state_name,
                trigger=trigger,
                transition=transition,
                delay=delay,
                force=force,
            )

    def snapshot(self, *, include_graph: bool = False) -> dict[str, object] | None:
        """Return a coherent status snapshot, or None near a control deadline."""
        if not self._wait_for_maintenance_window(self.period_sec):
            return None
        if not self._framework_lock.acquire(blocking=False):
            return None
        try:
            info = self.framework.snapshot(include_graph=include_graph)
        finally:
            self._framework_lock.release()
        info["control_timing"] = self.scheduler.status_snapshot()
        info["events"] = list(self._last_control_events)
        return info

    def _run_control_cycle(self) -> ControlCycleResult:
        if not self._platform.startup_step(time.monotonic()):
            return ControlCycleResult(state="startup", active=False)

        observation, events = self._platform.snapshot_control_inputs()
        events = tuple(events)
        self._last_control_events = events

        with self._framework_lock:
            cycle_period_sec = self.scheduler.period_sec
            frame = self.framework.update(
                observation,
                events,
                cycle_period_sec,
            )
            state_name = self.framework.current_state_name
            next_period_sec = self.framework.desired_control_period

        if frame is not None:
            self._platform.publish_motor_frame(frame)
        return ControlCycleResult(
            state=state_name,
            next_period_sec=next_period_sec,
        )

    def _maintenance_loop(self) -> None:
        period_sec = 1.0 / self.config.maintenance_hz
        next_maintenance_at = time.monotonic() + period_sec
        next_statistics_at = time.monotonic() + self.config.statistics_interval_sec
        while not self._stop_event.is_set():
            delay = max(0.0, next_maintenance_at - time.monotonic())
            if self._stop_event.wait(delay):
                break
            now = time.monotonic()
            next_maintenance_at += period_sec
            if now >= next_maintenance_at:
                skipped = int((now - next_maintenance_at) // period_sec) + 1
                next_maintenance_at += skipped * period_sec

            self._drain_deadline_misses()

            if self._wait_for_maintenance_window(self.period_sec):
                if self._framework_lock.acquire(blocking=False):
                    try:
                        self.framework.maintenance_update()
                    except Exception as exc:
                        self._log("warning", f"control maintenance failed: {exc}")
                    finally:
                        self._framework_lock.release()

            now = time.monotonic()
            if now >= next_statistics_at:
                self._report_control_timing()
                next_statistics_at = now + self.config.statistics_interval_sec

    def _enqueue_deadline_miss(self, miss: dict[str, object]) -> None:
        self._deadline_miss_queue.put(dict(miss))

    def _drain_deadline_misses(self) -> None:
        tolerance_ms = self.config.deadline_tolerance_sec * 1000.0
        while True:
            try:
                miss = self._deadline_miss_queue.get_nowait()
            except Empty:
                break
            self._merge_deadline_miss(miss, tolerance_ms=tolerance_ms)

        summary = self._pending_deadline_summary
        if summary is None:
            return
        now = time.monotonic()
        if (
            now - self._last_deadline_warning_at
            < self.config.deadline_warning_interval_sec
        ):
            return

        budget_ms = self.config.compute_budget_sec * 1000.0
        period_ms = float(summary["period_ms"])
        wake_events = int(summary["wake_events"])
        finish_events = int(summary["finish_events"])
        finish_detail = (
            f"，最大截止超限{summary['max_finish_late_ms']:.2f} ms"
            if finish_events > 0
            else ""
        )
        if wake_events > 0 and finish_events == 0:
            conclusion = (
                "初步判定：异常由调度唤醒延迟引起，"
                "周期执行未超过截止时间；"
            )
        elif wake_events == 0 and finish_events > 0:
            conclusion = (
                "初步判定：周期执行超过截止时间，"
                "调度唤醒延迟未超限；"
            )
        else:
            conclusion = (
                "初步判定：同时存在调度唤醒延迟和"
                "周期完成截止超限；"
            )
        logged = self._log(
            "warning",
            "控制周期时序异常："
            f"当前状态={summary['state']}；"
            f"本次汇总异常周期{summary['events']}次，其中"
            f"调度唤醒延迟超限{wake_events}次，"
            f"周期完成截止超限{finish_events}次；"
            f"最大唤醒延迟{summary['max_wake_late_ms']:.2f} ms"
            f"（允许偏差{tolerance_ms:.2f} ms），"
            f"最大周期执行耗时{summary['max_cycle_ms']:.2f} ms"
            f"（预算{budget_ms:.2f} ms，目标周期{period_ms:.2f} ms）"
            f"{finish_detail}。"
            f"{conclusion}"
            f"启动以来累计异常{summary['latest_count']}次。",
        )
        if logged:
            self._pending_deadline_summary = None
            self._last_deadline_warning_at = now

    def _merge_deadline_miss(
        self,
        miss: dict[str, object],
        *,
        tolerance_ms: float,
    ) -> None:
        """Merge hot-path timing events without doing log I/O per cycle."""
        wake_late_ms = float(miss["wake_late_ms"])
        cycle_ms = float(miss["cycle_ms"])
        finish_late_ms = float(miss["finish_late_ms"])
        summary = self._pending_deadline_summary
        if summary is None:
            summary = {
                "state": str(miss["state"]),
                "period_ms": float(miss["period_ms"]),
                "events": 0,
                "wake_events": 0,
                "finish_events": 0,
                "max_wake_late_ms": 0.0,
                "max_cycle_ms": 0.0,
                "max_finish_late_ms": 0.0,
                "latest_count": 0,
            }
            self._pending_deadline_summary = summary
        summary["state"] = str(miss["state"])
        summary["period_ms"] = float(miss["period_ms"])
        summary["events"] = int(summary["events"]) + 1
        if wake_late_ms > tolerance_ms:
            summary["wake_events"] = int(summary["wake_events"]) + 1
        if finish_late_ms > tolerance_ms:
            summary["finish_events"] = int(summary["finish_events"]) + 1
        summary["max_wake_late_ms"] = max(
            float(summary["max_wake_late_ms"]), wake_late_ms
        )
        summary["max_cycle_ms"] = max(
            float(summary["max_cycle_ms"]), cycle_ms
        )
        summary["max_finish_late_ms"] = max(
            float(summary["max_finish_late_ms"]), finish_late_ms
        )
        summary["latest_count"] = max(
            int(summary["latest_count"]), int(miss["count"])
        )

    def _join_maintenance_thread(self) -> Exception | None:
        thread = self._maintenance_thread
        if thread is not None and thread is not current_thread() and thread.ident:
            thread.join(timeout=5.0)
            if thread.is_alive():
                return RuntimeError(
                    "control maintenance thread did not stop within timeout"
                )
        self._maintenance_thread = None
        return None

    def _wait_for_maintenance_window(self, timeout_sec: float) -> bool:
        wait_deadline = time.monotonic() + max(0.0, timeout_sec)
        while not self.scheduler.maintenance_window_available(
            self.config.maintenance_guard_sec
        ):
            if self._stop_event.is_set() or time.monotonic() >= wait_deadline:
                return False
            self._stop_event.wait(0.0005)
        return not self._stop_event.is_set()

    def _report_control_timing(self) -> None:
        timing = self.scheduler.timing_snapshot(reset_window=True)
        total_cycles = int(timing["total_cycles"])
        total_budget_overruns = int(timing["total_budget_overruns"])
        total_misses = int(timing["total_deadline_misses"])
        total_skipped = int(timing["total_skipped_periods"])
        cycles_since_report = max(
            0,
            total_cycles - self._last_reported_total_cycles,
        )
        budget_overruns_since_report = max(
            0,
            total_budget_overruns - self._last_reported_total_budget_overruns,
        )
        misses_since_report = max(
            0,
            total_misses - self._last_reported_total_deadline_misses,
        )
        skipped_since_report = max(
            0,
            total_skipped - self._last_reported_total_skipped_periods,
        )
        wake = timing["wake_late_ms"]
        cycle = timing["cycle_ms"]
        interval = timing["frame_interval_ms"]
        period_ms = float(timing["period_ms"])
        budget_ms = self.config.compute_budget_sec * 1000.0
        miss_rate = (
            100.0 * misses_since_report / cycles_since_report
            if cycles_since_report > 0
            else 0.0
        )
        message = (
            "控制周期性能统计："
            f"统计窗口内执行{cycles_since_report}个周期，"
            f"当前状态={timing['state']}；"
            f"时序异常{misses_since_report}次（占{miss_rate:.2f}%），"
            f"计算预算超限{budget_overruns_since_report}次，"
            f"跳过{skipped_since_report}个周期。"
            f"P99唤醒延迟{wake['p99']:.2f} ms，"
            f"P99周期执行耗时{cycle['p99']:.2f} ms，"
            f"P99周期间隔{interval['p99']:.2f} ms；"
            f"最大周期执行耗时{cycle['max']:.2f} ms，"
            f"最大周期间隔{interval['max']:.2f} ms。"
            f"目标周期{period_ms:.2f} ms，计算预算{budget_ms:.2f} ms；"
            f"启动以来累计异常{total_misses}次。"
        )
        logged = self._log("info", message)
        if logged:
            self._last_reported_total_cycles = total_cycles
            self._last_reported_total_budget_overruns = total_budget_overruns
            self._last_reported_total_deadline_misses = total_misses
            self._last_reported_total_skipped_periods = total_skipped

    def _on_control_fatal(self, message: str) -> None:
        self._stop_event.set()
        callback = self._external_fatal_callback
        if callback is not None:
            callback(message)

    def _restore_python_switch_interval(self) -> None:
        original = self._original_python_switch_interval
        self._original_python_switch_interval = None
        if original is not None:
            sys.setswitchinterval(original)

    def _log(self, level: str, message: str) -> bool:
        logger = self._logger
        try:
            # rclpy binds severity to the Python source location of a log call.
            # Keep each severity on a distinct call line instead of using one
            # dynamic getattr call for INFO, WARNING and ERROR.
            if level == "info":
                logger.info(message)
                return True
            if level == "warning":
                logger.warning(message)
                return True
            if level == "error":
                logger.error(message)
                return True
            return False
        except Exception:
            return False


def _number(raw: Mapping[object, object], name: str, default: float) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"control_runtime.{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"control_runtime.{name} must be finite")
    return result


def _integer(raw: Mapping[object, object], name: str, default: int) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"control_runtime.{name} must be an integer")
    return int(value)


__all__ = [
    "ControlPlatformAdapter",
    "ControlRuntimeConfig",
    "RobotControlRuntime",
]
