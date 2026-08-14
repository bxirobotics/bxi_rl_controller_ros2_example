"""Absolute-time scheduler for the framework control data path."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from threading import current_thread, Event, Lock, Thread
import time

import numpy as np

from bxi_example_py_elf3.framework.mod_api.context import LoggerLike
from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    configure_current_thread,
    format_cpu_set,
)


@dataclass(frozen=True)
class ControlCycleResult:
    """State produced by one platform-specific control cycle."""

    state: str
    active: bool = True
    next_period_sec: float | None = None


def _percentile_summary(values_ns: list[int]) -> dict[str, float]:
    if not values_ns:
        return {
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    values_ms = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "p50": float(np.percentile(values_ms, 50)),
        "p95": float(np.percentile(values_ms, 95)),
        "p99": float(np.percentile(values_ms, 99)),
        "max": float(np.max(values_ms)),
    }


def _normalize_cpu_affinity(
    value: int | Iterable[int] | None,
) -> frozenset[int] | None:
    """Keep the legacy single-CPU form while accepting resolved CPU sets."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("control CPU affinity must contain CPU indices")
    if isinstance(value, int):
        if value == -1:
            return None
        if value < 0:
            raise ValueError("control CPU affinity must be -1 or non-negative")
        return frozenset((value,))
    if isinstance(value, (str, bytes)):
        raise ValueError("control CPU affinity must contain CPU indices")
    try:
        cpus = frozenset(value)
    except TypeError as exc:
        raise ValueError("control CPU affinity must contain CPU indices") from exc
    if not cpus:
        raise ValueError("control CPU affinity must not be empty")
    if any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
        for cpu in cpus
    ):
        raise ValueError("control CPU affinity must contain non-negative indices")
    return cpus


class ControlScheduler:
    """Run all framework states on one drift-free, absolute time line."""

    def __init__(
        self,
        cycle: Callable[[], ControlCycleResult],
        *,
        period_sec: float,
        compute_budget_sec: float,
        deadline_tolerance_sec: float = 0.001,
        spin_wait_us: int = -1,
        cpu_affinity: int | Iterable[int] | None = None,
        realtime_priority: int = 0,
        logger: LoggerLike,
        fatal_callback: Callable[[str], None] | None = None,
        deadline_miss_callback: Callable[[dict[str, object]], None] | None = None,
        thread_name: str = "bxi-control",
    ) -> None:
        if period_sec <= 0.0:
            raise ValueError("control period must be greater than zero")
        if compute_budget_sec < 0.0 or compute_budget_sec >= period_sec:
            raise ValueError(
                "control compute budget must be non-negative and less than period"
            )
        if deadline_tolerance_sec < 0.0:
            raise ValueError("control deadline tolerance must be non-negative")
        period_us = int(round(period_sec * 1_000_000.0))
        if spin_wait_us < -1 or spin_wait_us >= period_us:
            raise ValueError(
                f"control spin wait must be -1 or in [0, {period_us}) microseconds"
            )
        resolved_cpu_affinity = _normalize_cpu_affinity(cpu_affinity)
        if realtime_priority < 0 or realtime_priority > 99:
            raise ValueError("control realtime priority must be in [0, 99]")

        self._cycle = cycle
        self._period_sec = float(period_sec)
        self.period_ns = int(round(self._period_sec * 1_000_000_000.0))
        self.compute_budget_ns = int(round(compute_budget_sec * 1_000_000_000.0))
        self.deadline_tolerance_ns = int(
            round(deadline_tolerance_sec * 1_000_000_000.0)
        )
        self.spin_wait_ns = -1 if spin_wait_us < 0 else int(spin_wait_us) * 1_000
        self.cpu_affinity = resolved_cpu_affinity
        self.realtime_priority = int(realtime_priority)
        self._logger = logger
        self._fatal_callback = fatal_callback
        self._deadline_miss_callback = deadline_miss_callback
        self._thread_name = thread_name

        self._stop_event = Event()
        self._thread: Thread | None = None
        self._stats_lock = Lock()
        self._reset_window_locked()
        self._total_cycles = 0
        self._total_budget_overruns = 0
        self._total_deadline_misses = 0
        self._total_skipped_periods = 0
        self._last_state = "startup"
        self._last_miss: dict[str, object] | None = None
        self._fatal_error: str | None = None
        self._next_wake_ns: int | None = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def fatal_error(self) -> str | None:
        with self._stats_lock:
            return self._fatal_error

    @property
    def period_sec(self) -> float:
        """Exact logical period supplied to the current control cycle."""

        return self._period_sec

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        thread = Thread(
            target=self._run,
            name=self._thread_name,
            daemon=False,
        )
        self._thread = thread
        thread.start()
        cpu_description = (
            "未绑定"
            if self.cpu_affinity is None
            else format_cpu_set(self.cpu_affinity)
        )
        priority_description = (
            "普通调度"
            if self.realtime_priority == 0
            else f"SCHED_FIFO/{self.realtime_priority}"
        )
        spin_description = (
            "关闭"
            if self.spin_wait_ns < 0
            else f"最后{self.spin_wait_ns / 1_000.0:.0f}微秒"
        )
        self._log(
            "info",
            "控制调度器已启动："
            f"每{self.period_ns / 1_000_000.0:.2f}毫秒执行一轮，"
            f"目标计算预算{self.compute_budget_ns / 1_000_000.0:.2f}毫秒，"
            f"允许时间误差{self.deadline_tolerance_ns / 1_000_000.0:.2f}毫秒，"
            f"末段忙等={spin_description}，"
            f"CPU={cpu_description}，调度策略={priority_description}",
        )

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        if thread is not current_thread():
            thread.join(timeout=max(0.0, float(timeout_sec)))
        if thread.is_alive():
            raise RuntimeError("control scheduler did not stop within timeout")
        self._thread = None

    def timing_snapshot(self, *, reset_window: bool = False) -> dict[str, object]:
        with self._stats_lock:
            wake = list(self._wake_ns)
            cycle = list(self._cycle_ns)
            frame_interval = list(self._frame_interval_ns)
            finish_late = list(self._finish_late_ns)
            window_cycles = self._window_cycles
            window_budget_overruns = self._window_budget_overruns
            window_deadline_misses = self._window_deadline_misses
            window_skipped_periods = self._window_skipped_periods
            total_cycles = self._total_cycles
            total_budget_overruns = self._total_budget_overruns
            total_deadline_misses = self._total_deadline_misses
            total_skipped_periods = self._total_skipped_periods
            state = self._last_state
            last_miss = dict(self._last_miss) if self._last_miss else None
            fatal_error = self._fatal_error
            period_ns = self.period_ns
            if reset_window:
                self._reset_window_locked()
        return {
            "period_ms": period_ns / 1_000_000.0,
            "compute_budget_ms": self.compute_budget_ns / 1_000_000.0,
            "cycles": window_cycles,
            "budget_overruns": window_budget_overruns,
            "deadline_misses": window_deadline_misses,
            "skipped_periods": window_skipped_periods,
            "total_cycles": total_cycles,
            "total_budget_overruns": total_budget_overruns,
            "total_deadline_misses": total_deadline_misses,
            "total_skipped_periods": total_skipped_periods,
            "state": state,
            "wake_late_ms": _percentile_summary(wake),
            "cycle_ms": _percentile_summary(cycle),
            "frame_interval_ms": _percentile_summary(frame_interval),
            "finish_late_ms": _percentile_summary(finish_late),
            "last_miss": last_miss,
            "fatal_error": fatal_error,
        }

    def status_snapshot(self) -> dict[str, object]:
        """Return cumulative counters without percentile work on the hot path."""
        with self._stats_lock:
            return {
                "period_ms": self.period_ns / 1_000_000.0,
                "compute_budget_ms": self.compute_budget_ns / 1_000_000.0,
                "total_cycles": self._total_cycles,
                "total_budget_overruns": self._total_budget_overruns,
                "total_deadline_misses": self._total_deadline_misses,
                "total_skipped_periods": self._total_skipped_periods,
                "state": self._last_state,
                "last_miss": dict(self._last_miss) if self._last_miss else None,
                "fatal_error": self._fatal_error,
            }

    def maintenance_window_available(self, minimum_sec: float) -> bool:
        """Return whether non-control work has time before the next wakeup."""
        minimum_ns = max(0, int(round(float(minimum_sec) * 1_000_000_000.0)))
        next_wake_ns = self._next_wake_ns
        return next_wake_ns is None or time.monotonic_ns() + minimum_ns < next_wake_ns

    def _run(self) -> None:
        self._configure_current_thread()
        # Releases, rather than completion times or the compute budget, define
        # a drift-free time line.  A completed cycle may select the period for
        # the next release when the active state's inference rate changes.
        release_ns = time.monotonic_ns()
        last_active_started_ns: int | None = None
        while not self._stop_event.is_set():
            cycle_period_ns = self.period_ns
            self._next_wake_ns = release_ns
            if not self._wait_until(release_ns):
                break
            wake_ns = time.monotonic_ns()
            cycle_started_ns = wake_ns
            try:
                result = self._cycle()
                if not isinstance(result, ControlCycleResult):
                    raise TypeError("control cycle must return ControlCycleResult")
                next_period_sec, next_period_ns = self._resolve_next_period(
                    result.next_period_sec,
                    current_period_sec=self._period_sec,
                    current_period_ns=cycle_period_ns,
                )
            except BaseException as exc:
                message = f"control scheduler stopped after cycle failure: {exc}"
                with self._stats_lock:
                    self._fatal_error = message
                self._log("error", message)
                if self._fatal_callback is not None:
                    try:
                        self._fatal_callback(message)
                    except Exception as callback_exc:
                        self._log(
                            "error",
                            f"control fatal callback also failed: {callback_exc}",
                        )
                self._stop_event.set()
                break
            finished_ns = time.monotonic_ns()

            if not result.active:
                last_active_started_ns = None
                self._set_period(next_period_sec, next_period_ns, result.state)
                release_ns = self._next_release_after(
                    release_ns,
                    finished_ns,
                    next_period_ns,
                )
                continue
            if last_active_started_ns is None:
                with self._stats_lock:
                    self._reset_window_locked()

            wake_late_ns = max(0, wake_ns - release_ns)
            cycle_ns = max(0, finished_ns - cycle_started_ns)
            frame_interval_ns = (
                max(0, cycle_started_ns - last_active_started_ns)
                if last_active_started_ns is not None
                else 0
            )
            last_active_started_ns = cycle_started_ns

            # Each release owns the complete following period.  Starting late
            # and finishing after the next release are both deadline failures.
            deadline_ns = release_ns + cycle_period_ns
            finish_late_ns = max(0, finished_ns - deadline_ns)
            deadline_missed = (
                wake_late_ns > self.deadline_tolerance_ns
                or finish_late_ns > self.deadline_tolerance_ns
            )
            budget_overrun = cycle_ns > self.compute_budget_ns

            next_release_ns = release_ns + next_period_ns
            skipped_periods = 0
            # A small overrun runs the overdue release immediately and catches
            # the original time line on a later short cycle.  Skip only release
            # points for which another complete period has already elapsed.
            if finished_ns >= next_release_ns + next_period_ns:
                skipped_periods = (
                    finished_ns - next_release_ns
                ) // next_period_ns
                next_release_ns += skipped_periods * next_period_ns

            miss_event = self._record_cycle(
                result,
                period_ns=cycle_period_ns,
                wake_late_ns=wake_late_ns,
                cycle_ns=cycle_ns,
                frame_interval_ns=frame_interval_ns,
                finish_late_ns=finish_late_ns,
                budget_overrun=budget_overrun,
                deadline_missed=deadline_missed,
                skipped_periods=int(skipped_periods),
            )
            if miss_event is not None and self._deadline_miss_callback is not None:
                try:
                    self._deadline_miss_callback(miss_event)
                except Exception as exc:
                    self._log("error", f"deadline miss callback failed: {exc}")
            self._set_period(next_period_sec, next_period_ns, result.state)
            release_ns = next_release_ns
        self._next_wake_ns = None

    def _resolve_next_period(
        self,
        requested_sec: float | None,
        *,
        current_period_sec: float,
        current_period_ns: int,
    ) -> tuple[float, int]:
        if requested_sec is None:
            return current_period_sec, current_period_ns
        if isinstance(requested_sec, bool) or not isinstance(
            requested_sec, (int, float)
        ):
            raise ValueError("next control period must be a number")
        period_sec = float(requested_sec)
        if not math.isfinite(period_sec) or period_sec <= 0.0:
            raise ValueError("next control period must be finite and positive")
        period_ns = int(round(period_sec * 1_000_000_000.0))
        if period_ns <= 0:
            raise ValueError("next control period is below scheduler resolution")
        if period_ns <= self.compute_budget_ns:
            raise ValueError(
                "next control period must be greater than the configured "
                "compute budget"
            )
        if self.spin_wait_ns >= period_ns:
            raise ValueError(
                "next control period must be greater than the configured spin wait"
            )
        return period_sec, period_ns

    def _set_period(self, period_sec: float, period_ns: int, state: str) -> None:
        previous_ns = self.period_ns
        self._period_sec = period_sec
        self.period_ns = period_ns
        if period_ns != previous_ns:
            self._log(
                "info",
                "控制周期已切换："
                f"状态={state}，频率={1.0 / period_sec:.3f} Hz，"
                f"周期={period_ns / 1_000_000.0:.3f} ms",
            )

    @staticmethod
    def _next_release_after(
        release_ns: int,
        finished_ns: int,
        period_ns: int,
    ) -> int:
        """Keep inactive/startup work on the same period-aligned time line."""
        next_release_ns = release_ns + period_ns
        if finished_ns <= next_release_ns:
            return next_release_ns
        skipped = (
            finished_ns - next_release_ns + period_ns - 1
        ) // period_ns
        return next_release_ns + skipped * period_ns

    def _wait_until(self, target_ns: int) -> bool:
        while not self._stop_event.is_set():
            remaining_ns = target_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return True

            # Sleep for most of the wait, then optionally busy-spin over only
            # the configured final slice.  The absolute target is unchanged,
            # so the absolute release time line cannot drift.
            if self.spin_wait_ns >= 0 and remaining_ns <= self.spin_wait_ns:
                while time.monotonic_ns() < target_ns:
                    pass
                return not self._stop_event.is_set()

            sleep_ns = remaining_ns
            if self.spin_wait_ns >= 0:
                sleep_ns -= self.spin_wait_ns
            if self._stop_event.wait(sleep_ns / 1_000_000_000.0):
                return False
        return False

    def _record_cycle(
        self,
        result: ControlCycleResult,
        *,
        period_ns: int,
        wake_late_ns: int,
        cycle_ns: int,
        frame_interval_ns: int,
        finish_late_ns: int,
        budget_overrun: bool,
        deadline_missed: bool,
        skipped_periods: int,
    ) -> dict[str, object] | None:
        miss_event: dict[str, object] | None = None
        with self._stats_lock:
            self._window_cycles += 1
            self._total_cycles += 1
            self._last_state = result.state
            self._wake_ns.append(wake_late_ns)
            self._cycle_ns.append(cycle_ns)
            self._frame_interval_ns.append(frame_interval_ns)
            self._finish_late_ns.append(finish_late_ns)
            if budget_overrun:
                self._window_budget_overruns += 1
                self._total_budget_overruns += 1
            self._window_skipped_periods += skipped_periods
            self._total_skipped_periods += skipped_periods
            if deadline_missed:
                self._window_deadline_misses += 1
                self._total_deadline_misses += 1
                miss_event = {
                    "state": result.state,
                    "period_ms": period_ns / 1_000_000.0,
                    "wake_late_ms": wake_late_ns / 1_000_000.0,
                    "cycle_ms": cycle_ns / 1_000_000.0,
                    "finish_late_ms": finish_late_ns / 1_000_000.0,
                    "count": self._total_deadline_misses,
                }
                self._last_miss = miss_event
        return dict(miss_event) if miss_event is not None else None

    def _configure_current_thread(self) -> None:
        cpu_description = (
            "inherit"
            if self.cpu_affinity is None
            else format_cpu_set(self.cpu_affinity)
        )
        try:
            configure_current_thread(
                self.cpu_affinity,
                realtime_priority=self.realtime_priority,
            )
        except (AttributeError, OSError) as exc:
            policy = (
                "SCHED_OTHER"
                if self.realtime_priority == 0
                else f"SCHED_FIFO/{self.realtime_priority}"
            )
            self._log(
                "warning",
                "cannot configure control thread scheduling: "
                f"CPU={cpu_description}, policy={policy}: {exc}",
            )

    def _reset_window_locked(self) -> None:
        self._wake_ns: list[int] = []
        self._cycle_ns: list[int] = []
        self._frame_interval_ns: list[int] = []
        self._finish_late_ns: list[int] = []
        self._window_cycles = 0
        self._window_budget_overruns = 0
        self._window_deadline_misses = 0
        self._window_skipped_periods = 0

    def _log(self, level: str, message: str) -> None:
        logger = self._logger
        try:
            # rclpy binds severity to a Python log call's source location.
            # A shared dynamic call line can emit INFO once and then reject a
            # later WARNING or ERROR from that same line.
            if level == "info":
                logger.info(message)
                return
            if level == "warning":
                logger.warning(message)
                return
            if level == "error":
                logger.error(message)
        except Exception:
            pass


__all__ = ["ControlCycleResult", "ControlScheduler"]
