#!/usr/bin/env python3
"""Monitor one Linux process and per-core CPU usage through /proc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
import time


CLOCK_TICKS = int(os.sysconf("SC_CLK_TCK"))


@dataclass(frozen=True)
class TaskCounter:
    name: str
    cpu_ticks: int
    recent_cpu: int


@dataclass(frozen=True)
class Snapshot:
    monotonic_ns: int
    process_ticks: int
    tasks: dict[int, TaskCounter]
    system_cores: dict[int, tuple[int, int]]


def parse_task_stat(text: str) -> TaskCounter:
    command_start = text.find("(")
    command_end = text.rfind(")")
    if command_start < 0 or command_end <= command_start:
        raise ValueError("invalid task stat command field")
    fields = text[command_end + 1 :].split()  # noqa: E203
    # fields[0] is proc stat field 3; utime=14, stime=15, processor=39.
    if len(fields) <= 36:
        raise ValueError("invalid task stat field count")
    return TaskCounter(
        name=text[command_start + 1 : command_end],  # noqa: E203
        cpu_ticks=int(fields[11]) + int(fields[12]),
        recent_cpu=int(fields[36]),
    )


def read_task(path: Path) -> TaskCounter:
    return parse_task_stat(path.read_text(encoding="utf-8"))


def read_system_cores(proc_root: Path) -> dict[int, tuple[int, int]]:
    cores: dict[int, tuple[int, int]] = {}
    for line in (proc_root / "stat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu"):
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit():
            continue
        values = [int(value) for value in fields[1:9]]
        if len(values) < 4:
            raise ValueError(f"invalid {fields[0]} counters")
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        cores[int(suffix)] = (total, idle)
    if not cores:
        raise ValueError("no per-core CPU counters found")
    return cores


def read_snapshot(pid: int, proc_root: Path = Path("/proc")) -> Snapshot:
    process_root = proc_root / str(pid)
    process = read_task(process_root / "stat")
    tasks: dict[int, TaskCounter] = {}
    for task_path in (process_root / "task").iterdir():
        if not task_path.name.isdigit():
            continue
        try:
            tasks[int(task_path.name)] = read_task(task_path / "stat")
        except FileNotFoundError:
            # A thread may exit while /proc is being sampled.
            continue
    return Snapshot(
        monotonic_ns=time.monotonic_ns(),
        process_ticks=process.cpu_ticks,
        tasks=tasks,
        system_cores=read_system_cores(proc_root),
    )


def percent(delta_ticks: int, elapsed_sec: float) -> float:
    return 100.0 * max(0, delta_ticks) / (CLOCK_TICKS * elapsed_sec)


def format_core_values(values: dict[int, float]) -> str:
    return ", ".join(f"cpu{cpu}:{value:.1f}%" for cpu, value in sorted(values.items()))


def report(
    previous: Snapshot, current: Snapshot, show_threads: bool, limit: int
) -> None:
    elapsed_sec = (current.monotonic_ns - previous.monotonic_ns) / 1e9
    if elapsed_sec <= 0.0:
        return

    process_percent = percent(
        current.process_ticks - previous.process_ticks,
        elapsed_sec,
    )
    thread_usage: list[tuple[float, int, TaskCounter]] = []
    process_by_recent_cpu = {cpu: 0.0 for cpu in current.system_cores}
    for tid, task in current.tasks.items():
        old_task = previous.tasks.get(tid)
        if old_task is None:
            continue
        usage = percent(task.cpu_ticks - old_task.cpu_ticks, elapsed_sec)
        thread_usage.append((usage, tid, task))
        process_by_recent_cpu.setdefault(task.recent_cpu, 0.0)
        process_by_recent_cpu[task.recent_cpu] += usage

    system_usage: dict[int, float] = {}
    for cpu, (total, idle) in current.system_cores.items():
        old = previous.system_cores.get(cpu)
        if old is None:
            continue
        total_delta = max(0, total - old[0])
        idle_delta = max(0, idle - old[1])
        if total_delta:
            busy_delta = max(0, total_delta - min(idle_delta, total_delta))
            system_usage[cpu] = 100.0 * busy_delta / total_delta

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] process={process_percent:.1f}% "
        f"({process_percent / 100.0:.2f} cores)"
    )
    print(f"  process_by_recent_cpu~=[{format_core_values(process_by_recent_cpu)}]")
    print(f"  system_cores=[{format_core_values(system_usage)}]")
    if show_threads:
        print("  threads:     TID   CPU    %CPU  NAME")
        for usage, tid, task in sorted(thread_usage, reverse=True)[:limit]:
            print(
                f"             {tid:6d}  {task.recent_cpu:4d}  "
                f"{usage:6.1f}  {task.name}"
            )
    sys.stdout.flush()


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor one process and system per-core CPU usage via /proc.",
    )
    parser.add_argument("pid", type=positive_int, help="process ID to monitor")
    parser.add_argument(
        "-i",
        "--interval",
        type=positive_float,
        default=1.0,
        help="sampling interval in seconds (default: 1)",
    )
    parser.add_argument(
        "--threads",
        action="store_true",
        help="also print the busiest threads",
    )
    parser.add_argument(
        "--top-threads",
        type=positive_int,
        default=10,
        help="maximum thread rows when --threads is used (default: 10)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one interval and exit",
    )
    args = parser.parse_args()

    try:
        previous = read_snapshot(args.pid)
        while True:
            time.sleep(args.interval)
            current = read_snapshot(args.pid)
            report(previous, current, args.threads, args.top_threads)
            if args.once:
                return 0
            previous = current
    except KeyboardInterrupt:
        return 0
    except (FileNotFoundError, PermissionError, ProcessLookupError) as exc:
        print(f"cannot monitor PID {args.pid}: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"cannot read CPU counters: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
