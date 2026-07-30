#!/usr/bin/env python3
"""Inspect the real Linux scheduling state of a process tree.

Only procfs, sysfs and read-only scheduler syscalls are used.  The target does
not need to import this script or expose a diagnostics API.  Every thread is
queried separately because CPU affinity and scheduling policy are per-thread on
Linux, even when tools casually display them as process properties.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Iterable


PROC = Path("/proc")
SYS_CPU = Path("/sys/devices/system/cpu")
CGROUP = Path("/sys/fs/cgroup")
CLK_TCK = int(os.sysconf("SC_CLK_TCK"))
RESET_ON_FORK = 0x40000000
POLICY_MASK = RESET_ON_FORK - 1
POLICY_NAMES = {
    getattr(os, "SCHED_OTHER", 0): "OTHER",
    getattr(os, "SCHED_FIFO", 1): "FIFO",
    getattr(os, "SCHED_RR", 2): "RR",
    getattr(os, "SCHED_BATCH", 3): "BATCH",
    getattr(os, "SCHED_IDLE", 5): "IDLE",
    getattr(os, "SCHED_DEADLINE", 6): "DEADLINE",
}


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _read_int(path: Path) -> int | None:
    value = _read(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_stat(path: Path) -> dict[str, object] | None:
    raw = _read(path)
    if raw is None:
        return None
    closing = raw.rfind(")")
    opening = raw.find("(")
    if opening < 0 or closing < opening:
        return None
    try:
        pid = int(raw[:opening].strip())
        comm = raw[opening + 1 : closing]
        fields = raw[closing + 2 :].split()
        return {
            "pid": pid,
            "comm": comm,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "utime_ticks": int(fields[11]),
            "stime_ticks": int(fields[12]),
            "kernel_priority": int(fields[15]),
            "nice": int(fields[16]),
            "threads": int(fields[17]),
            "starttime_ticks": int(fields[19]),
            "last_cpu": int(fields[36]),
            "rt_priority_stat": int(fields[37]),
            "policy_stat": int(fields[38]),
        }
    except (IndexError, ValueError):
        return None


def _parse_key_values(path: Path) -> dict[str, str]:
    raw = _read(path)
    if raw is None:
        return {}
    result: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def _integer_prefix(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.match(r"-?\d+", value)
    return int(match.group(0)) if match else None


def _float_value(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def _cmdline(pid: int) -> str:
    try:
        raw = (PROC / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _process_index() -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    try:
        entries = tuple(PROC.iterdir())
    except OSError:
        return result
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = _parse_stat(entry / "stat")
        if stat is not None:
            result[int(stat["pid"])] = stat
    return result


def _descendants(root_pid: int, index: dict[int, dict[str, object]]) -> list[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for pid, stat in index.items():
        children[int(stat["ppid"])].append(pid)
    for values in children.values():
        values.sort()
    result: list[int] = []
    queue: deque[int] = deque((root_pid,))
    seen: set[int] = set()
    while queue:
        pid = queue.popleft()
        if pid in seen or pid not in index:
            continue
        seen.add(pid)
        result.append(pid)
        queue.extend(children.get(pid, ()))
    return result


def _resolve_pid(pid: int | None, match: str | None) -> int:
    if pid is not None:
        if _parse_stat(PROC / str(pid) / "stat") is None:
            raise SystemExit(f"process does not exist or is not readable: {pid}")
        return pid
    if not match:
        raise SystemExit("specify --pid PID or --match REGEX")
    try:
        pattern = re.compile(match)
    except re.error as exc:
        raise SystemExit(f"invalid --match regular expression: {exc}") from exc
    matches: list[tuple[int, str]] = []
    for candidate, stat in _process_index().items():
        if candidate == os.getpid():
            continue
        command = _cmdline(candidate)
        searchable = f"{stat['comm']} {command}"
        if pattern.search(searchable):
            matches.append((candidate, command or str(stat["comm"])))
    if not matches:
        raise SystemExit(f"no process matches: {match!r}")
    if len(matches) > 1:
        details = "\n".join(f"  {item_pid}: {command}" for item_pid, command in matches)
        raise SystemExit(
            f"multiple processes match {match!r}; choose one with --pid:\n{details}"
        )
    return matches[0][0]


def _scheduler(tid: int) -> tuple[str, int, bool]:
    raw_policy = os.sched_getscheduler(tid)
    reset = bool(raw_policy & RESET_ON_FORK)
    policy = raw_policy & POLICY_MASK
    name = POLICY_NAMES.get(policy, str(policy))
    priority = os.sched_getparam(tid).sched_priority
    return name, priority, reset


def _task_snapshot(pid: int, tid: int) -> dict[str, object] | None:
    task_root = PROC / str(pid) / "task" / str(tid)
    stat = _parse_stat(task_root / "stat")
    if stat is None:
        return None
    status = _parse_key_values(task_root / "status")
    sched = _parse_key_values(task_root / "sched")
    try:
        affinity = sorted(os.sched_getaffinity(tid))
        policy, rt_priority, reset_on_fork = _scheduler(tid)
        nice = os.getpriority(os.PRIO_PROCESS, tid)
    except (PermissionError, ProcessLookupError, OSError):
        return None
    return {
        "pid": pid,
        "tid": tid,
        "main_thread": tid == pid,
        "name": _read(task_root / "comm") or str(stat["comm"]),
        "state": stat["state"],
        "last_cpu": stat["last_cpu"],
        "affinity": affinity,
        "affinity_list": status.get("Cpus_allowed_list", _format_cpu_set(affinity)),
        "policy": policy,
        "reset_on_fork": reset_on_fork,
        "rt_priority": rt_priority,
        "nice": nice,
        "kernel_priority": stat["kernel_priority"],
        "runtime_ticks": int(stat["utime_ticks"]) + int(stat["stime_ticks"]),
        "voluntary_context_switches": _integer_prefix(
            status.get("voluntary_ctxt_switches")
        ),
        "involuntary_context_switches": _integer_prefix(
            status.get("nonvoluntary_ctxt_switches")
        ),
        "migrations": _integer_prefix(
            sched.get("se.nr_migrations", sched.get("nr_migrations"))
        ),
        "sum_exec_runtime_ms": _float_value(sched.get("se.sum_exec_runtime")),
        "uclamp_min": _float_value(sched.get("uclamp.min")),
        "uclamp_max": _float_value(sched.get("uclamp.max")),
    }


def _format_cpu_set(cpus: Iterable[int]) -> str:
    values = sorted(set(cpus))
    if not values:
        return "-"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _parse_cpu_set(value: str | None) -> frozenset[int]:
    if not value:
        return frozenset()
    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return frozenset(cpus)


def _physical_core_cpus(cpu: int) -> frozenset[int]:
    siblings = _read(SYS_CPU / f"cpu{cpu}/topology/thread_siblings_list")
    try:
        return _parse_cpu_set(siblings) or frozenset((cpu,))
    except ValueError:
        return frozenset((cpu,))


def _scheduling_analysis(sample: dict[str, object]) -> dict[str, object]:
    tasks = [
        task
        for process in sample["processes"]
        for task in process["tasks"]
    ]
    realtime = [
        task
        for task in tasks
        if task["policy"] in {"FIFO", "RR", "DEADLINE"}
        or int(task["rt_priority"]) > 0
    ]
    control_cpus = frozenset(
        cpu for task in realtime for cpu in task["affinity"]
    )
    reserved_cpus = frozenset().union(
        *(_physical_core_cpus(cpu) for cpu in control_cpus)
    )
    realtime_keys = {(int(task["pid"]), int(task["tid"])) for task in realtime}
    overlaps = []
    for task in tasks:
        key = (int(task["pid"]), int(task["tid"]))
        overlap = reserved_cpus & frozenset(task["affinity"])
        if key in realtime_keys or not overlap:
            continue
        overlaps.append(
            {
                "pid": key[0],
                "tid": key[1],
                "name": task["name"],
                "policy": task["policy"],
                "affinity": list(task["affinity"]),
                "reserved_overlap": sorted(overlap),
            }
        )
    if not realtime:
        status = "inactive"
    elif len(realtime) == 1 and not overlaps:
        status = "pass"
    else:
        status = "warning"
    return {
        "status": status,
        "realtime_threads": [
            {
                "pid": int(task["pid"]),
                "tid": int(task["tid"]),
                "name": task["name"],
                "policy": task["policy"],
                "rt_priority": int(task["rt_priority"]),
                "affinity": list(task["affinity"]),
            }
            for task in realtime
        ],
        "control_cpus": sorted(control_cpus),
        "reserved_physical_core_cpus": sorted(reserved_cpus),
        "non_control_reserved_overlaps": overlaps,
    }


def _limits(pid: int) -> dict[str, str]:
    raw = _read(PROC / str(pid) / "limits")
    if raw is None:
        return {}
    wanted = {
        "Max nice priority",
        "Max realtime priority",
        "Max locked memory",
    }
    result: dict[str, str] = {}
    for line in raw.splitlines()[1:]:
        for name in wanted:
            if line.startswith(name):
                fields = line[len(name) :].split()
                if len(fields) >= 2:
                    result[name] = f"soft={fields[0]}, hard={fields[1]}"
    return result


def _cgroup_info(pid: int) -> dict[str, object]:
    raw = _read(PROC / str(pid) / "cgroup")
    if raw is None:
        return {}
    unified_path: str | None = None
    memberships: list[str] = []
    for line in raw.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        memberships.append(line)
        if hierarchy == "0" and controllers == "":
            unified_path = path
    result: dict[str, object] = {"memberships": memberships}
    if unified_path is None:
        return result
    root = CGROUP / unified_path.lstrip("/")
    result["v2_path"] = unified_path
    for name in (
        "cpu.max",
        "cpu.weight",
        "cpuset.cpus",
        "cpuset.cpus.effective",
        "cpuset.mems.effective",
    ):
        value = _read(root / name)
        if value is not None:
            result[name] = value
    return result


def _process_snapshot(pid: int, stat: dict[str, object]) -> dict[str, object] | None:
    task_root = PROC / str(pid) / "task"
    try:
        tids = sorted(int(entry.name) for entry in task_root.iterdir() if entry.name.isdigit())
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    tasks = [task for tid in tids if (task := _task_snapshot(pid, tid)) is not None]
    status = _parse_key_values(PROC / str(pid) / "status")
    return {
        "pid": pid,
        "ppid": int(stat["ppid"]),
        "pgrp": int(stat["pgrp"]),
        "session": int(stat["session"]),
        "name": status.get("Name", str(stat["comm"])),
        "command": _cmdline(pid),
        "state": stat["state"],
        "threads": len(tasks),
        "runtime_ticks": int(stat["utime_ticks"]) + int(stat["stime_ticks"]),
        "starttime_ticks": int(stat["starttime_ticks"]),
        "oom_score_adj": _read(PROC / str(pid) / "oom_score_adj"),
        "limits": _limits(pid),
        "cgroup": _cgroup_info(pid),
        "tasks": tasks,
    }


def _cpu_info(cpus: Iterable[int]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for cpu in sorted(set(cpus)):
        root = SYS_CPU / f"cpu{cpu}"
        result.append(
            {
                "cpu": cpu,
                "online": _read_int(root / "online") not in (0,),
                "package": _read_int(root / "topology/physical_package_id"),
                "core": _read_int(root / "topology/core_id"),
                "capacity": _read_int(root / "cpu_capacity"),
                "max_khz": _read_int(root / "cpufreq/cpuinfo_max_freq"),
                "current_khz": _read_int(root / "cpufreq/scaling_cur_freq"),
                "governor": _read(root / "cpufreq/scaling_governor"),
            }
        )
    return result


def _kernel_scheduler_info() -> dict[str, object]:
    return {
        "kernel": platform.release(),
        "clock_ticks_per_second": CLK_TCK,
        "sched_rt_period_us": _read_int(PROC / "sys/kernel/sched_rt_period_us"),
        "sched_rt_runtime_us": _read_int(PROC / "sys/kernel/sched_rt_runtime_us"),
        "sched_autogroup_enabled": _read_int(PROC / "sys/kernel/sched_autogroup_enabled"),
    }


def _sample(root_pid: int) -> dict[str, object]:
    index = _process_index()
    pids = _descendants(root_pid, index)
    if not pids:
        raise ProcessLookupError(root_pid)
    processes = [
        process
        for pid in pids
        if (process := _process_snapshot(pid, index[pid])) is not None
    ]
    sample = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "root_pid": root_pid,
        "processes": processes,
    }
    sample["scheduling_analysis"] = _scheduling_analysis(sample)
    return sample


def _add_rates(
    sample: dict[str, object],
    previous: dict[tuple[int, int, int], tuple[int, int]] | None,
) -> dict[tuple[int, int, int], tuple[int, int]]:
    now_ns = int(sample["monotonic_ns"])
    current: dict[tuple[int, int, int], tuple[int, int]] = {}
    for process in sample["processes"]:
        pid = int(process["pid"])
        starttime = int(process["starttime_ticks"])
        process_key = (pid, 0, starttime)
        process_ticks = int(process["runtime_ticks"])
        current[process_key] = (now_ns, process_ticks)
        process["cpu_percent"] = _cpu_percent(previous, process_key, now_ns, process_ticks)
        for task in process["tasks"]:
            tid = int(task["tid"])
            key = (pid, tid, starttime)
            ticks = int(task["runtime_ticks"])
            current[key] = (now_ns, ticks)
            task["cpu_percent"] = _cpu_percent(previous, key, now_ns, ticks)
    return current


def _cpu_percent(
    previous: dict[tuple[int, int, int], tuple[int, int]] | None,
    key: tuple[int, int, int],
    now_ns: int,
    ticks: int,
) -> float | None:
    if previous is None or key not in previous:
        return None
    old_ns, old_ticks = previous[key]
    elapsed = (now_ns - old_ns) / 1_000_000_000.0
    if elapsed <= 0.0:
        return None
    return max(0.0, (ticks - old_ticks) / CLK_TCK / elapsed * 100.0)


def _format_optional(value: object, width: int = 0) -> str:
    text = "-" if value is None else str(value)
    return f"{text:>{width}}" if width else text


def _print_system(sample: dict[str, object]) -> None:
    used_cpus = {
        cpu
        for process in sample["processes"]
        for task in process["tasks"]
        for cpu in task["affinity"]
    }
    used_cpus.update(sample["scheduling_analysis"]["reserved_physical_core_cpus"])
    kernel = _kernel_scheduler_info()
    print(
        "Kernel scheduling: "
        f"kernel={kernel['kernel']}, HZ={kernel['clock_ticks_per_second']}, "
        f"rt_period_us={kernel['sched_rt_period_us']}, "
        f"rt_runtime_us={kernel['sched_rt_runtime_us']}, "
        f"autogroup={kernel['sched_autogroup_enabled']}"
    )
    cpu_info = _cpu_info(used_cpus)
    print("CPU topology used by the process tree:")
    print(
        f"{'CPU':>4} {'online':>6} {'pkg':>4} {'core':>5} "
        f"{'capacity':>8} {'max MHz':>8} {'cur MHz':>8} governor"
    )
    for item in cpu_info:
        max_mhz = None if item["max_khz"] is None else float(item["max_khz"]) / 1000.0
        cur_mhz = None if item["current_khz"] is None else float(item["current_khz"]) / 1000.0
        print(
            f"{item['cpu']:>4} {str(item['online']):>6} "
            f"{_format_optional(item['package'], 4)} "
            f"{_format_optional(item['core'], 5)} "
            f"{_format_optional(item['capacity'], 8)} "
            f"{_format_optional(None if max_mhz is None else f'{max_mhz:.0f}', 8)} "
            f"{_format_optional(None if cur_mhz is None else f'{cur_mhz:.0f}', 8)} "
            f"{item['governor'] or '-'}"
        )
    print()


def _print_sample(sample: dict[str, object], *, show_threads: bool) -> None:
    print(f"Snapshot {sample['timestamp']}  root_pid={sample['root_pid']}")
    for process in sample["processes"]:
        cpu = process.get("cpu_percent")
        cpu_text = "-" if cpu is None else f"{cpu:.1f}%"
        print(
            f"\nPID {process['pid']}  PPID {process['ppid']}  PGRP {process['pgrp']}  "
            f"SID {process['session']}  threads={process['threads']}  cpu={cpu_text}"
        )
        print(f"  name: {process['name']}")
        print(f"  cmd:  {process['command'] or '-'}")
        print(f"  oom_score_adj: {process['oom_score_adj'] or '-'}")
        limits = process["limits"]
        if limits:
            print("  limits: " + "; ".join(f"{key} {value}" for key, value in limits.items()))
        cgroup = process["cgroup"]
        if cgroup:
            fields = [f"path={cgroup.get('v2_path', '-')}"]
            for key in ("cpu.max", "cpu.weight", "cpuset.cpus.effective", "cpuset.mems.effective"):
                if key in cgroup:
                    fields.append(f"{key}={cgroup[key]}")
            print("  cgroup: " + ", ".join(fields))

    print()
    print(
        f"{'PID':>7} {'TID':>7} {'role':>6} {'thread':<18} {'S':>1} "
        f"{'CPU':>4} {'CPU%':>7} {'affinity':<13} {'policy':<12} "
        f"{'RT':>3} {'nice':>5} {'prio':>5} {'vcsw':>8} {'ivcsw':>8} {'migr':>7}"
    )
    print("-" * 133)
    for process in sample["processes"]:
        for task in process["tasks"]:
            if not show_threads and not task["main_thread"]:
                continue
            cpu = task.get("cpu_percent")
            cpu_text = "-" if cpu is None else f"{cpu:.1f}"
            policy = task["policy"] + ("+R" if task["reset_on_fork"] else "")
            print(
                f"{task['pid']:>7} {task['tid']:>7} "
                f"{('main' if task['main_thread'] else 'thread'):>6} "
                f"{task['name'][:18]:<18} {task['state']:>1} "
                f"{task['last_cpu']:>4} {cpu_text:>7} "
                f"{task['affinity_list']:<13} {policy:<12} "
                f"{task['rt_priority']:>3} {task['nice']:>5} "
                f"{task['kernel_priority']:>5} "
                f"{_format_optional(task['voluntary_context_switches'], 8)} "
                f"{_format_optional(task['involuntary_context_switches'], 8)} "
                f"{_format_optional(task['migrations'], 7)}"
            )

    analysis = sample["scheduling_analysis"]
    realtime = analysis["realtime_threads"]
    overlaps = analysis["non_control_reserved_overlaps"]
    if analysis["status"] == "inactive":
        print("\nScheduling isolation: INACTIVE (no real-time control thread found)")
        return
    label = "PASS" if analysis["status"] == "pass" else "WARNING"
    control_text = ", ".join(
        f"{item['pid']}/{item['tid']} {item['policy']}/{item['rt_priority']}"
        for item in realtime
    )
    print(
        f"\nScheduling isolation: {label}; control={control_text}; "
        f"CPU={_format_cpu_set(analysis['control_cpus'])}; "
        "reserved physical core="
        f"{_format_cpu_set(analysis['reserved_physical_core_cpus'])}; "
        f"non-control overlaps={len(overlaps)}"
    )
    for item in overlaps[:10]:
        print(
            "  overlap: "
            f"PID/TID={item['pid']}/{item['tid']} thread={item['name']} "
            f"policy={item['policy']} affinity={_format_cpu_set(item['affinity'])} "
            f"reserved={_format_cpu_set(item['reserved_overlap'])}"
        )
    if len(overlaps) > 10:
        print(f"  ... {len(overlaps) - 10} more overlapping threads")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect CPU affinity, scheduling policy and priority for a Linux "
            "process, all descendants, and every thread."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int, help="PID of the framework main process")
    target.add_argument("--match", help="regular expression matched against comm and cmdline")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of snapshots; 0 watches until Ctrl-C",
    )
    parser.add_argument(
        "--no-threads",
        action="store_true",
        help="show only each process main thread",
    )
    parser.add_argument("--json", type=Path, help="write all collected snapshots as JSON")
    args = parser.parse_args()
    if args.interval <= 0.0:
        parser.error("--interval must be positive")
    if args.count < 0:
        parser.error("--count must be non-negative")

    root_pid = _resolve_pid(args.pid, args.match)
    samples: list[dict[str, object]] = []
    previous = None
    first = True
    index = 0
    try:
        while args.count == 0 or index < args.count:
            if index:
                time.sleep(args.interval)
            try:
                sample = _sample(root_pid)
            except ProcessLookupError:
                print(f"target process exited: {root_pid}", file=sys.stderr)
                break
            previous = _add_rates(sample, previous)
            if first:
                _print_system(sample)
                first = False
            _print_sample(sample, show_threads=not args.no_threads)
            samples.append(sample)
            index += 1
    except KeyboardInterrupt:
        print("\nStopped.")

    if args.json is not None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kernel": _kernel_scheduler_info(),
            "samples": samples,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json.resolve()}")

"""
先找到主进程：

pgrep -af bxi_example_py

然后持续观察：

python3 tools/diagnostics/process_scheduling.py \
--pid 主进程PID \
--count 10 \
--interval 1 \
--json tools/benchmark/results/scheduling-my-platform.json
"""

if __name__ == "__main__":
    main()

    

