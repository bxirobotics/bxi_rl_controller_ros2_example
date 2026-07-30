"""Platform-neutral CPU topology discovery and framework-wide affinity roles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path


class CpuAffinityRole(str, Enum):
    """Stable role names shared by every supported platform."""

    CONTROL = "control"
    COMPUTE = "compute"
    BACKGROUND = "background"
    SHARED = "shared"
    ALL = "all"
    INHERIT = "inherit"


@dataclass(frozen=True)
class CpuAffinitySpec:
    """One validated role or an explicit compatibility CPU set."""

    role: CpuAffinityRole | None = None
    cpus: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        if self.role is not None:
            return self.role.value
        return format_cpu_set(self.cpus)


@dataclass(frozen=True)
class CpuTopology:
    """CPU facts discovered without embedding platform-specific core numbers."""

    allowed_cpus: frozenset[int]
    physical_cores: tuple[frozenset[int], ...]
    capacity_by_cpu: Mapping[int, int]

    @classmethod
    def discover(
        cls,
        *,
        sysfs_root: Path = Path("/sys/devices/system/cpu"),
    ) -> "CpuTopology":
        allowed = _allowed_cpus()
        physical = _physical_cores(allowed, sysfs_root)
        capacities = {
            cpu: _cpu_capacity(cpu, sysfs_root)
            for cpu in allowed
        }
        return cls(
            allowed_cpus=frozenset(allowed),
            physical_cores=physical,
            capacity_by_cpu=capacities,
        )


@dataclass(frozen=True)
class CpuAffinityPlan:
    """Resolved CPU sets for the framework's stable affinity roles."""

    topology: CpuTopology
    roles: Mapping[CpuAffinityRole, frozenset[int]]
    reserved_control_core: frozenset[int]

    @classmethod
    def discover(cls) -> "CpuAffinityPlan":
        return cls.from_topology(CpuTopology.discover())

    @classmethod
    def from_topology(cls, topology: CpuTopology) -> "CpuAffinityPlan":
        allowed = topology.allowed_cpus
        if not allowed:
            raise RuntimeError("CPU affinity discovery found no allowed CPUs")

        groups = topology.physical_cores or tuple(
            frozenset((cpu,)) for cpu in sorted(allowed)
        )

        def group_score(group: frozenset[int]) -> int:
            return max(topology.capacity_by_cpu.get(cpu, 1) for cpu in group)

        max_score = max(group_score(group) for group in groups)
        performance_groups = tuple(
            group for group in groups if group_score(group) == max_score
        )
        efficiency_groups = tuple(
            group for group in groups if group_score(group) < max_score
        )
        control_group = max(
            performance_groups,
            key=lambda group: (group_score(group), max(group)),
        )
        control_cpu = min(control_group)
        shared = allowed - control_group
        if not shared:
            shared = allowed

        compute = frozenset().union(
            *(group for group in performance_groups if group != control_group)
        )
        if not compute:
            compute = shared
        background = frozenset().union(*efficiency_groups)
        if not background:
            background = shared

        roles = {
            CpuAffinityRole.CONTROL: frozenset((control_cpu,)),
            CpuAffinityRole.COMPUTE: frozenset(compute),
            CpuAffinityRole.BACKGROUND: frozenset(background),
            CpuAffinityRole.SHARED: frozenset(shared),
            CpuAffinityRole.ALL: allowed,
            CpuAffinityRole.INHERIT: allowed,
        }
        return cls(
            topology=topology,
            roles=roles,
            reserved_control_core=control_group,
        )

    @property
    def allowed_cpus(self) -> frozenset[int]:
        return self.topology.allowed_cpus

    def resolve(self, spec: CpuAffinitySpec, *, context: str) -> frozenset[int] | None:
        if spec.role is CpuAffinityRole.INHERIT:
            return None
        if spec.role is not None:
            return self.roles[spec.role]
        requested = frozenset(spec.cpus)
        unavailable = requested - self.allowed_cpus
        if unavailable:
            raise ValueError(
                f"{context} requests CPUs outside the process allowance: "
                f"{format_cpu_set(unavailable)}; allowed="
                f"{format_cpu_set(self.allowed_cpus)}"
            )
        return requested

    def describe(self) -> str:
        role_text = ", ".join(
            f"{role.value}={format_cpu_set(self.roles[role])}"
            for role in (
                CpuAffinityRole.CONTROL,
                CpuAffinityRole.COMPUTE,
                CpuAffinityRole.BACKGROUND,
                CpuAffinityRole.SHARED,
                CpuAffinityRole.ALL,
            )
        )
        return f"CPU affinity roles: {role_text}"


def configure_current_thread(
    cpus: Sequence[int] | frozenset[int] | None,
    *,
    realtime_priority: int = 0,
) -> None:
    """Apply one complete scheduling role to the calling Linux thread."""

    resolved = None if cpus is None else frozenset(cpus)
    if resolved is not None and not resolved:
        raise ValueError("thread CPU affinity must not be empty")
    if not 0 <= realtime_priority <= 99:
        raise ValueError("thread realtime priority must be in [0, 99]")

    if realtime_priority == 0:
        # Drop inherited real-time policy before moving foreign/background work.
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        if resolved is not None:
            os.sched_setaffinity(0, resolved)
        return

    # Pin first so a newly promoted real-time thread cannot execute elsewhere.
    if resolved is not None:
        os.sched_setaffinity(0, resolved)
    os.sched_setscheduler(
        0,
        os.SCHED_FIFO,
        os.sched_param(realtime_priority),
    )


def bootstrap_process_scheduling() -> CpuAffinityPlan:
    """Discover the full topology, then establish the non-control baseline."""

    plan = CpuAffinityPlan.discover()
    configure_current_thread(
        plan.roles[CpuAffinityRole.SHARED],
        realtime_priority=0,
    )
    return plan


def read_cpu_affinity(
    value: object,
    context: str,
    *,
    default: CpuAffinityRole,
) -> CpuAffinitySpec:
    """Validate one role, legacy integer, or explicit CPU list."""

    if value is None:
        return CpuAffinitySpec(role=default)
    if isinstance(value, str):
        try:
            return CpuAffinitySpec(role=CpuAffinityRole(value))
        except ValueError as exc:
            allowed = ", ".join(role.value for role in CpuAffinityRole)
            raise ValueError(f"{context} must be one of: {allowed}") from exc
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a CPU role, index or index list")
    if isinstance(value, int):
        if value == -1:
            return CpuAffinitySpec(role=CpuAffinityRole.INHERIT)
        if value < 0:
            raise ValueError(f"{context} CPU index must be non-negative or -1")
        return CpuAffinitySpec(cpus=(value,))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a CPU role, index or index list")
    cpus: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context} entries must be non-negative CPU indices")
        cpus.append(item)
    if not cpus:
        raise ValueError(f"{context} CPU index list must not be empty")
    return CpuAffinitySpec(cpus=tuple(dict.fromkeys(cpus)))


def format_cpu_set(cpus: Sequence[int] | frozenset[int]) -> str:
    values = sorted(set(cpus))
    if not values:
        return "none"
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


def _allowed_cpus() -> frozenset[int]:
    try:
        return frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
        return frozenset(range(count))


def _physical_cores(
    allowed: frozenset[int], sysfs_root: Path
) -> tuple[frozenset[int], ...]:
    remaining = set(allowed)
    groups: list[frozenset[int]] = []
    while remaining:
        cpu = min(remaining)
        siblings_path = sysfs_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        try:
            siblings = _parse_cpu_list(siblings_path.read_text(encoding="utf-8"))
        except OSError:
            siblings = frozenset((cpu,))
        group = frozenset(siblings & allowed) or frozenset((cpu,))
        groups.append(group)
        remaining.difference_update(group)
    return tuple(groups)


def _cpu_capacity(cpu: int, sysfs_root: Path) -> int:
    candidates = (
        sysfs_root / f"cpu{cpu}" / "cpu_capacity",
        sysfs_root / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq",
        sysfs_root / f"cpu{cpu}" / "cpufreq" / "scaling_max_freq",
    )
    for path in candidates:
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if value > 0:
            return value
    return 1


def _parse_cpu_list(value: str) -> frozenset[int]:
    cpus: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return frozenset(cpus)


__all__ = [
    "bootstrap_process_scheduling",
    "configure_current_thread",
    "CpuAffinityPlan",
    "CpuAffinityRole",
    "CpuAffinitySpec",
    "CpuTopology",
    "format_cpu_set",
    "read_cpu_affinity",
]
