"""Low-overhead per-stage inference timing."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass(frozen=True, slots=True)
class TimingSummary:
    samples: int
    p50_us: float
    p95_us: float
    p99_us: float
    maximum_us: float


class _TimingRing:
    def __init__(self, capacity: int) -> None:
        self.values = np.empty(capacity, dtype=np.int64)
        self.capacity = capacity
        self.count = 0
        self.index = 0

    def append(self, value: int) -> None:
        self.values[self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def snapshot(self) -> np.ndarray:
        return self.values[: self.count].copy()


class InferenceMonitor:
    """Collect timings on the hot path and summarize them off-path."""

    STAGES = ("input", "backend", "output", "total")

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("monitor capacity must be greater than zero")
        self._rings: dict[str, dict[str, _TimingRing]] = {}
        self._capacity = int(capacity)
        self._lock = Lock()

    def record(
        self,
        policy: str,
        input_ns: int,
        backend_ns: int,
        output_ns: int,
        total_ns: int,
    ) -> None:
        rings = self._rings.get(policy)
        if rings is None:
            # Policy creation and first inference are outside steady state; a
            # short lock here avoids burdening every subsequent control cycle.
            with self._lock:
                rings = self._rings.setdefault(
                    policy,
                    {stage: _TimingRing(self._capacity) for stage in self.STAGES},
                )
        rings["input"].append(int(input_ns))
        rings["backend"].append(int(backend_ns))
        rings["output"].append(int(output_ns))
        rings["total"].append(int(total_ns))

    def summary(self, policy: str) -> dict[str, TimingSummary]:
        rings = self._rings.get(policy)
        if rings is None:
            return {}
        result = {}
        for stage, ring in rings.items():
            values = ring.snapshot()
            if values.size == 0:
                continue
            percentiles = np.percentile(values, [50, 95, 99]) / 1_000.0
            result[stage] = TimingSummary(
                samples=int(values.size),
                p50_us=float(percentiles[0]),
                p95_us=float(percentiles[1]),
                p99_us=float(percentiles[2]),
                maximum_us=float(np.max(values) / 1_000.0),
            )
        return result


__all__ = ["InferenceMonitor", "TimingSummary"]
