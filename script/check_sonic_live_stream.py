#!/usr/bin/env python3
"""Validate live ELF3 SONIC ``pose`` and ``smpl_ref`` ZMQ streams.

Run this after PICO body data is available, ABXY calibration has completed and
A+X has switched the manager to POSE.  The check is read-only: it subscribes to
the two local streams and never publishes commands to the robot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import time

import numpy as np
import zmq


HEADER_SIZE = 1280
DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}


def scalar(fields: dict[str, np.ndarray], name: str, default):
    value = fields.get(name)
    if value is None:
        return default
    flat = np.asarray(value).reshape(-1)
    return flat[-1].item() if flat.size else default


def decode_message(message: bytes, topic: str) -> dict[str, np.ndarray]:
    prefix = topic.encode("utf-8")
    if not message.startswith(prefix):
        raise ValueError(f"message does not start with topic {topic!r}")
    offset = len(prefix)
    header_raw = message[offset : offset + HEADER_SIZE]
    if len(header_raw) != HEADER_SIZE:
        raise ValueError("truncated packed-message header")
    header = json.loads(header_raw.split(b"\0", 1)[0].decode("utf-8"))
    offset += HEADER_SIZE
    result: dict[str, np.ndarray] = {}
    for spec in header.get("fields", []):
        name = str(spec["name"])
        dtype_name = str(spec["dtype"])
        if dtype_name not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r} for field {name!r}")
        shape = tuple(int(value) for value in spec["shape"])
        if any(value < 0 for value in shape):
            raise ValueError(f"negative shape for field {name!r}: {shape}")
        dtype = DTYPES[dtype_name]
        count = math.prod(shape)
        size = count * dtype.itemsize
        end = offset + size
        if end > len(message):
            raise ValueError(f"truncated field {name!r}")
        result[name] = np.frombuffer(message[offset:end], dtype=dtype).reshape(shape).copy()
        offset = end
    return result


@dataclass
class StreamStats:
    name: str
    count: int = 0
    decode_errors: int = 0
    first_frame: int | None = None
    last_frame: int | None = None
    advancing: int = 0
    non_advancing: int = 0
    mode_ready: int = 0
    first_values: dict[str, np.ndarray] = field(default_factory=dict)
    max_delta: dict[str, float] = field(default_factory=dict)
    last_fields: dict[str, np.ndarray] = field(default_factory=dict)

    def update(self, fields: dict[str, np.ndarray]) -> None:
        self.count += 1
        frame = int(scalar(fields, "frame_index", -1))
        if self.first_frame is None:
            self.first_frame = frame
        if self.last_frame is not None:
            if frame > self.last_frame:
                self.advancing += 1
            else:
                self.non_advancing += 1
        self.last_frame = frame

        if self.name == "pose":
            ready = int(scalar(fields, "stream_mode", -1)) == 1 and bool(
                scalar(fields, "calibration_ready", False)
            )
            watched = ()
        else:
            ready = bool(scalar(fields, "source_ready", False)) and int(
                scalar(fields, "source_stream_mode", -1)
            ) == 1
            watched = ("term1_local", "root_quat", "wrist")
        if ready:
            self.mode_ready += 1

        for name in watched:
            if name not in fields:
                continue
            value = np.asarray(fields[name], dtype=np.float32)
            if name not in self.first_values:
                self.first_values[name] = value.copy()
                self.max_delta[name] = 0.0
            elif value.shape == self.first_values[name].shape:
                delta = float(np.max(np.abs(value - self.first_values[name])))
                self.max_delta[name] = max(self.max_delta[name], delta)
        self.last_fields = fields


def validate(stats: StreamStats, elapsed: float, min_rate: float) -> list[str]:
    failures: list[str] = []
    rate = stats.count / max(elapsed, 1.0e-6)
    if stats.count == 0:
        return ["no messages received"]
    if stats.decode_errors:
        failures.append(f"{stats.decode_errors} packed messages failed to decode")
    if rate < min_rate:
        failures.append(f"rate {rate:.1f} Hz is below {min_rate:.1f} Hz")
    if stats.first_frame is None or stats.last_frame is None or stats.last_frame <= stats.first_frame:
        failures.append(
            f"frame_index did not advance ({stats.first_frame} -> {stats.last_frame})"
        )
    transitions = stats.advancing + stats.non_advancing
    if transitions and stats.advancing / transitions < 0.8:
        failures.append(
            f"only {stats.advancing}/{transitions} frame transitions advanced"
        )
    if stats.mode_ready / stats.count < 0.9:
        failures.append(
            f"only {stats.mode_ready}/{stats.count} messages were ready in POSE mode"
        )

    fields = stats.last_fields
    if stats.name == "pose":
        if int(scalar(fields, "stream_mode", -1)) != 1:
            failures.append("latest stream_mode is not POSE (1)")
        if not bool(scalar(fields, "calibration_ready", False)):
            failures.append("latest calibration_ready is false")
    else:
        required_shapes = {
            "term1_local": (10, 72),
            "root_quat": (10, 4),
            "wrist": (10, 6),
        }
        if not bool(scalar(fields, "source_ready", False)):
            failures.append("latest source_ready is false")
        if int(scalar(fields, "source_stream_mode", -1)) != 1:
            failures.append("latest source_stream_mode is not POSE (1)")
        for name, shape in required_shapes.items():
            value = fields.get(name)
            if value is None:
                failures.append(f"missing {name}")
                continue
            if tuple(value.shape) != shape:
                failures.append(f"{name} shape is {value.shape}, expected {shape}")
            if not np.isfinite(value).all():
                failures.append(f"{name} contains non-finite values")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0, help="sample seconds")
    parser.add_argument("--min-rate", type=float, default=30.0, help="minimum pass rate in Hz")
    parser.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--smpl-endpoint", default="tcp://127.0.0.1:5557")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.min_rate <= 0:
        raise SystemExit("--duration and --min-rate must be positive")

    context = zmq.Context()
    poller = zmq.Poller()
    sockets: dict[zmq.Socket, tuple[str, StreamStats]] = {}
    for topic, endpoint in (("pose", args.pose_endpoint), ("smpl_ref", args.smpl_endpoint)):
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 100)
        socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        socket.connect(endpoint)
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = (topic, StreamStats(topic))
        print(f"[live-check] SUB {endpoint} topic={topic!r}")

    started = time.monotonic()
    try:
        while time.monotonic() - started < args.duration:
            events = dict(poller.poll(timeout=100))
            for socket, event in events.items():
                if not event & zmq.POLLIN:
                    continue
                topic, stats = sockets[socket]
                while True:
                    try:
                        message = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        stats.update(decode_message(message, topic))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        stats.decode_errors += 1
                        if stats.decode_errors <= 3:
                            print(f"[live-check] {topic} decode error: {exc}")
    except KeyboardInterrupt:
        print("\n[live-check] interrupted")
        return 130
    finally:
        for socket in sockets:
            poller.unregister(socket)
            socket.close(linger=0)
        context.term()

    elapsed = time.monotonic() - started
    failed = False
    for _socket, (topic, stats) in sockets.items():
        rate = stats.count / max(elapsed, 1.0e-6)
        failures = validate(stats, elapsed, args.min_rate)
        print(
            f"[{topic}] count={stats.count} rate={rate:.1f}Hz "
            f"frame={stats.first_frame}->{stats.last_frame} "
            f"advanced={stats.advancing} ready={stats.mode_ready}/{stats.count}"
        )
        if topic == "smpl_ref":
            print(
                "[smpl_ref] max_delta_from_first "
                + " ".join(
                    f"{name}={stats.max_delta.get(name, 0.0):.6g}"
                    for name in ("term1_local", "root_quat", "wrist")
                )
            )
            if stats.max_delta and max(stats.max_delta.values()) <= 1.0e-6:
                print("[WARN] smpl_ref values are effectively static during the sample")
        if failures:
            failed = True
            for failure in failures:
                print(f"[FAIL] {topic}: {failure}")
        else:
            print(f"[PASS] {topic}: live stream is healthy")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
