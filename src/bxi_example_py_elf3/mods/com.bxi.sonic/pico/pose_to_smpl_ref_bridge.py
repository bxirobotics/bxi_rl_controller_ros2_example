"""Adapt the official PICO ``pose`` stream into ordered ELF3 source chunks.

The official ``gear_sonic/scripts/pico_manager_thread_server.py --manager`` sends
packed ZMQ messages on topic ``pose``.  For the ELF3 native _smpl.onnx deploy we
publish complete rolling chunks for the downstream SONIC policy:

    term1_local    : float32 [N,72] SMPL joints local, flattened per frame
    root_quat      : float32 [N,4]  SMPL root quaternion, wxyz
    wrist          : float32 [N,6]  ELF3 native wrist x/y/z, left then right
    head_joint_pos : float32 [N,2]  ELF3 head pitch/yaw per frame

The bridge owns no playback clock, cursor or acknowledgement channel.  It
forwards each new complete source chunk once; the policy alone merges chunks,
gathers a complete ten-frame window and advances after successful inference.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import zmq

from rclpy.node import Node
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.mod_api import NodeBuildContext

from .zmq_messages import pack_pose_message
from .streamed_smpl_ref import (
    IncomingChunk,
    WINDOW,
    classify_frame_progress,
    new_stream_epoch,
)
from .runtime_config import (
    PICO_HOST,
    PICO_PORT,
    PICO_STALE_SECONDS,
    PICO_TOPIC,
    SMPL_REF_HOST,
    SMPL_REF_PORT,
    SMPL_REF_TOPIC,
)


HEADER_SIZE = 1280
DTYPE_MAP = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}

# The packaged PICO manager writes directly to the ELF3 29-DoF joint order.
# Export the six wrist joints as:
#   l_wrist_x, l_wrist_y, l_wrist_z, r_wrist_x, r_wrist_y, r_wrist_z.
ELF3_NATIVE_WRIST_IDX = [19, 20, 21, 26, 27, 28]
DEFAULT_RATE_HZ = 50.0
POSE_STREAM_MODE = 1
READY_CONSECUTIVE_MESSAGES = 3


PICO_BUTTON_FIELDS = (
    "left_trigger",
    "right_trigger",
    "left_grip",
    "right_grip",
)

BRIDGE_DEFAULTS: dict[str, object] = {
    "pico_host": PICO_HOST,
    "pico_port": PICO_PORT,
    "pico_topic": PICO_TOPIC,
    "out_host": SMPL_REF_HOST,
    "out_port": SMPL_REF_PORT,
    "out_topic": SMPL_REF_TOPIC,
    "rate_hz": DEFAULT_RATE_HZ,
    "stale_warning_seconds": PICO_STALE_SECONDS,
}


def _field_scalar(fields: dict[str, np.ndarray], name: str) -> float | None:
    value = fields.get(name)
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    scalar = float(arr[0])
    return scalar if np.isfinite(scalar) else None


class PicoSourceReadinessGate:
    """Require calibrated POSE metadata and progressing, finite raw frames."""

    def __init__(self, required_consecutive: int = READY_CONSECUTIVE_MESSAGES):
        self.required_consecutive = max(1, int(required_consecutive))
        self.reset()

    def reset(self) -> None:
        self.streak = 0
        self.last_frame_index: int | None = None
        self.last_message_mono: float | None = None
        self.last_ready_mono: float | None = None

    def observe(
        self,
        fields: dict[str, np.ndarray],
        now_mono: float,
        stale_seconds: float,
    ) -> bool:
        if (
            self.last_message_mono is not None
            and now_mono - self.last_message_mono > stale_seconds
        ):
            self.reset()
        self.last_message_mono = now_mono

        frame_index: int | None = None
        try:
            mode = int(np.asarray(fields["stream_mode"]).reshape(-1)[-1])
            calibrated = bool(np.asarray(fields["calibration_ready"]).reshape(-1)[-1])
            frame_index = int(np.asarray(fields["frame_index"]).reshape(-1)[-1])
            finite = all(
                np.asarray(fields[name]).size > 0
                and np.isfinite(np.asarray(fields[name])).all()
                for name in (
                    "smpl_joints",
                    "body_quat_w",
                    "joint_pos",
                    "head_joint_pos",
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            mode, calibrated, finite = -1, False, False

        source_valid = mode == POSE_STREAM_MODE and calibrated and finite
        if (
            source_valid
            and frame_index is not None
            and self.last_frame_index is not None
            and frame_index < self.last_frame_index
        ):
            # PICO restarts its frame counter when a new POSE session starts.
            # Treat the first lower-index packet as frame one of that session;
            # otherwise the gate would wait for the counter to overtake the
            # previous session before live references could become ready again.
            self.streak = 0
            self.last_frame_index = None
            self.last_ready_mono = None

        if not source_valid or frame_index is None:
            self.streak = 0
            self.last_ready_mono = None
            return False

        # A rolling PICO chunk may be observed twice when its source frame is
        # late by one bridge tick.  It is not fresh progress and is filtered by
        # the caller, but it must not revoke an already-established readiness
        # state or force three subsequent progressing chunks to re-arm it.
        if self.last_frame_index is not None and frame_index == self.last_frame_index:
            return self.streak >= self.required_consecutive

        self.last_frame_index = frame_index
        self.streak += 1
        ready = self.streak >= self.required_consecutive
        if ready:
            self.last_ready_mono = now_mono
        return ready

    def is_fresh(self, now_mono: float, stale_seconds: float) -> bool:
        return (
            self.streak >= self.required_consecutive
            and self.last_message_mono is not None
            and self.last_ready_mono is not None
            and now_mono - self.last_ready_mono <= stale_seconds
        )


def _decode_packed_message(msg: bytes, topic: str) -> dict[str, np.ndarray] | None:
    prefix = topic.encode("utf-8")
    if not msg.startswith(prefix):
        return None
    payload = msg[len(prefix) :]
    if len(payload) < HEADER_SIZE:
        return None

    raw_header = payload[:HEADER_SIZE].split(b"\x00", 1)[0]
    if not raw_header:
        return None
    header: dict[str, Any] = json.loads(raw_header.decode("utf-8"))
    data = memoryview(payload[HEADER_SIZE:])

    out: dict[str, np.ndarray] = {}
    offset = 0
    for field in header.get("fields", []):
        name = field["name"]
        dtype = DTYPE_MAP.get(field["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported dtype for field {name}: {field['dtype']}")
        shape = tuple(int(x) for x in field.get("shape", []))
        count = int(np.prod(shape)) if shape else 1
        nbytes = dtype.itemsize * count
        if offset + nbytes > len(data):
            raise ValueError(f"field {name} exceeds payload bounds")
        arr = np.frombuffer(data[offset : offset + nbytes], dtype=dtype, count=count)
        out[name] = arr.reshape(shape).copy()
        offset += nbytes
    return out


def _as_frame_matrix(arr: np.ndarray, width: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, width)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != width:
        raise ValueError(f"{name} has shape {arr.shape}; expected (*,{width})")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _extract_wrist_frames(joint_pos: np.ndarray) -> np.ndarray:
    jp = np.asarray(joint_pos, dtype=np.float32)
    if jp.ndim == 1:
        jp = jp.reshape(1, -1)
    if jp.shape[1] < 29:
        raise ValueError(
            f"joint_pos has shape {jp.shape}; expected at least 29 columns"
        )

    return np.ascontiguousarray(jp[:, ELF3_NATIVE_WRIST_IDX], dtype=np.float32)


def _parse_incoming_chunk(fields: dict[str, np.ndarray]) -> IncomingChunk:
    missing = [
        k
        for k in (
            "frame_index",
            "smpl_joints",
            "body_quat_w",
            "joint_pos",
            "head_joint_pos",
        )
        if k not in fields
    ]
    if missing:
        raise ValueError(f"PICO pose message missing required fields: {missing}")

    smpl_joints = np.asarray(fields["smpl_joints"], dtype=np.float32)
    if smpl_joints.ndim == 2 and smpl_joints.shape[1] == 72:
        term1 = _as_frame_matrix(smpl_joints, 72, "smpl_joints")
    elif smpl_joints.ndim == 3 and smpl_joints.shape[1:] == (24, 3):
        term1 = _as_frame_matrix(
            smpl_joints.reshape(smpl_joints.shape[0], 72),
            72,
            "smpl_joints",
        )
    else:
        raise ValueError(
            f"smpl_joints has shape {smpl_joints.shape}; expected (N,24,3)"
        )

    root_quat = _as_frame_matrix(fields["body_quat_w"], 4, "body_quat_w")
    wrist = _extract_wrist_frames(fields["joint_pos"])
    head_joint_pos = _as_frame_matrix(
        fields["head_joint_pos"], 2, "head_joint_pos"
    )
    frame_indices = np.asarray(fields["frame_index"], dtype=np.int64).reshape(-1)

    n = term1.shape[0]
    if (
        root_quat.shape[0] != n
        or wrist.shape[0] != n
        or head_joint_pos.shape[0] != n
        or frame_indices.shape[0] != n
    ):
        raise ValueError(
            "PICO pose frame count mismatch: "
            f"frame_index={frame_indices.shape[0]} term1={term1.shape[0]} "
            f"root={root_quat.shape[0]} wrist={wrist.shape[0]} "
            f"head={head_joint_pos.shape[0]}"
        )
    if n < WINDOW:
        raise ValueError(
            f"PICO pose message has {n} frames; need at least {WINDOW}"
        )
    if np.any(np.diff(frame_indices) != 1):
        raise ValueError(
            "frame_index must be consecutive with step 1: "
            f"{frame_indices.tolist()}"
        )
    for name, values in (
        ("smpl_joints", term1),
        ("body_quat_w", root_quat),
        ("joint_pos", wrist),
        ("head_joint_pos", head_joint_pos),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    return IncomingChunk(
        frame_indices=np.ascontiguousarray(frame_indices, dtype=np.int64),
        term1_local=term1,
        root_quat=root_quat,
        wrist=wrist,
        head_joint_pos=head_joint_pos,
    )


def _source_chunk_fields(
    chunk: IncomingChunk,
    *,
    source_stream_epoch: int,
    received_monotonic_ns: int,
) -> dict[str, np.ndarray]:
    """Build the one-way rolling chunk consumed by the policy merger."""
    return {
        "source_chunk": np.asarray([1], dtype=np.uint8),
        "source_stream_epoch": np.asarray(
            [source_stream_epoch], dtype=np.int64
        ),
        "source_received_monotonic_ns": np.asarray(
            [received_monotonic_ns], dtype=np.int64
        ),
        "frame_index": np.ascontiguousarray(
            chunk.frame_indices, dtype=np.int64
        ),
        "term1_local": np.ascontiguousarray(
            chunk.term1_local, dtype=np.float32
        ),
        "root_quat": np.ascontiguousarray(
            chunk.root_quat, dtype=np.float32
        ),
        "wrist": np.ascontiguousarray(chunk.wrist, dtype=np.float32),
        "head_joint_pos": np.ascontiguousarray(
            chunk.head_joint_pos, dtype=np.float32
        ),
        "valid_horizon": np.asarray([WINDOW], dtype=np.int32),
        "clamp_slots": np.asarray([0], dtype=np.int32),
    }


def _validated_params(raw: dict[str, object]) -> dict[str, object]:
    unknown = set(raw) - set(BRIDGE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown SONIC bridge params: {sorted(unknown)}")
    params = {
        name: raw.get(name, default) for name, default in BRIDGE_DEFAULTS.items()
    }
    for name in ("pico_host", "pico_topic", "out_host", "out_topic"):
        if not isinstance(params[name], str) or not params[name]:
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("pico_port", "out_port"):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 65535
        ):
            raise ValueError(f"{name} must be an integer from 1 to 65535")
    for name in ("rate_hz", "stale_warning_seconds"):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError(f"{name} must be greater than zero")
    return params


class SmplRefBridgeNode(Node):
    """Non-blocking ROS node owned by the framework's shared executor."""

    def __init__(self, context: NodeBuildContext):
        params = _validated_params(dict(context.params))
        super().__init__(context.node_name, namespace=context.namespace or None)

        self._pico_topic = str(params["pico_topic"])
        self._out_topic = str(params["out_topic"])
        self._stale_seconds = float(params["stale_warning_seconds"])
        self._source_gate = PicoSourceReadinessGate()
        self._button_publishers = {
            name: self.create_publisher(Float32, f"pico/{name}", 10)
            for name in PICO_BUTTON_FIELDS
        }
        self._invalid_button_fields: set[str] = set()
        self._last_received_mono: float | None = None
        self._last_valid_newest_frame: int | None = None
        self._source_stream_epoch = new_stream_epoch()
        self._received = 0
        self._sent = 0
        self._skipped = 0
        self._duplicate_chunks = 0
        self._counter_restarts = 0
        self._send_dropped = 0
        self._stale_was_reported = False
        self._stream_state: str | None = None
        self._closed = False

        self._zmq_context = zmq.Context()
        self._sub = self._zmq_context.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.RCVHWM, 64)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, self._pico_topic)
        self._pico_endpoint = f"tcp://{params['pico_host']}:{params['pico_port']}"
        self._sub.connect(self._pico_endpoint)

        self._pub = self._zmq_context.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.setsockopt(zmq.SNDHWM, 8)
        self._out_endpoint = f"tcp://{params['out_host']}:{params['out_port']}"
        try:
            self._pub.bind(self._out_endpoint)
            self._timer = self.create_timer(1.0 / float(params["rate_hz"]), self._tick)
        except Exception:
            self._sub.close(linger=0)
            self._pub.close(linger=0)
            self._zmq_context.term()
            super().destroy_node()
            raise

        self.get_logger().info(
            f"SONIC bridge SUB {self._pico_endpoint} topic='{self._pico_topic}', "
            f"PUB {self._out_endpoint} topic='{self._out_topic}', "
            f"poll={float(params['rate_hz']):g}Hz, "
            "playback_owner=policy, ack_channel=none"
        )

    def _publish_buttons(self, fields: dict[str, np.ndarray]) -> None:
        for name, publisher in self._button_publishers.items():
            value = _field_scalar(fields, name)
            if value is None:
                if name in fields and name not in self._invalid_button_fields:
                    self.get_logger().warning(
                        f"invalid PICO {name}; retaining the last valid input"
                    )
                    self._invalid_button_fields.add(name)
                continue
            if name in self._invalid_button_fields:
                self.get_logger().info(f"PICO {name} recovered")
                self._invalid_button_fields.remove(name)
            message = Float32()
            message.data = value
            publisher.publish(message)

    def _drain_input(self) -> None:
        while True:
            try:
                message = self._sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                fields = _decode_packed_message(message, self._pico_topic)
            except (KeyError, TypeError, ValueError) as exc:
                self._skipped += 1
                self.get_logger().warning(f"skipped malformed PICO packet: {exc}")
                continue
            if fields is None:
                continue
            self._publish_buttons(fields)
            self._received += 1
            received_mono = time.monotonic()
            if not self._source_gate.observe(
                fields,
                received_mono,
                self._stale_seconds,
            ):
                continue

            try:
                chunk = _parse_incoming_chunk(fields)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                self._skipped += 1
                self.get_logger().warning(f"skipped invalid PICO pose: {exc}")
                continue

            newest = int(chunk.frame_indices[-1])
            progress = classify_frame_progress(
                newest,
                self._last_valid_newest_frame,
            )
            if progress == "duplicate":
                self._duplicate_chunks += 1
                continue
            if progress == "restart":
                previous_epoch = self._source_stream_epoch
                self._source_stream_epoch = new_stream_epoch(previous_epoch)
                self._counter_restarts += 1
                self.get_logger().info(
                    "PICO frame counter restarted; "
                    f"newest={newest} previous={self._last_valid_newest_frame} "
                    f"source_epoch={previous_epoch}->{self._source_stream_epoch}"
                )

            source_fields = _source_chunk_fields(
                chunk,
                source_stream_epoch=self._source_stream_epoch,
                received_monotonic_ns=int(received_mono * 1.0e9),
            )
            try:
                self._pub.send(
                    pack_pose_message(
                        source_fields,
                        topic=self._out_topic,
                        version=5,
                    ),
                    flags=zmq.NOBLOCK,
                )
                self._sent += 1
            except zmq.Again:
                self._send_dropped += 1

            self._last_valid_newest_frame = newest
            self._last_received_mono = received_mono
            self._stale_was_reported = False

    def _tick(self) -> None:
        if self._closed:
            return
        self._drain_input()
        now = time.monotonic()
        input_age = (
            now - self._last_received_mono
            if self._last_received_mono is not None
            else float("inf")
        )
        if (
            self._last_received_mono is not None
            and input_age > self._stale_seconds
            and not self._stale_was_reported
        ):
            self.get_logger().warning(
                "PICO pose input stale; "
                f"age_ms={input_age * 1000.0:.0f} received={self._received}. "
                "No synthetic window is published; policy consumes its buffer "
                "then holds the last complete window."
            )
            self._stale_was_reported = True

        current_state = (
            "streaming"
            if self._last_received_mono is not None and not self._stale_was_reported
            else "stale" if self._stale_was_reported else "waiting"
        )
        if current_state == self._stream_state:
            return
        self._stream_state = current_state
        if current_state == "waiting":
            self.get_logger().info(
                "waiting for calibrated, fresh PICO pose frames; "
                f"received={self._received} skipped={self._skipped}"
            )
        elif current_state == "streaming":
            self.get_logger().info(
                "PICO source chunks ready; "
                f"sent={self._sent} newest={self._last_valid_newest_frame} "
                f"epoch={self._source_stream_epoch} "
                f"duplicates={self._duplicate_chunks} "
                f"restarts={self._counter_restarts} "
                f"dropped={self._send_dropped}"
            )

    def destroy_node(self):
        if not self._closed:
            self._closed = True
            if hasattr(self, "_timer"):
                self.destroy_timer(self._timer)
            for name, close_resource in (
                ("PICO subscriber", lambda: self._sub.close(linger=0)),
                ("smpl_ref publisher", lambda: self._pub.close(linger=0)),
                ("ZMQ context", self._zmq_context.term),
            ):
                try:
                    close_resource()
                except Exception as exc:
                    self.get_logger().warning(f"failed to close {name}: {exc}")
        return super().destroy_node()


def create_node(context: NodeBuildContext) -> SmplRefBridgeNode:
    return SmplRefBridgeNode(context)


__all__ = [
    "BRIDGE_DEFAULTS",
    "IncomingChunk",
    "PicoSourceReadinessGate",
    "SmplRefBridgeNode",
    "_source_chunk_fields",
    "create_node",
]
