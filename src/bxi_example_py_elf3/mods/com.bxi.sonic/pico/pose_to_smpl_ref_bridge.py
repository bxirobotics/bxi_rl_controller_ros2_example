"""Bridge official PICO manager ``pose`` stream to ELF3 ``smpl_ref`` stream.

The official ``gear_sonic/scripts/pico_manager_thread_server.py --manager`` sends
packed ZMQ messages on topic ``pose``.  For the ELF3 native _smpl.onnx deploy we
publish the same long-lived reference contract used by ``smpl_ref_bridge.py``:

    term1_local : float32 [10,72]   SMPL joints local, flattened per frame
    root_quat   : float32 [10,4]    SMPL root quaternion, wxyz
    wrist       : float32 [10,6]    ELF3 native wrist x/y/z, left then right

The bridge is intentionally only an adapter: PICO/SMPL normalization stays in
the official PICO manager, while the downstream SONIC policy consumes the
normalized reference tensors.  Live PICO chunks are merged with the same
sliding-window semantics as the official C++ StreamedMotionMerger, so the
published 10-frame smpl_ref window is a true future window instead of a tiled
latest frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import time
from typing import Any

import numpy as np
import zmq

from rclpy.node import Node
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.mod_api import NodeBuildContext

from .zmq_messages import pack_pose_message
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
WINDOW = 10
HISTORY_FRAMES = 5
MAX_GAP_FRAMES = 200
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
    "history_frames": HISTORY_FRAMES,
    "max_gap_frames": MAX_GAP_FRAMES,
    "catch_up_enabled": True,
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


@dataclass
class IncomingChunk:
    frame_indices: np.ndarray
    term1_local: np.ndarray
    root_quat: np.ndarray
    wrist: np.ndarray


@dataclass
class MergeResult:
    did_catchup_reset: bool = False
    frame_offset_adjustment: int = 0
    frame_step: int = 1


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
                for name in ("smpl_joints", "body_quat_w", "joint_pos")
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

        progressing = frame_index is not None and (
            self.last_frame_index is None or frame_index > self.last_frame_index
        )
        if frame_index is not None and (
            self.last_frame_index is None or frame_index > self.last_frame_index
        ):
            self.last_frame_index = frame_index
        if source_valid and progressing:
            self.streak += 1
        else:
            self.streak = 0
            self.last_ready_mono = None
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
        for k in ("frame_index", "smpl_joints", "body_quat_w", "joint_pos")
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
    frame_indices = np.asarray(fields["frame_index"], dtype=np.int64).reshape(-1)

    n = term1.shape[0]
    if root_quat.shape[0] != n or wrist.shape[0] != n or frame_indices.shape[0] != n:
        raise ValueError(
            "PICO pose frame count mismatch: "
            f"frame_index={frame_indices.shape[0]} term1={term1.shape[0]} "
            f"root={root_quat.shape[0]} wrist={wrist.shape[0]}"
        )
    if n <= 0:
        raise ValueError("PICO pose message has zero frames")
    if n > 1 and np.any(np.diff(frame_indices) <= 0):
        raise ValueError(
            f"frame_index must be strictly increasing: {frame_indices.tolist()}"
        )

    return IncomingChunk(
        frame_indices=np.ascontiguousarray(frame_indices, dtype=np.int64),
        term1_local=term1,
        root_quat=root_quat,
        wrist=wrist,
    )


class StreamedSmplRefMerger:
    """Python port of C++ StreamedMotionMerger for live PICO SMPL refs."""

    def __init__(
        self,
        history_frames: int = HISTORY_FRAMES,
        max_gap_frames: int = MAX_GAP_FRAMES,
        catch_up_enabled: bool = True,
    ):
        self.history_frames = int(history_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.catch_up_enabled = bool(catch_up_enabled)
        self.reset()

    def reset(self) -> None:
        self.term1_local = np.zeros((0, 72), dtype=np.float32)
        self.root_quat = np.zeros((0, 4), dtype=np.float32)
        self.wrist = np.zeros((0, 6), dtype=np.float32)
        self.stream_window_start = 0
        self.current_frame = 0
        self.frame_step = 1
        self.total_merges = 0
        self.catchup_count = 0

    @property
    def timesteps(self) -> int:
        return int(self.term1_local.shape[0])

    def _calculate_frame_step(self, frame_indices: np.ndarray) -> int:
        if frame_indices.shape[0] < 2:
            return 1
        step = abs(int(frame_indices[1]) - int(frame_indices[0]))
        return step if step > 0 else 1

    def _calculate_sliding_window(
        self,
        incoming_frame_start: int,
        incoming_frame_end: int,
        frame_step: int,
    ) -> tuple[int, int, bool]:
        if self.timesteps <= 0:
            return incoming_frame_start, 0, True

        global_playback_frame = self.stream_window_start + frame_step * max(
            0, self.current_frame - self.history_frames
        )
        max_gap_frames = (
            self.max_gap_frames + self.history_frames
            if self.catch_up_enabled
            else sys.maxsize
        )
        stream_window_end = self.stream_window_start + frame_step * (self.timesteps - 1)

        if incoming_frame_start <= self.stream_window_start:
            return incoming_frame_start, 0, True
        if incoming_frame_end <= stream_window_end:
            return incoming_frame_start, 0, True

        desired_window_start = global_playback_frame
        tentative_window_start = min(desired_window_start, incoming_frame_start)
        delta_to_incoming = incoming_frame_start - tentative_window_start
        tentative_merge_dst = delta_to_incoming // frame_step if frame_step > 0 else 0
        large_gap_from_old = incoming_frame_start > stream_window_end + frame_step

        if tentative_merge_dst > max_gap_frames or large_gap_from_old:
            return incoming_frame_start, 0, True
        return tentative_window_start, tentative_merge_dst, False

    def merge(self, chunk: IncomingChunk) -> MergeResult:
        frame_step = self._calculate_frame_step(chunk.frame_indices)
        incoming_frame_start = int(chunk.frame_indices[0])
        incoming_frame_end = int(chunk.frame_indices[-1])

        new_window_start, merge_dst_frame, did_catchup = self._calculate_sliding_window(
            incoming_frame_start,
            incoming_frame_end,
            frame_step,
        )

        new_len = merge_dst_frame + int(chunk.frame_indices.shape[0])
        new_term1 = np.zeros((new_len, 72), dtype=np.float32)
        new_root = np.zeros((new_len, 4), dtype=np.float32)
        new_wrist = np.zeros((new_len, 6), dtype=np.float32)

        old_window_start = self.stream_window_start
        if merge_dst_frame > 0 and self.timesteps > 0:
            old_window_end = old_window_start + frame_step * self.timesteps
            need_start_global = new_window_start
            need_end_global = incoming_frame_start
            overlap_start_global = max(need_start_global, old_window_start)
            overlap_end_global = min(need_end_global, old_window_end)
            if overlap_start_global < overlap_end_global:
                start_offset_old = overlap_start_global - old_window_start
                start_offset_new = overlap_start_global - new_window_start
                overlap_span = overlap_end_global - overlap_start_global
                copy_src_idx = start_offset_old // frame_step if frame_step > 0 else 0
                copy_dst_idx = start_offset_new // frame_step if frame_step > 0 else 0
                copy_count = overlap_span // frame_step if frame_step > 0 else 0
                if copy_count > 0:
                    new_term1[
                        copy_dst_idx : copy_dst_idx + copy_count
                    ] = self.term1_local[copy_src_idx : copy_src_idx + copy_count]
                    new_root[copy_dst_idx : copy_dst_idx + copy_count] = self.root_quat[
                        copy_src_idx : copy_src_idx + copy_count
                    ]
                    new_wrist[copy_dst_idx : copy_dst_idx + copy_count] = self.wrist[
                        copy_src_idx : copy_src_idx + copy_count
                    ]

        n_in = int(chunk.frame_indices.shape[0])
        new_term1[merge_dst_frame : merge_dst_frame + n_in] = chunk.term1_local
        new_root[merge_dst_frame : merge_dst_frame + n_in] = chunk.root_quat
        new_wrist[merge_dst_frame : merge_dst_frame + n_in] = chunk.wrist

        window_shift_ticks = new_window_start - old_window_start
        window_shift = window_shift_ticks // frame_step if frame_step > 0 else 0

        self.term1_local = new_term1
        self.root_quat = new_root
        self.wrist = new_wrist
        self.stream_window_start = new_window_start
        self.frame_step = frame_step
        self.total_merges += 1

        if did_catchup:
            self.current_frame = 0
            self.catchup_count += 1
            frame_offset_adjustment = 0
        else:
            self.current_frame = max(0, self.current_frame - window_shift)
            frame_offset_adjustment = window_shift

        return MergeResult(
            did_catchup_reset=did_catchup,
            frame_offset_adjustment=frame_offset_adjustment,
            frame_step=frame_step,
        )

    def advance(self, playing: bool = True) -> None:
        if self.timesteps <= 0:
            return
        if playing and self.current_frame + 1 < self.timesteps:
            self.current_frame += 1
        self.current_frame = int(np.clip(self.current_frame, 0, self.timesteps - 1))

    def build_smpl_ref(self) -> dict[str, np.ndarray] | None:
        if self.timesteps <= 0:
            return None
        self.advance(playing=True)
        idx = np.minimum(
            self.current_frame + np.arange(WINDOW, dtype=np.int64),
            self.timesteps - 1,
        )
        current_global_frame = (
            self.stream_window_start + self.current_frame * self.frame_step
        )
        return {
            "term1_local": np.ascontiguousarray(
                self.term1_local[idx], dtype=np.float32
            ),
            "root_quat": np.ascontiguousarray(self.root_quat[idx], dtype=np.float32),
            "wrist": np.ascontiguousarray(self.wrist[idx], dtype=np.float32),
            "frame_index": np.asarray([current_global_frame], dtype=np.int64),
        }


def _build_live_smpl_ref_if_ready(
    source_gate: PicoSourceReadinessGate,
    merger: StreamedSmplRefMerger,
    now_mono: float,
    stale_seconds: float,
) -> dict[str, np.ndarray] | None:
    """Build one live output only while the calibrated POSE source is fresh."""
    if not source_gate.is_fresh(now_mono, stale_seconds):
        if merger.timesteps:
            merger.reset()
        return None

    smpl_ref = merger.build_smpl_ref()
    if smpl_ref is None:
        return None
    smpl_ref["source_ready"] = np.array([True], dtype=bool)
    smpl_ref["source_stream_mode"] = np.array([POSE_STREAM_MODE], dtype=np.int32)
    smpl_ref["source_calibration_ready"] = np.array([True], dtype=bool)
    return smpl_ref


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
    for name in ("history_frames", "max_gap_frames"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("rate_hz", "stale_warning_seconds"):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError(f"{name} must be greater than zero")
    if not isinstance(params["catch_up_enabled"], bool):
        raise ValueError("catch_up_enabled must be a boolean")
    return params


class SmplRefBridgeNode(Node):
    """Non-blocking ROS node owned by the framework's shared executor."""

    def __init__(self, context: NodeBuildContext):
        params = _validated_params(dict(context.params))
        super().__init__(context.node_name, namespace=context.namespace or None)

        self._pico_topic = str(params["pico_topic"])
        self._out_topic = str(params["out_topic"])
        self._stale_seconds = float(params["stale_warning_seconds"])
        self._merger = StreamedSmplRefMerger(
            history_frames=int(params["history_frames"]),
            max_gap_frames=int(params["max_gap_frames"]),
            catch_up_enabled=bool(params["catch_up_enabled"]),
        )
        self._source_gate = PicoSourceReadinessGate()
        self._button_publishers = {
            name: self.create_publisher(Float32, f"pico/{name}", 10)
            for name in PICO_BUTTON_FIELDS
        }
        self._invalid_button_fields: set[str] = set()
        self._pending_fields: dict[str, np.ndarray] | None = None
        self._last_received_mono: float | None = None
        self._received = 0
        self._skipped = 0
        self._stale_was_reported = False
        self._stream_state: str | None = None
        self._closed = False

        self._zmq_context = zmq.Context()
        self._sub = self._zmq_context.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.RCVHWM, 1)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, self._pico_topic)
        self._pico_endpoint = f"tcp://{params['pico_host']}:{params['pico_port']}"
        self._sub.connect(self._pico_endpoint)

        self._pub = self._zmq_context.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.setsockopt(zmq.SNDHWM, 2)
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
            f"rate={float(params['rate_hz']):g}Hz"
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
            received_mono = time.monotonic()
            self._pending_fields = (
                fields
                if self._source_gate.observe(
                    fields,
                    received_mono,
                    self._stale_seconds,
                )
                else None
            )
            self._last_received_mono = received_mono
            self._stale_was_reported = False
            self._received += 1

    def _tick(self) -> None:
        if self._closed:
            return
        self._drain_input()
        now = time.monotonic()
        if self._pending_fields is not None and self._source_gate.is_fresh(
            now, self._stale_seconds
        ):
            try:
                self._merger.merge(_parse_incoming_chunk(self._pending_fields))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                self._skipped += 1
                self._source_gate.reset()
                self.get_logger().warning(f"skipped invalid PICO pose: {exc}")
            self._pending_fields = None

        smpl_ref = _build_live_smpl_ref_if_ready(
            self._source_gate,
            self._merger,
            now,
            self._stale_seconds,
        )
        if smpl_ref is not None:
            try:
                self._pub.send(
                    pack_pose_message(smpl_ref, topic=self._out_topic, version=4),
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                self._skipped += 1

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
                "Live smpl_ref publication stopped; policy uses idle reference."
            )
            self._stale_was_reported = True

        current_state = (
            "streaming"
            if smpl_ref is not None
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
        elif current_state == "streaming" and smpl_ref is not None:
            self.get_logger().info(
                "PICO stream ready; "
                f"frame={int(smpl_ref['frame_index'][0])} "
                f"input_age_ms={input_age * 1000.0:.0f}"
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
    "StreamedSmplRefMerger",
    "create_node",
]
