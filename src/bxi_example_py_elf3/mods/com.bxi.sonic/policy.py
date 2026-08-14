"""SONIC teleoperation policy wrapper for the official ELF3 BXI controller.

This module is intentionally a policy/inference module only.  It does not
publish ActuatorCmds, call reset services, or own the robot state machine.  The
BXI demo remains the only motor-command publisher and calls this class from a
RobotControlState.
"""

from __future__ import annotations

import json
import math
import os
import queue
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Mapping, Optional

import numpy as np
import zmq

from bxi_example_py_elf3.framework.inference import (
    InferenceFrame,
    InferenceRuntime,
    JointPolicy,
    ModelSpec,
    PolicyJointContract,
    PolicyOutput,
    default_runtime,
)
from bxi_example_py_elf3.framework.joints import JointParameterSet
from bxi_example_py_elf3.framework.mod_api import LoggerLike
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

from .pico.runtime_config import SMPL_REF_HOST, SMPL_REF_PORT, SMPL_REF_TOPIC
from .pico.streamed_smpl_ref import (
    IncomingChunk,
    StreamedSmplRefMerger,
)


HEADER_SIZE = 1280
WINDOW = 10
NUM_JOINTS = 29
SMPL_TOKENIZER_DIM = 840
PROPRIOCEPTION_DIM = 930
MODEL_INPUT_DIM = SMPL_TOKENIZER_DIM + PROPRIOCEPTION_DIM
ACTION_CLIP = 20.0
DEFAULT_IDLE_FRAME_START = 3509

SMPL_JOINTS_START = 0
SMPL_ROOT_ORI_START = 720
SMPL_WRIST_START = 780

DTYPE_MAP = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}

SONIC_PARAMETERS = JointParameterSet.from_rows(
    ELF3_POLICY_JOINTS,
    (
        ("waist_y_joint", 0.0, 108.448, 6.904, 0.230525229),
        ("waist_x_joint", 0.0, 162.672, 10.356, 0.153683486),
        ("waist_z_joint", 0.0, 176.421, 11.231, 0.141706486),
        ("l_hip_y_joint", -0.3, 176.421, 11.231, 0.141706486),
        ("l_hip_x_joint", 0.0, 176.421, 11.231, 0.141706486),
        ("l_hip_z_joint", 0.0, 54.224, 3.452, 0.230525229),
        ("l_knee_y_joint", 0.6, 176.421, 11.231, 0.212559729),
        ("l_ankle_y_joint", -0.3, 33.493, 2.132, 0.373212313),
        ("l_ankle_x_joint", 0.0, 21.771, 1.386, 0.229663314),
        ("r_hip_y_joint", -0.3, 176.421, 11.231, 0.141706486),
        ("r_hip_x_joint", 0.0, 176.421, 11.231, 0.141706486),
        ("r_hip_z_joint", 0.0, 54.224, 3.452, 0.230525229),
        ("r_knee_y_joint", 0.6, 176.421, 11.231, 0.212559729),
        ("r_ankle_y_joint", -0.3, 33.493, 2.132, 0.373212313),
        ("r_ankle_x_joint", 0.0, 21.771, 1.386, 0.229663314),
        ("l_shoulder_y_joint", 0.2, 54.224, 3.452, 0.230525229),
        ("l_shoulder_x_joint", 0.2, 54.224, 3.452, 0.230525229),
        ("l_shoulder_z_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("l_elbow_y_joint", 0.6, 54.224, 3.452, 0.230525229),
        ("l_wrist_x_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("l_wrist_y_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("l_wrist_z_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("r_shoulder_y_joint", 0.2, 54.224, 3.452, 0.230525229),
        ("r_shoulder_x_joint", -0.2, 54.224, 3.452, 0.230525229),
        ("r_shoulder_z_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("r_elbow_y_joint", 0.6, 54.224, 3.452, 0.230525229),
        ("r_wrist_x_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("r_wrist_y_joint", 0.0, 16.747, 1.066, 0.37320117),
        ("r_wrist_z_joint", 0.0, 16.747, 1.066, 0.37320117),
    ),
)


@dataclass
class SmplReferenceFrame:
    term1_local: np.ndarray
    root_quat: np.ndarray
    wrist: np.ndarray
    head_joint_pos: np.ndarray
    anchor_quat: Optional[np.ndarray] = None
    frame_index: int = -1
    sequence: int = 0
    stream_epoch: Optional[int] = None
    source_stale: bool = False
    source_age_ms: Optional[float] = None
    playback_hold: bool = False
    newest_frame_index: int = -1
    lead_frames: int = -1
    valid_horizon: int = 0
    clamp_slots: int = -1


@dataclass(frozen=True, slots=True)
class PolicyPlaybackTelemetry:
    """One successfully consumed live reference window."""

    frame_index: int
    newest_frame_index: int
    lead_frames: int
    playback_hold: bool
    catchup_count: int
    stream_epoch: Optional[int]
    successful_inference_tick: int


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _axis_angle_quat_wxyz(axis: str, angle: float) -> np.ndarray:
    half = 0.5 * float(angle)
    c = math.cos(half)
    s = math.sin(half)
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=np.float64)
    raise ValueError(f"unsupported axis {axis}")


def _waist_z_quat_from_torso_wxyz(
    torso_quat_wxyz: np.ndarray,
    waist_y: float,
    waist_x: float,
    waist_z: float,
) -> np.ndarray:
    q = _normalize_quat_wxyz(torso_quat_wxyz)
    q = _quat_mul_wxyz(q, _axis_angle_quat_wxyz("y", waist_y))
    q = _quat_mul_wxyz(q, _axis_angle_quat_wxyz("x", waist_x))
    q = _quat_mul_wxyz(q, _axis_angle_quat_wxyz("z", waist_z))
    return _normalize_quat_wxyz(q)


def _projected_gravity_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(q)
    v = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    qc = np.array([w, -x, -y, -z], dtype=np.float64)
    return np.array(
        [
            v[0] * (qc[0] * qc[0] + qc[1] * qc[1] - qc[2] * qc[2] - qc[3] * qc[3])
            + v[1] * 2.0 * (qc[1] * qc[2] - qc[0] * qc[3])
            + v[2] * 2.0 * (qc[1] * qc[3] + qc[0] * qc[2]),
            v[0] * 2.0 * (qc[1] * qc[2] + qc[0] * qc[3])
            + v[1] * (qc[0] * qc[0] - qc[1] * qc[1] + qc[2] * qc[2] - qc[3] * qc[3])
            + v[2] * 2.0 * (qc[2] * qc[3] - qc[0] * qc[1]),
            v[0] * 2.0 * (qc[1] * qc[3] - qc[0] * qc[2])
            + v[1] * 2.0 * (qc[2] * qc[3] + qc[0] * qc[1])
            + v[2] * (qc[0] * qc[0] - qc[1] * qc[1] - qc[2] * qc[2] + qc[3] * qc[3]),
        ],
        dtype=np.float32,
    )


def _sixd_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    r, i, j, k = _normalize_quat_wxyz(q)
    two_s = 2.0 / (r * r + i * i + j * j + k * k)
    return np.array(
        [
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
        ],
        dtype=np.float32,
    )


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = _normalize_quat_wxyz(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _decode_packed_message(msg: bytes, topic: str) -> Optional[dict[str, np.ndarray]]:
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


def _as_window(arr: np.ndarray, width: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, width)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != width:
        raise ValueError(f"{name} has shape {arr.shape}; expected (*,{width})")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if arr.shape[0] >= WINDOW:
        return np.ascontiguousarray(arr[:WINDOW], dtype=np.float32)
    return np.ascontiguousarray(
        np.concatenate([arr, np.repeat(arr[-1:], WINDOW - arr.shape[0], axis=0)]),
        dtype=np.float32,
    )


def _as_exact_live_window(
    arr: np.ndarray,
    width: int,
    name: str,
) -> np.ndarray:
    """Validate a live window without silently tiling its final frame."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != width:
            raise ValueError(
                f"{name} has shape {arr.shape}; expected ({WINDOW},{width})"
            )
        arr = arr.reshape(1, width)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape != (WINDOW, width):
        raise ValueError(
            f"{name} has shape {arr.shape}; expected ({WINDOW},{width})"
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


STRICT_LIVE_WINDOW_METADATA = frozenset(
    ("stream_epoch", "valid_horizon", "clamp_slots")
)


class SonicTeleopPolicy(JointPolicy):
    """SONIC SMPL teleoperation policy using the shared inference runtime."""

    joint_contract = PolicyJointContract(
        observation=ELF3_POLICY_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        model_onnx_path: str,
        stream_reference_npz: str,
        use_smpl_ref_zmq: bool = True,
        smpl_ref_zmq_host: str = SMPL_REF_HOST,
        smpl_ref_zmq_port: int = SMPL_REF_PORT,
        smpl_ref_zmq_topic: str = SMPL_REF_TOPIC,
        yaw_bias_rad: float = math.pi / 2.0,
        live_ref_timeout_s: float = 0.5,
        idle_frame_start: int = DEFAULT_IDLE_FRAME_START,
        source_blend_duration_s: float = 0.4,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ):
        super().__init__()
        self.model_onnx_path = str(model_onnx_path)
        self.stream_reference_npz = str(stream_reference_npz)
        self._runtime = runtime or default_runtime()
        self._policy_name = "sonic"
        self.use_smpl_ref_zmq = bool(use_smpl_ref_zmq)
        self.smpl_ref_zmq_host = str(smpl_ref_zmq_host)
        self.smpl_ref_zmq_port = int(smpl_ref_zmq_port)
        self.smpl_ref_zmq_topic = str(smpl_ref_zmq_topic)
        self.yaw_bias_rad = float(yaw_bias_rad)
        self.live_ref_timeout_s = float(live_ref_timeout_s)
        self.idle_frame_start = int(idle_frame_start)
        self.source_blend_duration_s = float(source_blend_duration_s)
        self._validate_runtime_config()

        self._parameters = SONIC_PARAMETERS
        self.default_dof_pos = self._parameters.default_position
        self.target_dof_pos = self._target_buffer.position
        np.copyto(self.target_dof_pos, self.default_dof_pos)
        self.kps = self._parameters.kp
        self.kds = self._parameters.kd
        self.action_scale = self._parameters.action_scale
        self.joint_names = self.joint_contract.action.names
        self.obs_history_len = WINDOW

        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.base_ang_vel_history = np.zeros((WINDOW, 3), dtype=np.float32)
        self.joint_pos_history = np.zeros((WINDOW, NUM_JOINTS), dtype=np.float32)
        self.joint_vel_history = np.zeros((WINDOW, NUM_JOINTS), dtype=np.float32)
        self.action_history = np.zeros((WINDOW, NUM_JOINTS), dtype=np.float32)
        self.gravity_history = np.zeros((WINDOW, 3), dtype=np.float32)

        self.motion_cursor = self.idle_frame_start
        self.yaw_aligned = False
        self.yaw_offset = 0.0
        self.stream_merger = StreamedSmplRefMerger()
        self.source_stream_epoch: Optional[int] = None
        self.last_source_newest_frame: Optional[int] = None
        self.last_source_rx_mono = 0.0
        self.source_chunk_messages = 0
        self.source_chunk_duplicates = 0
        self.source_chunk_restarts = 0
        self.source_queue_drops = 0
        self.source_pre_reset_drops = 0
        self._reference_generation = 0
        self._reference_reset_cutoff_mono = 0.0
        self.has_seen_live_reference = False
        self.live_reference_protocol = "none"
        self.active_reference_kind = "none"
        self.latest_live_ref: Optional[SmplReferenceFrame] = None
        self.head_joint_target = np.zeros(2, dtype=np.float32)
        self.latest_live_ref_time = 0.0
        self.live_sequence = 0
        self.stream_epoch: Optional[int] = None
        self.invalid_live_ref_messages = 0
        self.live_reference_stale = False
        self.successful_inference_tick = 0
        self.latest_playback_telemetry: Optional[
            PolicyPlaybackTelemetry
        ] = None
        self.telemetry_log_every = max(
            0,
            int(os.environ.get("BXI_SONIC_TELEMETRY_LOG_EVERY", "0")),
        )
        self.reference_source: Optional[str] = None
        self.source_blend_from = self.default_dof_pos.copy()
        self.source_blend_started_at = 0.0
        self.source_blend_active = False
        self.source_transition_from: Optional[str] = None
        self.policy_active = False
        self.last_status = "not_started"
        self._reported_status: Optional[str] = None
        self._logger: LoggerLike | None = None

        self._load_stream_reference()
        self._init_backend(backend)
        self._init_zmq()
        self.publish_output(self.target_dof_pos, self.kps, self.kds)

    def bind_logger(self, logger: LoggerLike) -> None:
        self._logger = logger

    def _validate_runtime_config(self) -> None:
        if not math.isfinite(self.yaw_bias_rad):
            raise ValueError("yaw_bias_rad must be finite")
        if not math.isfinite(self.live_ref_timeout_s) or self.live_ref_timeout_s <= 0.0:
            raise ValueError("live_ref_timeout_s must be positive and finite")
        if self.idle_frame_start < 0:
            raise ValueError("idle_frame_start must be non-negative")
        if (
            not math.isfinite(self.source_blend_duration_s)
            or self.source_blend_duration_s < 0.0
        ):
            raise ValueError("source_blend_duration_s must be non-negative and finite")

    def configure_runtime(
        self,
        *,
        yaw_bias_rad: float,
        live_ref_timeout_s: float,
        idle_frame_start: int,
        source_blend_duration_s: float,
    ) -> None:
        """Apply state-owned behavior parameters before entering that state."""
        self.yaw_bias_rad = float(yaw_bias_rad)
        self.live_ref_timeout_s = float(live_ref_timeout_s)
        self.idle_frame_start = int(idle_frame_start)
        self.source_blend_duration_s = float(source_blend_duration_s)
        self._validate_runtime_config()
        if hasattr(self, "ref_term1"):
            self.idle_frame_start = int(
                np.clip(self.idle_frame_start, 0, self.ref_term1.shape[0] - WINDOW)
            )

    def _init_backend(self, backend: str) -> None:
        spec = ModelSpec.portable_onnx(
            self.model_onnx_path,
            input_names=("obs_dict",),
            output_names=("action",),
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)
        self.input_buffer = np.zeros((1, MODEL_INPUT_DIM), dtype=np.float32)
        self._inputs = {"obs_dict": self.input_buffer}
        self._backend.warmup(self._inputs, self._runtime.options.warmup_runs)

    def _init_zmq(self) -> None:
        self._reference_messages: queue.Queue[
            tuple[bytes, float, int]
        ] = queue.Queue(maxsize=64)
        self._zmq_stop = Event()
        self._zmq_ready = Event()
        self._zmq_error: BaseException | None = None
        self._zmq_thread: Thread | None = None
        if not self.use_smpl_ref_zmq:
            return
        thread = Thread(
            target=self._run_reference_receiver,
            name="sonic-reference",
            daemon=False,
        )
        self._zmq_thread = thread
        thread.start()
        self._zmq_ready.wait()
        if self._zmq_error is not None:
            raise RuntimeError(
                f"cannot initialize SONIC reference receiver: {self._zmq_error}"
            ) from self._zmq_error

    def _run_reference_receiver(self) -> None:
        context = None
        socket = None
        poller = None
        try:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVHWM, 64)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt_string(zmq.SUBSCRIBE, self.smpl_ref_zmq_topic)
            socket.connect(
                f"tcp://{self.smpl_ref_zmq_host}:{self.smpl_ref_zmq_port}"
            )
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
        except BaseException as exc:
            self._zmq_error = exc
        finally:
            self._zmq_ready.set()
        if self._zmq_error is not None:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
            return

        try:
            while not self._zmq_stop.is_set():
                events = dict(poller.poll(timeout=50))
                if socket not in events:
                    continue
                while True:
                    # Capture the reset generation before recv.  If reset()
                    # races with this receive/queue handoff, poll_reference()
                    # rejects the packet instead of reactivating the previous
                    # live session after the Python queue was drained.
                    generation = self._reference_generation
                    received_mono = time.monotonic()
                    try:
                        message = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._queue_reference_message(
                        message,
                        received_mono,
                        generation,
                    )
        except zmq.ZMQError as exc:
            if not self._zmq_stop.is_set():
                self._zmq_error = exc
        finally:
            try:
                poller.unregister(socket)
            except (KeyError, zmq.ZMQError):
                pass
            socket.close(linger=0)
            context.term()

    def _queue_reference_message(
        self,
        message: bytes,
        received_mono: float,
        generation: int | None = None,
    ) -> None:
        """Keep source chunks ordered; discard only the oldest on overflow."""
        item = (
            message,
            float(received_mono),
            self._reference_generation if generation is None else int(generation),
        )
        try:
            self._reference_messages.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._reference_messages.get_nowait()
            self.source_queue_drops += 1
        except queue.Empty:
            pass
        try:
            self._reference_messages.put_nowait(item)
        except queue.Full:
            self.source_queue_drops += 1

    def close(self) -> None:
        """Release ZMQ resources owned by this policy instance."""
        stop = getattr(self, "_zmq_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_zmq_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise RuntimeError("SONIC reference receiver did not stop")
            self._zmq_thread = None
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.close()
            self._backend = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_stream_reference(self) -> None:
        d = np.load(self.stream_reference_npz)
        self.ref_term1 = np.asarray(d["term1_local"], dtype=np.float32)
        self.ref_root_quat = np.asarray(d["root_quat"], dtype=np.float32)
        self.ref_wrist = np.asarray(d["wrist"], dtype=np.float32)
        self.ref_anchor_quat = (
            np.asarray(d["anchor_quat"], dtype=np.float32)
            if "anchor_quat" in d.files
            else None
        )
        if self.ref_term1.ndim != 2 or self.ref_term1.shape[1] != 72:
            raise ValueError(
                f"term1_local shape {self.ref_term1.shape}, expected (T,72)"
            )
        if self.ref_root_quat.shape != (self.ref_term1.shape[0], 4):
            raise ValueError("root_quat shape does not match term1_local")
        if self.ref_wrist.shape != (self.ref_term1.shape[0], 6):
            raise ValueError("wrist shape does not match term1_local")
        if self.ref_term1.shape[0] < WINDOW:
            raise ValueError(
                f"reference only has {self.ref_term1.shape[0]} frames; "
                f"expected at least {WINDOW}"
            )
        self.idle_frame_start = int(
            np.clip(self.idle_frame_start, 0, self.ref_term1.shape[0] - WINDOW)
        )

    def reset(self, frame: InferenceFrame | None = None) -> None:
        # Invalidate any receive operation that began before this reset.  The
        # receiver is the sole ZMQ socket owner, so reset only changes this
        # lock-free generation and drains the thread-safe handoff queue.
        self._reference_generation += 1
        self._reference_reset_cutoff_mono = time.monotonic()
        if frame is not None:
            self.bind_joints(frame)
        self.last_action.fill(0.0)
        self.base_ang_vel_history.fill(0.0)
        self.joint_pos_history.fill(0.0)
        self.joint_vel_history.fill(0.0)
        self.action_history.fill(0.0)
        self.gravity_history.fill(0.0)
        self.motion_cursor = self.idle_frame_start
        self.reset_yaw_alignment()
        self.stream_merger.reset()
        self.source_stream_epoch = None
        self.last_source_newest_frame = None
        self.last_source_rx_mono = 0.0
        self.has_seen_live_reference = False
        self.live_reference_protocol = "none"
        self.active_reference_kind = "none"
        self.latest_live_ref = None
        self.head_joint_target.fill(0.0)
        self.latest_live_ref_time = 0.0
        self.live_sequence = 0
        self.stream_epoch = None
        self.live_reference_stale = False
        self.latest_playback_telemetry = None
        self.reference_source = None
        self.source_blend_from = self.default_dof_pos.copy()
        self.source_blend_started_at = 0.0
        self.source_blend_active = False
        self.source_transition_from = None
        self.policy_active = False
        self.last_status = "reset"
        self._reported_status = None
        np.copyto(self.target_dof_pos, self.default_dof_pos)
        self.publish_output(self.target_dof_pos, self.kps, self.kds)
        self._drain_reference_socket()

    def has_fresh_live_reference(self, timeout_s: float | None = None) -> bool:
        self.poll_reference()
        timeout = self.live_ref_timeout_s if timeout_s is None else float(timeout_s)
        return bool(
            self.latest_live_ref is not None
            and time.monotonic() - self.latest_live_ref_time <= timeout
        )

    def reset_yaw_alignment(self) -> None:
        self.yaw_aligned = False
        self.yaw_offset = 0.0

    def _drain_reference_socket(self) -> None:
        """Discard packets queued before a SONIC reset/re-entry."""
        while True:
            try:
                self._reference_messages.get_nowait()
            except queue.Empty:
                return

    def poll_reference(self) -> Optional[SmplReferenceFrame]:
        while True:
            try:
                msg, received_mono, generation = (
                    self._reference_messages.get_nowait()
                )
            except queue.Empty:
                break
            if (
                generation != self._reference_generation
                or received_mono < self._reference_reset_cutoff_mono
            ):
                self.source_pre_reset_drops += 1
                continue
            try:
                fields = _decode_packed_message(
                    msg,
                    self.smpl_ref_zmq_topic,
                )
                if not fields:
                    raise ValueError("invalid smpl_ref message")
                source_received_ns = int(
                    self._field_scalar(
                        fields,
                        "source_received_monotonic_ns",
                        0,
                    )
                )
                if (
                    source_received_ns > 0
                    and source_received_ns
                    < int(self._reference_reset_cutoff_mono * 1.0e9)
                ):
                    self.source_pre_reset_drops += 1
                    continue
                if bool(self._field_scalar(fields, "source_chunk", False)):
                    self._merge_source_fields(fields, received_mono)
                    continue
                frame = self._frame_from_fields(fields)
            except (
                IndexError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
            ):
                self.invalid_live_ref_messages += 1
                continue

            if not self.has_seen_live_reference:
                self.reset_yaw_alignment()
            self.has_seen_live_reference = True
            self.stream_merger.reset()
            self.source_stream_epoch = None
            self.last_source_newest_frame = None
            self.live_reference_protocol = "legacy_window"
            self.latest_live_ref = frame
            self.latest_live_ref_time = received_mono

        if self.stream_merger.timesteps >= WINDOW:
            age_s = max(0.0, time.monotonic() - self.last_source_rx_mono)
            fields = self.stream_merger.build_smpl_ref(
                source_age_ms=age_s * 1000.0,
                source_stale=age_s > self.live_ref_timeout_s,
            )
            if fields is not None:
                if not self.has_seen_live_reference:
                    self.reset_yaw_alignment()
                self.has_seen_live_reference = True
                self.live_reference_protocol = "source_chunk"
                self.latest_live_ref = self._frame_from_fields(fields)
                self.latest_live_ref_time = self.last_source_rx_mono
        return self.latest_live_ref

    @staticmethod
    def _field_scalar(
        fields: dict[str, np.ndarray],
        name: str,
        default: Any = None,
    ) -> Any:
        value = fields.get(name)
        if value is None or np.asarray(value).size == 0:
            return default
        return np.asarray(value).reshape(-1)[-1]

    @staticmethod
    def _source_chunk_from_fields(
        fields: dict[str, np.ndarray],
    ) -> IncomingChunk:
        frame_indices = np.asarray(
            fields["frame_index"], dtype=np.int64
        ).reshape(-1)
        n = int(frame_indices.size)
        if n < WINDOW:
            raise ValueError(
                f"source chunk has {n} frames; need at least {WINDOW}"
            )
        if np.any(np.diff(frame_indices) != 1):
            raise ValueError(
                "source chunk frame_index must be consecutive: "
                f"{frame_indices.tolist()}"
            )

        def matrix(name: str, width: int) -> np.ndarray:
            arr = np.asarray(fields[name], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, width)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.shape != (n, width):
                raise ValueError(
                    f"source chunk {name} has shape {arr.shape}; "
                    f"expected ({n},{width})"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"source chunk {name} contains non-finite values")
            return np.ascontiguousarray(arr, dtype=np.float32)

        return IncomingChunk(
            frame_indices=np.ascontiguousarray(
                frame_indices, dtype=np.int64
            ),
            term1_local=matrix("term1_local", 72),
            root_quat=matrix("root_quat", 4),
            wrist=matrix("wrist", 6),
            head_joint_pos=matrix("head_joint_pos", 2),
        )

    def _merge_source_fields(
        self,
        fields: dict[str, np.ndarray],
        received_mono: float,
    ) -> bool:
        chunk = self._source_chunk_from_fields(fields)
        source_epoch = int(
            self._field_scalar(fields, "source_stream_epoch", 0)
        )
        if source_epoch <= 0:
            raise ValueError("source chunk missing positive source_stream_epoch")

        if self.source_stream_epoch != source_epoch:
            self.stream_merger.reset()
            self.source_stream_epoch = source_epoch
            self.last_source_newest_frame = None
            self.source_chunk_restarts += int(self.has_seen_live_reference)

        newest = int(chunk.frame_indices[-1])
        if self.last_source_newest_frame is not None:
            if newest == self.last_source_newest_frame:
                self.source_chunk_duplicates += 1
                return False
            if newest < self.last_source_newest_frame:
                self.stream_merger.reset()
                self.last_source_newest_frame = None
                self.source_chunk_restarts += 1

        previous_merger_epoch = self.stream_merger.stream_epoch
        self.stream_merger.merge(chunk)
        if self.stream_merger.stream_epoch != previous_merger_epoch:
            self.reset_yaw_alignment()
        self.last_source_newest_frame = newest
        self.last_source_rx_mono = float(received_mono)
        self.source_chunk_messages += 1
        return True

    def _frame_from_fields(self, fields: dict[str, np.ndarray]) -> SmplReferenceFrame:
        source_ready = fields.get("source_ready")
        if source_ready is not None and not bool(
            np.asarray(source_ready).reshape(-1)[-1]
        ):
            raise ValueError("smpl_ref source is not ready")
        stream_mode = fields.get("source_stream_mode")
        if (
            stream_mode is not None
            and int(np.asarray(stream_mode).reshape(-1)[-1]) != 1
        ):
            raise ValueError("smpl_ref source is not in POSE mode")
        calibration_ready = fields.get("source_calibration_ready")
        if calibration_ready is not None and not bool(
            np.asarray(calibration_ready).reshape(-1)[-1]
        ):
            raise ValueError("smpl_ref source is not calibrated")

        def scalar(name: str, default: Any) -> Any:
            return self._field_scalar(fields, name, default)

        valid_horizon = int(scalar("valid_horizon", 0))
        clamp_slots = int(scalar("clamp_slots", -1))
        strict_live_window = any(
            name in fields for name in STRICT_LIVE_WINDOW_METADATA
        )
        if strict_live_window:
            if valid_horizon != WINDOW:
                raise ValueError(
                    f"live reference must declare valid_horizon={WINDOW}; "
                    f"got {valid_horizon}"
                )
            if clamp_slots != 0:
                raise ValueError(
                    f"live reference must declare clamp_slots=0; got {clamp_slots}"
                )
            as_live_window = _as_exact_live_window
        else:
            as_live_window = _as_window
        anchor = fields.get("anchor_quat")
        frame = SmplReferenceFrame(
            term1_local=as_live_window(
                fields["term1_local"], 72, "term1_local"
            ),
            root_quat=as_live_window(fields["root_quat"], 4, "root_quat"),
            wrist=as_live_window(fields["wrist"], 6, "wrist"),
            head_joint_pos=as_live_window(
                fields["head_joint_pos"], 2, "head_joint_pos"
            ),
            anchor_quat=as_live_window(anchor, 4, "anchor_quat")
            if anchor is not None
            else None,
            frame_index=int(scalar("frame_index", -1)),
            sequence=self.live_sequence + 1,
            stream_epoch=(
                int(scalar("stream_epoch", 0))
                if fields.get("stream_epoch") is not None
                else None
            ),
            source_stale=bool(scalar("source_stale", False)),
            source_age_ms=(
                float(scalar("source_age_ms", 0.0))
                if fields.get("source_age_ms") is not None
                else None
            ),
            playback_hold=bool(scalar("playback_hold", False)),
            newest_frame_index=int(scalar("newest_frame_index", -1)),
            lead_frames=int(scalar("lead_frames", -1)),
            valid_horizon=valid_horizon,
            clamp_slots=clamp_slots,
        )
        arrays = [
            frame.term1_local,
            frame.root_quat,
            frame.wrist,
            frame.head_joint_pos,
        ]
        if frame.anchor_quat is not None:
            arrays.append(frame.anchor_quat)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("smpl_ref contains non-finite values")
        if np.any(np.linalg.norm(frame.root_quat, axis=1) <= 1.0e-6):
            raise ValueError("smpl_ref contains an invalid root quaternion")
        if frame.anchor_quat is not None and np.any(
            np.linalg.norm(frame.anchor_quat, axis=1) <= 1.0e-6
        ):
            raise ValueError("smpl_ref contains an invalid anchor quaternion")
        self.live_sequence += 1
        return frame

    def _offline_frame(self) -> SmplReferenceFrame:
        t = self.ref_term1.shape[0]
        start = int(np.clip(self.idle_frame_start, 0, t - WINDOW))
        idx = np.arange(start, start + WINDOW)
        anchor = self.ref_anchor_quat[idx] if self.ref_anchor_quat is not None else None
        return SmplReferenceFrame(
            term1_local=np.ascontiguousarray(self.ref_term1[idx], dtype=np.float32),
            root_quat=np.ascontiguousarray(self.ref_root_quat[idx], dtype=np.float32),
            wrist=np.ascontiguousarray(self.ref_wrist[idx], dtype=np.float32),
            head_joint_pos=np.zeros((WINDOW, 2), dtype=np.float32),
            anchor_quat=np.ascontiguousarray(anchor, dtype=np.float32)
            if anchor is not None
            else None,
            frame_index=int(idx[0]),
            sequence=0,
        )

    def _active_reference(self) -> tuple[SmplReferenceFrame, str, float]:
        live = self.poll_reference()
        now_mono = time.monotonic()
        if live is not None:
            if live.stream_epoch is not None and live.stream_epoch != self.stream_epoch:
                self.stream_epoch = live.stream_epoch
                self.reset_yaw_alignment()

            local_age_s = max(0.0, now_mono - self.latest_live_ref_time)
            source_age_stale = (
                live.source_age_ms is not None
                and live.source_age_ms > self.live_ref_timeout_s * 1000.0
            )
            self.live_reference_stale = bool(
                live.source_stale
                or source_age_stale
                or local_age_s > self.live_ref_timeout_s
            )
            self.active_reference_kind = self.live_reference_protocol
            return live, "live", now_mono

        self.live_reference_stale = False
        self.active_reference_kind = "idle"
        return self._offline_frame(), "idle", now_mono

    def _begin_source_transition(self, source: str, now_mono: float) -> None:
        if source == self.reference_source:
            return
        self.source_transition_from = self.reference_source
        self.reference_source = source
        self.reset_yaw_alignment()
        self.source_blend_from = self.target_dof_pos.copy()
        self.source_blend_started_at = now_mono
        self.source_blend_active = self.source_blend_duration_s > 0.0

    def _blend_source_target(
        self, candidate: np.ndarray, now_mono: float
    ) -> np.ndarray:
        if not self.source_blend_active:
            return candidate
        progress = np.clip(
            (now_mono - self.source_blend_started_at) / self.source_blend_duration_s,
            0.0,
            1.0,
        )
        alpha = float(progress * progress * (3.0 - 2.0 * progress))
        blended = (1.0 - alpha) * self.source_blend_from + alpha * candidate
        if progress >= 1.0:
            self.source_blend_active = False
            self.source_transition_from = None
        return np.asarray(blended, dtype=np.float32)

    def _update_history(
        self, q: np.ndarray, dq: np.ndarray, quat_wxyz: np.ndarray, omega: np.ndarray
    ) -> np.ndarray:
        anchor = _waist_z_quat_from_torso_wxyz(quat_wxyz, q[0], q[1], q[2])
        gravity = _projected_gravity_from_quat_wxyz(anchor)
        self.base_ang_vel_history[:-1] = self.base_ang_vel_history[1:]
        self.joint_pos_history[:-1] = self.joint_pos_history[1:]
        self.joint_vel_history[:-1] = self.joint_vel_history[1:]
        self.action_history[:-1] = self.action_history[1:]
        self.gravity_history[:-1] = self.gravity_history[1:]
        self.base_ang_vel_history[-1] = np.asarray(omega, dtype=np.float32).reshape(3)
        self.joint_pos_history[-1] = (
            np.asarray(q, dtype=np.float32).reshape(NUM_JOINTS) - self.default_dof_pos
        )
        self.joint_vel_history[-1] = np.asarray(dq, dtype=np.float32).reshape(
            NUM_JOINTS
        )
        self.action_history[-1] = self.last_action
        self.gravity_history[-1] = gravity
        return anchor

    def _capture_yaw_if_needed(
        self, frame: SmplReferenceFrame, anchor_quat_wxyz: np.ndarray
    ) -> None:
        if self.yaw_aligned:
            return
        if frame.anchor_quat is not None:
            reference_yaw = _yaw_from_quat_wxyz(frame.anchor_quat[0])
            bias = 0.0
        else:
            reference_yaw = _yaw_from_quat_wxyz(frame.root_quat[0])
            bias = self.yaw_bias_rad
        self.yaw_offset = reference_yaw - _yaw_from_quat_wxyz(anchor_quat_wxyz) + bias
        self.yaw_aligned = True

    def _write_smpl_tokenizer(
        self,
        frame: SmplReferenceFrame,
        anchor_quat_wxyz: np.ndarray,
        out: np.ndarray,
    ) -> None:
        out[SMPL_JOINTS_START : SMPL_JOINTS_START + 720] = frame.term1_local.reshape(-1)
        out[SMPL_WRIST_START : SMPL_WRIST_START + 60] = frame.wrist.reshape(-1)
        conj_anchor = _quat_conjugate_wxyz(anchor_quat_wxyz)
        for k in range(WINDOW):
            root_quat = _normalize_quat_wxyz(frame.root_quat[k])
            rel = _quat_mul_wxyz(conj_anchor, root_quat)
            out[
                SMPL_ROOT_ORI_START + k * 6 : SMPL_ROOT_ORI_START + (k + 1) * 6
            ] = _sixd_from_quat_wxyz(rel)

    def _build_model_input(
        self,
        frame: SmplReferenceFrame,
        q: np.ndarray,
        dq: np.ndarray,
        quat_wxyz: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        anchor = self._update_history(q, dq, quat_wxyz, omega)
        self._capture_yaw_if_needed(frame, anchor)
        anchor_aligned = _quat_mul_wxyz(
            _axis_angle_quat_wxyz("z", self.yaw_offset), anchor
        )

        model_input = np.zeros(MODEL_INPUT_DIM, dtype=np.float32)
        self._write_smpl_tokenizer(frame, anchor_aligned, model_input)
        proprio = np.concatenate(
            [
                self.base_ang_vel_history.reshape(-1),
                self.joint_pos_history.reshape(-1),
                self.joint_vel_history.reshape(-1),
                self.action_history.reshape(-1),
                self.gravity_history.reshape(-1),
            ]
        ).astype(np.float32)
        model_input[SMPL_TOKENIZER_DIM:] = proprio
        return model_input.reshape(1, -1)

    def inference_step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        quat_wxyz: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32).reshape(NUM_JOINTS)
        dq = np.asarray(dq, dtype=np.float32).reshape(NUM_JOINTS)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        omega = np.asarray(omega, dtype=np.float32).reshape(3)

        frame, source, now_mono = self._active_reference()
        if source == "live":
            np.copyto(self.head_joint_target, frame.head_joint_pos[-1])
        else:
            self.head_joint_target.fill(0.0)
        self._begin_source_transition(source, now_mono)

        model_input = self._build_model_input(frame, q, dq, quat_wxyz, omega)
        np.copyto(self.input_buffer, model_input)
        raw_action = np.asarray(self._backend.run(self._inputs)["action"]).reshape(-1)
        if raw_action.size != NUM_JOINTS:
            raise ValueError(
                f"SONIC output has {raw_action.size} values; expected {NUM_JOINTS}"
            )
        action = np.clip(raw_action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
        candidate = self.default_dof_pos + action * self.action_scale
        np.copyto(
            self.target_dof_pos,
            self._blend_source_target(candidate, now_mono),
        )
        self.last_action = (
            (self.target_dof_pos - self.default_dof_pos) / self.action_scale
        ).astype(np.float32)
        self.policy_active = True
        if source == "live":
            self.last_status = (
                "stale_hold"
                if self.live_reference_stale
                else "live_reference"
            )
        else:
            self.last_status = "idle_reference"
        if self.last_status != self._reported_status:
            if self._logger is None:
                raise RuntimeError("SONIC policy logger is not bound")
            self._logger.info(f"reference status: {self.last_status}")
            self._reported_status = self.last_status

        # Official order: gather -> successful inference/action -> advance.
        # Any backend/output exception above therefore leaves the cursor intact.
        if self.active_reference_kind == "source_chunk":
            advanced = self.stream_merger.advance_after_successful_tick()
            self._record_playback_telemetry(frame, advanced=advanced)
        return self.target_dof_pos

    def _record_playback_telemetry(
        self,
        frame: SmplReferenceFrame,
        *,
        advanced: bool,
    ) -> None:
        self.successful_inference_tick += 1
        telemetry = PolicyPlaybackTelemetry(
            frame_index=int(frame.frame_index),
            newest_frame_index=int(frame.newest_frame_index),
            lead_frames=int(frame.lead_frames),
            playback_hold=not bool(advanced),
            catchup_count=int(self.stream_merger.catchup_count),
            stream_epoch=(
                int(frame.stream_epoch)
                if frame.stream_epoch is not None
                else None
            ),
            successful_inference_tick=self.successful_inference_tick,
        )
        self.latest_playback_telemetry = telemetry
        if (
            self.telemetry_log_every > 0
            and self.successful_inference_tick % self.telemetry_log_every == 0
        ):
            payload = {
                "frame_index": telemetry.frame_index,
                "newest_frame_index": telemetry.newest_frame_index,
                "lead_frames": telemetry.lead_frames,
                "playback_hold": telemetry.playback_hold,
                "catchup_count": telemetry.catchup_count,
                "stream_epoch": telemetry.stream_epoch,
                "successful_inference_tick": telemetry.successful_inference_tick,
            }
            print(
                "[sonic-playback-telemetry] "
                + json.dumps(payload, sort_keys=True, separators=(",", ":")),
                flush=True,
            )

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        if not advance:
            return self.output
        joints = self.bind_joints(frame)
        self.inference_step(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
        )
        return self.output

    def decode_into(self, outputs: Mapping[str, np.ndarray]) -> None:
        """Satisfy JointPolicy's decoder contract; custom step decodes inline."""


__all__ = [
    "PolicyPlaybackTelemetry",
    "SONIC_PARAMETERS",
    "SmplReferenceFrame",
    "SonicTeleopPolicy",
]
