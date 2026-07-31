"""SONIC teleoperation policy wrapper for the official ELF3 BXI controller.

This module is intentionally a policy/inference module only.  It does not
publish ActuatorCmds, call reset services, or own the robot state machine.  The
BXI demo remains the only motor-command publisher and calls this class from a
RobotControlState.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
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
    anchor_quat: Optional[np.ndarray] = None
    frame_index: int = -1
    sequence: int = 0


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
        self.latest_live_ref: Optional[SmplReferenceFrame] = None
        self.latest_live_ref_time = 0.0
        self.live_sequence = 0
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
        self._reference_messages: deque[bytes] = deque(maxlen=1)
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
            socket.setsockopt(zmq.RCVHWM, 1)
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
                latest = None
                while True:
                    try:
                        latest = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest is not None:
                    self._reference_messages.append(latest)
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
                f"reference only has {self.ref_term1.shape[0]} frames; expected at least {WINDOW}"
            )
        self.idle_frame_start = int(
            np.clip(self.idle_frame_start, 0, self.ref_term1.shape[0] - WINDOW)
        )

    def reset(self, frame: InferenceFrame | None = None) -> None:
        if frame is not None:
            self.bind_joints(frame)
        self.last_action.fill(0.0)
        self.base_ang_vel_history.fill(0.0)
        self.joint_pos_history.fill(0.0)
        self.joint_vel_history.fill(0.0)
        self.action_history.fill(0.0)
        self.gravity_history.fill(0.0)
        self.motion_cursor = self.idle_frame_start
        self.yaw_aligned = False
        self.yaw_offset = 0.0
        self.latest_live_ref = None
        self.latest_live_ref_time = 0.0
        self.live_sequence = 0
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
        self._reference_messages.clear()

    def poll_reference(self) -> Optional[SmplReferenceFrame]:
        if not self.use_smpl_ref_zmq:
            return None
        try:
            msg = self._reference_messages.pop()
            self._reference_messages.clear()
        except IndexError:
            return self.latest_live_ref
        try:
            fields = _decode_packed_message(msg, self.smpl_ref_zmq_topic)
            if not fields:
                return self.latest_live_ref
            frame = self._frame_from_fields(fields)
        except (
            IndexError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return self.latest_live_ref
        self.latest_live_ref = frame
        self.latest_live_ref_time = time.monotonic()
        return self.latest_live_ref

    def _frame_from_fields(self, fields: dict[str, np.ndarray]) -> SmplReferenceFrame:
        source_ready = fields.get("source_ready")
        if source_ready is None or not bool(np.asarray(source_ready).reshape(-1)[-1]):
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

        frame_index = fields.get("frame_index")
        if frame_index is None or frame_index.size == 0:
            index = -1
        else:
            index = int(np.asarray(frame_index).reshape(-1)[-1])
        anchor = fields.get("anchor_quat")
        frame = SmplReferenceFrame(
            term1_local=_as_window(fields["term1_local"], 72, "term1_local"),
            root_quat=_as_window(fields["root_quat"], 4, "root_quat"),
            wrist=_as_window(fields["wrist"], 6, "wrist"),
            anchor_quat=_as_window(anchor, 4, "anchor_quat")
            if anchor is not None
            else None,
            frame_index=index,
            sequence=self.live_sequence + 1,
        )
        arrays = [frame.term1_local, frame.root_quat, frame.wrist]
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
            anchor_quat=np.ascontiguousarray(anchor, dtype=np.float32)
            if anchor is not None
            else None,
            frame_index=int(idx[0]),
            sequence=0,
        )

    def _active_reference(self) -> tuple[SmplReferenceFrame, str, float]:
        live = self.poll_reference()
        now_mono = time.monotonic()
        if (
            live is not None
            and now_mono - self.latest_live_ref_time <= self.live_ref_timeout_s
        ):
            return live, "live", now_mono
        if live is not None:
            self.latest_live_ref = None
            self.latest_live_ref_time = 0.0
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
            self.last_status = "live_reference"
        elif self.source_blend_active and self.source_transition_from == "live":
            self.last_status = "live_stale_to_idle"
        else:
            self.last_status = "idle_reference"
        if self.last_status != self._reported_status:
            if self._logger is None:
                raise RuntimeError("SONIC policy logger is not bound")
            self._logger.info(f"reference status: {self.last_status}")
            self._reported_status = self.last_status
        return self.target_dof_pos

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


__all__ = ["SONIC_PARAMETERS", "SmplReferenceFrame", "SonicTeleopPolicy"]
