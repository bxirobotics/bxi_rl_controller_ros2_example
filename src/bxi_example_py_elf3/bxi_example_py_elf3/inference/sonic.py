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
import time
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

import numpy as np
import onnxruntime as ort
import zmq

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - allows source-tree tooling without ROS setup
    get_package_share_directory = None


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

def _find_package_data_file(relative_path: str) -> str:
    if get_package_share_directory is not None:
        try:
            package_data = (
                Path(get_package_share_directory("bxi_example_py_elf3"))
                / "data"
                / relative_path
            )
            if package_data.exists():
                return str(package_data)
        except Exception:
            pass

    source_file = Path(__file__).resolve()
    for parent in source_file.parents:
        source_data = parent / "data" / relative_path
        if source_data.exists():
            return str(source_data)

    return str(Path.cwd() / "src" / "bxi_example_py_elf3" / "data" / relative_path)


DEFAULT_MODEL_ONNX = _find_package_data_file(
    "sonic_model/elf3_step28800_smpl/model_step_028800_smpl.onnx"
)
DEFAULT_STREAM_REFERENCE = _find_package_data_file(
    "sonic_reference/elf3_pico_stand_clean_001/stream_reference.npz"
)

JOINT_NAMES = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
    "r_hip_y_joint",
    "r_hip_x_joint",
    "r_hip_z_joint",
    "r_knee_y_joint",
    "r_ankle_y_joint",
    "r_ankle_x_joint",
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
)

DEFAULT_DOF_POS = np.array(
    [
        0.0,
        0.0,
        0.0,
        -0.3,
        0.0,
        0.0,
        0.6,
        -0.3,
        0.0,
        -0.3,
        0.0,
        0.0,
        0.6,
        -0.3,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

KPS = np.array(
    [
        108.448,
        162.672,
        176.421,
        176.421,
        176.421,
        54.224,
        176.421,
        33.493,
        21.771,
        176.421,
        176.421,
        54.224,
        176.421,
        33.493,
        21.771,
        54.224,
        54.224,
        16.747,
        54.224,
        16.747,
        16.747,
        16.747,
        54.224,
        54.224,
        16.747,
        54.224,
        16.747,
        16.747,
        16.747,
    ],
    dtype=np.float32,
)

KDS = np.array(
    [
        6.904,
        10.356,
        11.231,
        11.231,
        11.231,
        3.452,
        11.231,
        2.132,
        1.386,
        11.231,
        11.231,
        3.452,
        11.231,
        2.132,
        1.386,
        3.452,
        3.452,
        1.066,
        3.452,
        1.066,
        1.066,
        1.066,
        3.452,
        3.452,
        1.066,
        3.452,
        1.066,
        1.066,
        1.066,
    ],
    dtype=np.float32,
)

ACTION_SCALE = np.array(
    [
        0.230525229,
        0.153683486,
        0.141706486,
        0.141706486,
        0.141706486,
        0.230525229,
        0.212559729,
        0.373212313,
        0.229663314,
        0.141706486,
        0.141706486,
        0.230525229,
        0.212559729,
        0.373212313,
        0.229663314,
        0.230525229,
        0.230525229,
        0.37320117,
        0.230525229,
        0.37320117,
        0.37320117,
        0.37320117,
        0.230525229,
        0.230525229,
        0.37320117,
        0.230525229,
        0.37320117,
        0.37320117,
        0.37320117,
    ],
    dtype=np.float32,
)


@dataclass
class SmplReferenceFrame:
    term1_local: np.ndarray
    root_quat: np.ndarray
    wrist: np.ndarray
    anchor_quat: Optional[np.ndarray] = None
    frame_index: int = -1
    sequence: int = 0


def _as_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


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
    payload = msg[len(prefix):]
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
        arr = np.frombuffer(data[offset:offset + nbytes], dtype=dtype, count=count)
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


class SonicTeleopPolicy:
    """SONIC _smpl.onnx policy for BXI RobotControlState integration."""

    def __init__(
        self,
        model_onnx_path: Optional[str] = None,
        stream_reference_npz: Optional[str] = None,
        use_smpl_ref_zmq: Optional[bool] = None,
        smpl_ref_zmq_host: Optional[str] = None,
        smpl_ref_zmq_port: Optional[int] = None,
        smpl_ref_zmq_topic: Optional[str] = None,
        require_live_reference: Optional[bool] = None,
        yaw_bias_rad: Optional[float] = None,
    ):
        self.model_onnx_path = model_onnx_path or os.environ.get(
            "BXI_SONIC_MODEL_ONNX", DEFAULT_MODEL_ONNX
        )
        self.stream_reference_npz = stream_reference_npz or os.environ.get(
            "BXI_SONIC_STREAM_REFERENCE_NPZ", DEFAULT_STREAM_REFERENCE
        )
        self.use_smpl_ref_zmq = (
            _as_bool_env("BXI_SONIC_USE_SMPL_REF_ZMQ", True)
            if use_smpl_ref_zmq is None
            else bool(use_smpl_ref_zmq)
        )
        self.smpl_ref_zmq_host = smpl_ref_zmq_host or os.environ.get(
            "BXI_SONIC_SMPL_REF_ZMQ_HOST", "127.0.0.1"
        )
        self.smpl_ref_zmq_port = int(
            smpl_ref_zmq_port
            if smpl_ref_zmq_port is not None
            else os.environ.get("BXI_SONIC_SMPL_REF_ZMQ_PORT", "5557")
        )
        self.smpl_ref_zmq_topic = smpl_ref_zmq_topic or os.environ.get(
            "BXI_SONIC_SMPL_REF_ZMQ_TOPIC", "smpl_ref"
        )
        self.require_live_reference = (
            _as_bool_env("BXI_SONIC_REQUIRE_LIVE_REFERENCE", True)
            if require_live_reference is None
            else bool(require_live_reference)
        )
        self.yaw_bias_rad = float(
            yaw_bias_rad
            if yaw_bias_rad is not None
            else os.environ.get("BXI_SONIC_YAW_BIAS_RAD", "1.57079632679")
        )
        self.live_ref_timeout_s = float(os.environ.get("BXI_SONIC_LIVE_REF_TIMEOUT_S", "0.5"))
        self.idle_frame_start = int(
            os.environ.get("BXI_SONIC_IDLE_FRAME_START", str(DEFAULT_IDLE_FRAME_START))
        )
        self.source_blend_duration_s = max(
            0.0, float(os.environ.get("BXI_SONIC_SOURCE_BLEND_SECONDS", "0.4"))
        )

        self.default_dof_pos = DEFAULT_DOF_POS.copy()
        self.target_dof_pos = DEFAULT_DOF_POS.copy()
        self.kps = KPS.copy()
        self.kds = KDS.copy()
        self.action_scale = ACTION_SCALE.copy()
        self.joint_names = JOINT_NAMES
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

        self._load_stream_reference()
        self._init_onnx()
        self._init_zmq()

    def _init_onnx(self) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            self.model_onnx_path,
            providers=providers,
            sess_options=options,
        )
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        self.input_buffer = np.zeros((1, MODEL_INPUT_DIM), dtype=np.float32)
        if int(self.input_info.shape[-1]) != MODEL_INPUT_DIM:
            raise ValueError(
                f"SONIC ONNX input dim is {self.input_info.shape}; expected (1,{MODEL_INPUT_DIM})"
            )

    def _init_zmq(self) -> None:
        self.zmq_context = None
        self.zmq_socket = None
        if not self.use_smpl_ref_zmq:
            return
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, self.smpl_ref_zmq_topic)
        self.zmq_socket.connect(
            f"tcp://{self.smpl_ref_zmq_host}:{self.smpl_ref_zmq_port}"
        )
        self.zmq_poller = zmq.Poller()
        self.zmq_poller.register(self.zmq_socket, zmq.POLLIN)

    def close(self) -> None:
        """Release ZMQ resources owned by this policy instance."""
        socket = getattr(self, "zmq_socket", None)
        poller = getattr(self, "zmq_poller", None)
        context = getattr(self, "zmq_context", None)
        if socket is not None:
            if poller is not None:
                try:
                    poller.unregister(socket)
                except (KeyError, zmq.ZMQError):
                    pass
            socket.close(linger=0)
            self.zmq_socket = None
        if context is not None:
            context.term()
            self.zmq_context = None

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
            raise ValueError(f"term1_local shape {self.ref_term1.shape}, expected (T,72)")
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

    def reset(self) -> None:
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
        self.target_dof_pos = self.default_dof_pos.copy()
        self._drain_reference_socket()

    def reset_yaw_alignment(self) -> None:
        self.yaw_aligned = False
        self.yaw_offset = 0.0

    def _drain_reference_socket(self) -> None:
        """Discard packets queued before a SONIC reset/re-entry."""
        if self.zmq_socket is None:
            return
        while True:
            try:
                self.zmq_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def poll_reference(self) -> Optional[SmplReferenceFrame]:
        if self.zmq_socket is None:
            return None
        while True:
            events = dict(self.zmq_poller.poll(timeout=0))
            if self.zmq_socket not in events:
                break
            try:
                msg = self.zmq_socket.recv(flags=zmq.NOBLOCK)
                fields = _decode_packed_message(msg, self.smpl_ref_zmq_topic)
                if not fields:
                    continue
                frame = self._frame_from_fields(fields)
            except (
                IndexError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                zmq.Again,
            ):
                continue
            self.latest_live_ref = frame
            self.latest_live_ref_time = time.monotonic()
        return self.latest_live_ref

    def _frame_from_fields(self, fields: dict[str, np.ndarray]) -> SmplReferenceFrame:
        source_ready = fields.get("source_ready")
        if source_ready is None or not bool(np.asarray(source_ready).reshape(-1)[-1]):
            raise ValueError("smpl_ref source is not ready")
        stream_mode = fields.get("source_stream_mode")
        if stream_mode is not None and int(np.asarray(stream_mode).reshape(-1)[-1]) != 1:
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
            anchor_quat=_as_window(anchor, 4, "anchor_quat") if anchor is not None else None,
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
            anchor_quat=np.ascontiguousarray(anchor, dtype=np.float32) if anchor is not None else None,
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

    def _update_history(self, q: np.ndarray, dq: np.ndarray, quat_wxyz: np.ndarray, omega: np.ndarray) -> np.ndarray:
        anchor = _waist_z_quat_from_torso_wxyz(quat_wxyz, q[0], q[1], q[2])
        gravity = _projected_gravity_from_quat_wxyz(anchor)
        self.base_ang_vel_history[:-1] = self.base_ang_vel_history[1:]
        self.joint_pos_history[:-1] = self.joint_pos_history[1:]
        self.joint_vel_history[:-1] = self.joint_vel_history[1:]
        self.action_history[:-1] = self.action_history[1:]
        self.gravity_history[:-1] = self.gravity_history[1:]
        self.base_ang_vel_history[-1] = np.asarray(omega, dtype=np.float32).reshape(3)
        self.joint_pos_history[-1] = np.asarray(q, dtype=np.float32).reshape(NUM_JOINTS) - self.default_dof_pos
        self.joint_vel_history[-1] = np.asarray(dq, dtype=np.float32).reshape(NUM_JOINTS)
        self.action_history[-1] = self.last_action
        self.gravity_history[-1] = gravity
        return anchor

    def _capture_yaw_if_needed(self, frame: SmplReferenceFrame, anchor_quat_wxyz: np.ndarray) -> None:
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
        out[SMPL_JOINTS_START:SMPL_JOINTS_START + 720] = frame.term1_local.reshape(-1)
        out[SMPL_WRIST_START:SMPL_WRIST_START + 60] = frame.wrist.reshape(-1)
        conj_anchor = _quat_conjugate_wxyz(anchor_quat_wxyz)
        for k in range(WINDOW):
            root_quat = _normalize_quat_wxyz(frame.root_quat[k])
            rel = _quat_mul_wxyz(conj_anchor, root_quat)
            out[SMPL_ROOT_ORI_START + k * 6:SMPL_ROOT_ORI_START + (k + 1) * 6] = _sixd_from_quat_wxyz(rel)

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
        anchor_aligned = _quat_mul_wxyz(_axis_angle_quat_wxyz("z", self.yaw_offset), anchor)

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
        raw_action = self.session.run(
            [self.output_info.name], {self.input_info.name: self.input_buffer}
        )[0].reshape(-1)
        action = np.clip(raw_action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
        candidate = self.default_dof_pos + action * self.action_scale
        self.target_dof_pos = self._blend_source_target(candidate, now_mono)
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
            print(f"[SONIC] reference status: {self.last_status}", flush=True)
            self._reported_status = self.last_status
        return self.target_dof_pos
