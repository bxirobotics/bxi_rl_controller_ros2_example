"""Motion-tracking policies built on shared backend and history primitives."""

from __future__ import annotations

import time

import numpy as np

from bxi_example_py_elf3.framework.mod_api.geometry import (
    get_gravity_orientation,
    matrix_to_quaternion_simple,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    yaw_quat,
)
from .joints import ELF3_ISAAC_JOINTS, ELF3_POLICY_JOINTS

from bxi_example_py_elf3.framework.inference.api import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.inference.contract import PolicyJointContract
from bxi_example_py_elf3.framework.inference.history import HistoryBuffer
from bxi_example_py_elf3.framework.inference.model import ModelSpec
from bxi_example_py_elf3.framework.inference.policy import JointPolicy
from bxi_example_py_elf3.framework.inference.runtime import InferenceRuntime, default_runtime


MUJOCO_TO_ISAAC = np.asarray(
    [
        15,
        22,
        0,
        16,
        23,
        1,
        17,
        24,
        2,
        18,
        25,
        3,
        9,
        19,
        26,
        4,
        10,
        20,
        27,
        5,
        11,
        21,
        28,
        6,
        12,
        7,
        13,
        8,
        14,
    ],
    dtype=np.int64,
)
ISAAC_TO_MUJOCO = np.asarray(
    [
        2,
        5,
        8,
        11,
        15,
        19,
        23,
        25,
        27,
        12,
        16,
        20,
        24,
        26,
        28,
        0,
        3,
        6,
        9,
        13,
        17,
        21,
        1,
        4,
        7,
        10,
        14,
        18,
        22,
    ],
    dtype=np.int64,
)
ELF3_KPS = np.asarray(
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
ELF3_KDS = np.asarray(
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


def _csv_floats(value: str) -> np.ndarray:
    return np.fromstring(value, sep=",", dtype=np.float32)


class _MotionGeometry:
    init_to_world: np.ndarray

    def compute_init_to_world(self, robot_quat, motion_quat):
        yaw_motion_matrix = quaternion_to_rotation_matrix(
            yaw_quat(motion_quat)
        ).reshape(3, 3)
        yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_quat(robot_quat)).reshape(
            3, 3
        )
        self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

    def compute_relmatrix(self, robot_quat, motion_quat):
        rel_quat = quaternion_multiply(
            matrix_to_quaternion_simple(self.init_to_world), motion_quat
        )
        rel_quat = quaternion_multiply(quaternion_conjugate(robot_quat), rel_quat)
        rel_quat /= max(float(np.linalg.norm(rel_quat)), 1e-8)
        return quaternion_to_rotation_matrix(rel_quat)[:, :2].reshape(-1)


class _LegacyMotionPolicy(_MotionGeometry, JointPolicy):
    joint_contract = PolicyJointContract(
        observation=ELF3_POLICY_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        motion_npz_path: str,
        model_onnx_path: str | ModelSpec,
        start_frame: int = 0,
        fixed_pos: bool = False,
        *,
        include_gravity: bool,
        isaac_order: bool,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
        JointPolicy.__init__(self)
        self.motion_npz_path = motion_npz_path
        self.fixed_pos = fixed_pos
        self._alignment_samples = 0
        self._include_gravity = include_gravity
        self._isaac_order = isaac_order
        self.num_obs = 157 if include_gravity else 154
        self._runtime = runtime or default_runtime()
        self._policy_name = "motion"

        self.motion = np.load(motion_npz_path)
        self._motion_position = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]

        spec = (
            model_onnx_path
            if isinstance(model_onnx_path, ModelSpec)
            else ModelSpec.portable_onnx(
                model_onnx_path,
                input_names=("obs", "time_step"),
                output_names=("actions",),
            )
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)
        meta = dict(self._backend.metadata)

        metadata_joint_names = tuple(
            name.strip() for name in meta["joint_names"].split(",")
        )
        self.num_actions = len(metadata_joint_names)
        expected_model_layout = ELF3_ISAAC_JOINTS if isaac_order else ELF3_POLICY_JOINTS
        if metadata_joint_names != expected_model_layout.names:
            raise ValueError(
                "model joint_names metadata does not match the class-defined "
                f"layout for {type(self).__name__}"
            )
        self._default_policy_position = _csv_floats(meta["default_joint_pos"])
        self._action_scale = _csv_floats(meta["action_scale"])
        if isaac_order:
            self.mujoco_to_isaac_idx = MUJOCO_TO_ISAAC.tolist()
            self.isaac_to_mujoco_idx = ISAAC_TO_MUJOCO.tolist()
            self._default_position = self._default_policy_position[
                ISAAC_TO_MUJOCO
            ].copy()
            self._kp = ELF3_KPS.copy()
            self._kd = ELF3_KDS.copy()
            self._decode_scale = self._action_scale[ISAAC_TO_MUJOCO]
        else:
            self._kp = _csv_floats(meta["joint_stiffness"])
            self._kd = _csv_floats(meta["joint_damping"])
            self._default_position = self._default_policy_position
            self._decode_scale = self._action_scale

        self._start_frame = int(start_frame)
        self._end_frame = self._motion_position.shape[0] - 1
        self._frame = float(self._start_frame)
        self._obs = np.zeros(self.num_obs, dtype=np.float32)
        self._input = self._obs.reshape(1, -1)
        self._time_input = np.zeros((1, 1), dtype=np.float32)
        self._inputs = {"obs": self._input, "time_step": self._time_input}
        self._action = np.zeros(self.num_actions, dtype=np.float32)
        self._previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self._alignment_checkpoint = np.empty((3, 3), dtype=np.float64)
        self._target = self._target_buffer.position
        np.copyto(self._target, self._default_position)
        self._joint_delta = np.empty(self.num_actions, dtype=np.float32)
        self._policy_joint_delta = np.empty(self.num_actions, dtype=np.float32)
        self._policy_velocity = np.empty(self.num_actions, dtype=np.float32)
        self._decoded_action = np.empty(self.num_actions, dtype=np.float32)
        self._scaled_action = np.empty(self.num_actions, dtype=np.float32)
        self.publish_output(self._target, self._kp, self._kd)

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        alignment_samples = self._alignment_samples
        if not advance and alignment_samples:
            np.copyto(self._alignment_checkpoint, self.init_to_world)
        joints = self.bind_joints(frame)
        monitor = self._runtime.options.monitor_enabled
        if monitor:
            started = time.perf_counter_ns()
        self._build_input(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
        )
        self._time_input[0, 0] = int(self._frame)
        if monitor:
            input_finished = time.perf_counter_ns()
        raw = self._backend.run(self._inputs)["actions"]
        if monitor:
            backend_finished = time.perf_counter_ns()
        np.copyto(self._action, np.asarray(raw).reshape(-1))
        if advance:
            np.copyto(self._previous_action, self._action)

        if self._isaac_order:
            np.take(self._action, ISAAC_TO_MUJOCO, out=self._decoded_action)
        else:
            np.copyto(self._decoded_action, self._action)
        np.multiply(self._decoded_action, self._decode_scale, out=self._scaled_action)
        np.add(self._default_position, self._scaled_action, out=self._target)
        if advance:
            self.advance(dt)
        else:
            self._alignment_samples = alignment_samples
            if alignment_samples:
                np.copyto(self.init_to_world, self._alignment_checkpoint)
        if monitor:
            finished = time.perf_counter_ns()
            self._record(started, input_finished, backend_finished, finished)
        self._policy_output.completed = self.finished()
        return self.output

    def _build_input(self, q, dq, quat, omega):
        t = min(int(self._frame), self.motioninputpos.shape[0] - 1)
        motion_quat = self.motionquat[t, 0, :]
        if self.fixed_pos:
            if self._alignment_samples < 2:
                self._alignment_samples += 1
                self.compute_init_to_world(quat, motion_quat)
        else:
            self.compute_init_to_world(quat, motion_quat)

        obs = self._obs
        obs[0:29] = self.motioninputpos[t]
        obs[29:58] = self.motioninputvel[t]
        offset = 58
        obs[offset : offset + 6] = self.compute_relmatrix(quat, motion_quat)
        offset += 6
        if self._include_gravity:
            obs[offset : offset + 3] = get_gravity_orientation(quat)
            offset += 3
        obs[offset : offset + 3] = omega
        offset += 3

        if self._isaac_order:
            np.subtract(q, self._default_position, out=self._joint_delta)
            np.take(self._joint_delta, MUJOCO_TO_ISAAC, out=self._policy_joint_delta)
            np.take(dq, MUJOCO_TO_ISAAC, out=self._policy_velocity)
            obs[offset : offset + 29] = self._policy_joint_delta
            offset += 29
            obs[offset : offset + 29] = self._policy_velocity
        else:
            np.subtract(
                q,
                self._default_policy_position,
                out=obs[offset : offset + 29],
            )
            offset += 29
            obs[offset : offset + 29] = dq
        offset += 29
        obs[offset : offset + 29] = self._previous_action

    def configure_range(self, start_frame=None, end_frame=None) -> None:
        if start_frame is not None:
            self._start_frame = int(start_frame)
        if end_frame is not None:
            self._end_frame = int(end_frame)

    def reset(self, frame: InferenceFrame) -> None:
        self.bind_joints(frame)
        self._frame = float(self._start_frame)
        self._alignment_samples = 0
        self._action.fill(0.0)
        self._previous_action.fill(0.0)
        np.copyto(self._target, self._default_position)
        self.publish_output(self._target, self._kp, self._kd)

    def advance(self, dt: float) -> None:
        self._frame += 50.0 * dt

    def finished(self, trim: int = 0) -> bool:
        return self._frame > self._end_frame - trim

    def _record(self, start, input_done, backend_done, done):
        if self._runtime.options.monitor_enabled:
            self._runtime.monitor.record(
                self._policy_name,
                input_done - start,
                backend_done - input_done,
                done - backend_done,
                done - start,
            )

    def close(self):
        self._backend.close()
        self.motion.close()


class DanceMotionPolicyMjlab(_LegacyMotionPolicy):
    def __init__(
        self, motion_npz_path, model_onnx_path, start_frame=0, fixed_pos=False, **kwargs
    ):
        super().__init__(
            motion_npz_path,
            model_onnx_path,
            start_frame,
            fixed_pos,
            include_gravity=False,
            isaac_order=False,
            **kwargs,
        )


class DanceMotionPolicyGravityMjlab(_LegacyMotionPolicy):
    def __init__(
        self, motion_npz_path, model_onnx_path, start_frame=0, fixed_pos=False, **kwargs
    ):
        super().__init__(
            motion_npz_path,
            model_onnx_path,
            start_frame,
            fixed_pos,
            include_gravity=True,
            isaac_order=False,
            **kwargs,
        )


class DanceMotionPolicyGravityIsaaclab(_LegacyMotionPolicy):
    def __init__(
        self, motion_npz_path, model_onnx_path, start_frame=0, fixed_pos=False, **kwargs
    ):
        super().__init__(
            motion_npz_path,
            model_onnx_path,
            start_frame,
            fixed_pos,
            include_gravity=True,
            isaac_order=True,
            **kwargs,
        )


class _HistoryMotionPolicy(_MotionGeometry, JointPolicy):
    joint_contract = PolicyJointContract(
        observation=ELF3_POLICY_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        motion_npz_path: str,
        model_onnx_path: str | ModelSpec,
        start_frame: int = 0,
        fixed_pos: bool = False,
        *,
        reference_residual: bool,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
        JointPolicy.__init__(self)
        self.motion_npz_path = motion_npz_path
        self.fixed_pos = fixed_pos
        self._alignment_samples = 0
        self._reference_residual = reference_residual
        self._runtime = runtime or default_runtime()
        self._policy_name = "motion-history"
        self.mujoco_to_isaac_idx = MUJOCO_TO_ISAAC.tolist()
        self.isaac_to_mujoco_idx = ISAAC_TO_MUJOCO.tolist()
        self._kp = ELF3_KPS.copy()
        self._kd = ELF3_KDS.copy()

        self.motion = np.load(motion_npz_path)
        self._motion_position = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]

        output_names = ("actions", "joint_pos") if reference_residual else ("actions",)
        spec = (
            model_onnx_path
            if isinstance(model_onnx_path, ModelSpec)
            else ModelSpec.portable_onnx(
                model_onnx_path,
                input_names=("obs", "time_step"),
                output_names=output_names,
            )
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)
        meta = dict(self._backend.metadata)
        metadata_joint_names = tuple(
            name.strip() for name in meta["joint_names"].split(",")
        )
        self.num_actions = len(metadata_joint_names)
        if metadata_joint_names != ELF3_ISAAC_JOINTS.names:
            raise ValueError(
                "model joint_names metadata does not match the class-defined "
                f"layout for {type(self).__name__}"
            )
        self._default_policy_position = _csv_floats(meta["default_joint_pos"])
        self._default_position = self._default_policy_position[ISAAC_TO_MUJOCO].copy()
        self._action_scale = _csv_floats(meta["action_scale"])
        self.command_window_offsets = np.fromstring(
            meta.get("command_window_offsets", "-2,0,2,5,10"), sep=",", dtype=np.float32
        ).astype(np.int64)
        self.observation_names = meta.get(
            "observation_names",
            "command,motion_anchor_ori_b,projected_gravity,base_ang_vel,joint_pos,joint_vel,actions",
        ).split(",")
        self.observation_history_lengths = [
            int(float(value))
            for value in meta.get(
                "observation_history_lengths", "1,1,10,10,10,10,10"
            ).split(",")
        ]

        self._start_frame = int(start_frame)
        self._end_frame = self.motioninputpos.shape[0] - 1
        self._frame = float(self._start_frame)
        self.num_obs = int(self._backend.input_shape("obs")[1])
        self._obs = np.zeros(self.num_obs, dtype=np.float32)
        self._input = self._obs.reshape(1, -1)
        self._time_input = np.zeros((1, 1), dtype=np.float32)
        self._inputs = {"obs": self._input, "time_step": self._time_input}
        self._action = np.zeros(self.num_actions, dtype=np.float32)
        self._previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self._alignment_checkpoint = np.empty((3, 3), dtype=np.float64)
        self._target = self._target_buffer.position
        np.copyto(self._target, self._default_position)
        self._target_isaac = np.empty(self.num_actions, dtype=np.float32)
        self._scaled_action = np.empty(self.num_actions, dtype=np.float32)
        self._joint_delta = np.empty(self.num_actions, dtype=np.float32)

        feature_dims = {
            "command": int(self.command_window_offsets.size * self.num_actions * 2),
            "motion_anchor_ori_b": 6,
            "projected_gravity": 3,
            "base_ang_vel": 3,
            "base_lin_vel": 3,
            "joint_pos": self.num_actions,
            "joint_vel": self.num_actions,
            "actions": self.num_actions,
        }
        if "motion_anchor_pos_b" in self.observation_names:
            raise ValueError("models requiring motion_anchor_pos_b are not supported")
        self._feature_values = {
            name: np.empty(feature_dims[name], dtype=np.float32)
            for name in self.observation_names
        }
        self.history_buffers: dict[str, HistoryBuffer] = {}
        self._feature_slices: dict[str, slice] = {}
        offset = 0
        for name, history_length in zip(
            self.observation_names, self.observation_history_lengths
        ):
            dim = feature_dims[name]
            width = dim * history_length
            self._feature_slices[name] = slice(offset, offset + width)
            if history_length > 1:
                self.history_buffers[name] = HistoryBuffer(history_length, (dim,))
            offset += width
        if offset != self.num_obs:
            raise ValueError(f"obs dim mismatch: layout={offset}, model={self.num_obs}")
        self.publish_output(self._target, self._kp, self._kd)

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        alignment_samples = self._alignment_samples
        if not advance and alignment_samples:
            np.copyto(self._alignment_checkpoint, self.init_to_world)
        joints = self.bind_joints(frame)
        monitor = self._runtime.options.monitor_enabled
        if monitor:
            started = time.perf_counter_ns()
        self._build_input(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
            frame.base_linear_velocity,
            frame.world_position,
            advance_history=advance,
        )
        self._time_input[0, 0] = int(self._frame)
        if monitor:
            input_done = time.perf_counter_ns()
        outputs = self._backend.run(self._inputs)
        if monitor:
            backend_done = time.perf_counter_ns()
        np.copyto(self._action, np.asarray(outputs["actions"]).reshape(-1))
        if advance:
            np.copyto(self._previous_action, self._action)
        np.multiply(self._action, self._action_scale, out=self._scaled_action)
        if self._reference_residual:
            reference = np.asarray(outputs["joint_pos"]).reshape(-1)
            np.add(reference, self._scaled_action, out=self._target_isaac)
        else:
            np.add(
                self._default_policy_position,
                self._scaled_action,
                out=self._target_isaac,
            )
        np.take(self._target_isaac, ISAAC_TO_MUJOCO, out=self._target)
        if advance:
            self.advance(dt)
        else:
            self._alignment_samples = alignment_samples
            if alignment_samples:
                np.copyto(self.init_to_world, self._alignment_checkpoint)
        if monitor:
            done = time.perf_counter_ns()
            self._runtime.monitor.record(
                self._policy_name,
                input_done - started,
                backend_done - input_done,
                done - backend_done,
                done - started,
            )
        self._policy_output.completed = self.finished()
        return self.output

    def _build_input(
        self,
        q,
        dq,
        quat,
        omega,
        base_lin_vel_b=None,
        robot_pos_w=None,
        *,
        advance_history: bool,
    ):
        t = min(int(self._frame), self._end_frame)
        motion_quat = self.motionquat[t, 0, :]
        if self.fixed_pos:
            if self._alignment_samples < 2:
                self._alignment_samples += 1
                self.compute_init_to_world(quat, motion_quat)
        else:
            self.compute_init_to_world(quat, motion_quat)

        values = self._feature_values
        self._build_command_window_into(t, values["command"])
        values["motion_anchor_ori_b"][:] = self.compute_relmatrix(quat, motion_quat)
        values["projected_gravity"][:] = get_gravity_orientation(quat)
        values["base_ang_vel"][:] = omega
        np.subtract(q, self._default_position, out=self._joint_delta)
        np.take(self._joint_delta, MUJOCO_TO_ISAAC, out=values["joint_pos"])
        np.take(dq, MUJOCO_TO_ISAAC, out=values["joint_vel"])
        values["actions"][:] = self._previous_action
        if "base_lin_vel" in values:
            if base_lin_vel_b is None:
                raise ValueError("current model requires base_lin_vel_b")
            values["base_lin_vel"][:] = base_lin_vel_b

        for name, history_length in zip(
            self.observation_names, self.observation_history_lengths
        ):
            target = self._obs[self._feature_slices[name]]
            value = values[name]
            if history_length == 1:
                target[:] = value
                continue
            history = self.history_buffers[name]
            if not history.initialized:
                if advance_history:
                    history.fill(value)
                    history.write_into(target)
                else:
                    target.reshape((history_length, *value.shape))[...] = value
            elif advance_history:
                history.append(value)
                history.write_into(target)
            else:
                history.preview_append_into(value, target)

    def _build_command_window_into(self, timestep: int, output: np.ndarray):
        offset = 0
        for frame_offset in self.command_window_offsets:
            frame = min(max(timestep + int(frame_offset), 0), self._end_frame)
            output[offset : offset + self.num_actions] = self.motioninputpos[frame]
            offset += self.num_actions
            output[offset : offset + self.num_actions] = self.motioninputvel[frame]
            offset += self.num_actions

    def configure_range(self, start_frame=None, end_frame=None) -> None:
        if start_frame is not None:
            self._start_frame = int(start_frame)
        if end_frame is not None:
            self._end_frame = int(end_frame)

    def reset(self, frame: InferenceFrame) -> None:
        self.bind_joints(frame)
        self._frame = float(self._start_frame)
        self._alignment_samples = 0
        self._action.fill(0.0)
        self._previous_action.fill(0.0)
        np.copyto(self._target, self._default_position)
        for history in self.history_buffers.values():
            history.clear()
        self.publish_output(self._target, self._kp, self._kd)

    def advance(self, dt: float) -> None:
        self._frame += 50.0 * dt

    def finished(self, trim: int = 0) -> bool:
        return self._frame > self._end_frame - trim

    def close(self):
        self._backend.close()
        self.motion.close()


class DanceMotionPolicyGravityIsaaclabV2(_HistoryMotionPolicy):
    def __init__(
        self, motion_npz_path, model_onnx_path, start_frame=0, fixed_pos=False, **kwargs
    ):
        super().__init__(
            motion_npz_path,
            model_onnx_path,
            start_frame,
            fixed_pos,
            reference_residual=True,
            **kwargs,
        )


class DanceMotionPolicyGravityIsaaclabV3(_HistoryMotionPolicy):
    def __init__(
        self, motion_npz_path, model_onnx_path, start_frame=0, fixed_pos=False, **kwargs
    ):
        super().__init__(
            motion_npz_path,
            model_onnx_path,
            start_frame,
            fixed_pos,
            reference_residual=False,
            **kwargs,
        )


__all__ = [
    "DanceMotionPolicyGravityIsaaclab",
    "DanceMotionPolicyGravityIsaaclabV2",
    "DanceMotionPolicyGravityIsaaclabV3",
    "DanceMotionPolicyGravityMjlab",
    "DanceMotionPolicyMjlab",
]
