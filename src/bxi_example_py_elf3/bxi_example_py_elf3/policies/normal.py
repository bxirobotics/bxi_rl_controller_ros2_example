"""Normal locomotion policy using the shared inference runtime."""

from __future__ import annotations

import ast

import numpy as np

from .joints import ELF3_POLICY_JOINTS

from bxi_example_py_elf3.framework.inference.api import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.inference.contract import PolicyJointContract
from bxi_example_py_elf3.framework.inference.model import ModelSpec
from bxi_example_py_elf3.framework.inference.policy import JointPolicy
from bxi_example_py_elf3.framework.inference.runtime import (
    InferenceRuntime,
    default_runtime,
)
from bxi_example_py_elf3.framework.joints import JointParameterSet


class NormalMotionPolicyMjlab(JointPolicy):
    """96D proprioceptive locomotion policy."""

    joint_contract = PolicyJointContract(
        observation=ELF3_POLICY_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        model: str | ModelSpec,
        *,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self._runtime = runtime or default_runtime()
        self._policy_name = "normal"
        spec = (
            model
            if isinstance(model, ModelSpec)
            else ModelSpec.portable_onnx(
                model,
                input_names=("obs",),
                output_names=("actions",),
            )
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)

        metadata = dict(self._backend.metadata)
        raw_names = metadata.get("joint_names")
        if not raw_names:
            raise ValueError("model metadata is missing required joint_names")
        names = tuple(name.strip() for name in raw_names.split(","))
        if names != self.joint_contract.observation.names:
            raise ValueError(
                "model joint_names metadata does not match the class-defined "
                "NormalMotionPolicyMjlab joint contract"
            )
        self._parameters = JointParameterSet.from_arrays(
            self.joint_contract.action,
            default_position=self._metadata_array(metadata, "default_joint_pos"),
            kp=self._metadata_array(metadata, "joint_stiffness"),
            kd=self._metadata_array(metadata, "joint_damping"),
            action_scale=self._metadata_array(metadata, "action_scale"),
        )

        self._obs = np.zeros((1, 96), dtype=np.float32)
        count = self.joint_contract.action.dof_num
        self._action = np.zeros(count, dtype=np.float32)
        self._action_checkpoint = np.empty_like(self._action)
        self._scaled_action = np.zeros(count, dtype=np.float32)
        self._target = self._target_buffer.position
        np.copyto(self._target, self._parameters.default_position)
        self._gravity = np.empty(3, dtype=np.float32)
        self._inputs = {"obs": self._obs}
        self._backend.warmup(self._inputs, self._runtime.options.warmup_runs)
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    @staticmethod
    def _metadata_array(metadata: dict[str, str], key: str) -> np.ndarray:
        return np.asarray(ast.literal_eval(metadata[key]), dtype=np.float32)

    def reset(self, frame: InferenceFrame) -> None:
        self.bind_joints(frame)
        self._action.fill(0.0)
        self._scaled_action.fill(0.0)
        np.copyto(self._target, self._parameters.default_position)
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        del dt
        if not advance:
            np.copyto(self._action_checkpoint, self._action)
        joints = self.bind_joints(frame)
        command = frame.command
        if command is None:
            raise ValueError("NormalMotionPolicyMjlab requires frame.command")
        obs = self._obs[0]
        obs[0:3] = frame.angular_velocity
        self._project_gravity(frame.quat_wxyz, self._gravity)
        obs[3:6] = self._gravity
        np.subtract(
            joints.position,
            self._parameters.default_position,
            out=obs[6:35],
        )
        obs[35:64] = joints.velocity
        obs[64:93] = self._action
        obs[93:96] = command
        outputs = self._backend.run(self._inputs)
        np.copyto(self._action, np.asarray(outputs["actions"]).reshape(-1))
        np.multiply(
            self._action,
            self._parameters.action_scale,
            out=self._scaled_action,
        )
        np.add(
            self._parameters.default_position,
            self._scaled_action,
            out=self._target,
        )
        if not advance:
            np.copyto(self._action, self._action_checkpoint)
        return self.output

    @staticmethod
    def _project_gravity(quaternion, output) -> None:
        w, x, y, z = quaternion
        output[0] = 2.0 * (w * y - x * z)
        output[1] = -2.0 * (w * x + y * z)
        output[2] = 2.0 * (x * x + y * y) - 1.0

    def close(self) -> None:
        self._backend.close()


__all__ = ["NormalMotionPolicyMjlab"]
