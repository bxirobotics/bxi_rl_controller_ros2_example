"""Normal locomotion policy using the shared inference runtime."""

from __future__ import annotations

import ast
import time

import numpy as np

from .model import ModelSpec
from .runtime import InferenceRuntime, default_runtime


dof_num = 29


class NormalMotionPolicyMjlab:
    """96D proprioceptive locomotion policy."""

    def __init__(
        self,
        model: str | ModelSpec,
        *,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
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
        self.kp = self._metadata_array(metadata, "joint_stiffness")
        self.kd = self._metadata_array(metadata, "joint_damping")
        self._action_scale = self._metadata_array(metadata, "action_scale")
        self.default_position = self._metadata_array(metadata, "default_joint_pos")

        self._obs = np.zeros((1, 96), dtype=np.float32)
        self._action = np.zeros(dof_num, dtype=np.float32)
        self._scaled_action = np.zeros(dof_num, dtype=np.float32)
        self.target = self.default_position.copy()
        self._gravity = np.empty(3, dtype=np.float32)
        self._inputs = {"obs": self._obs}
        self._backend.warmup(self._inputs, self._runtime.options.warmup_runs)

    @staticmethod
    def _metadata_array(metadata: dict[str, str], key: str) -> np.ndarray:
        return np.asarray(ast.literal_eval(metadata[key]), dtype=np.float32)

    def reset(self, *_state) -> None:
        self._action.fill(0.0)
        self._scaled_action.fill(0.0)

    def step(self, q, dq, quat, omega, command):
        monitor = self._runtime.options.monitor_enabled
        if monitor:
            total_started = time.perf_counter_ns()
        obs = self._obs[0]
        obs[0:3] = omega
        self._project_gravity(quat, self._gravity)
        obs[3:6] = self._gravity
        np.subtract(q, self.default_position, out=obs[6:35])
        obs[35:64] = dq
        obs[64:93] = self._action
        obs[93:96] = command
        if monitor:
            input_finished = time.perf_counter_ns()

        outputs = self._backend.run(self._inputs)
        if monitor:
            backend_finished = time.perf_counter_ns()
        np.copyto(self._action, np.asarray(outputs["actions"]).reshape(-1))
        np.multiply(self._action, self._action_scale, out=self._scaled_action)
        np.add(self.default_position, self._scaled_action, out=self.target)
        if monitor:
            output_finished = time.perf_counter_ns()

        if monitor:
            self._runtime.monitor.record(
                self._policy_name,
                input_finished - total_started,
                backend_finished - input_finished,
                output_finished - backend_finished,
                output_finished - total_started,
            )
        return self.target

    @staticmethod
    def _project_gravity(quaternion, output) -> None:
        w, x, y, z = quaternion
        output[0] = 2.0 * (w * y - x * z)
        output[1] = -2.0 * (w * x + y * z)
        output[2] = 2.0 * (x * x + y * y) - 1.0

    def close(self) -> None:
        self._backend.close()


__all__ = ["NormalMotionPolicyMjlab"]
