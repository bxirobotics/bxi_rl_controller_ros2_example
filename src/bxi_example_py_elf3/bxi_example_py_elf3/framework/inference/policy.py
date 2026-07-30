"""Simple composition API for new inference policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.joints import (
    JointStateView,
    JointTargetBuffer,
)

from .api import InferenceFrame, PolicyOutput
from .contract import JointInputBinding, PolicyJointContract
from .model import ModelSpec
from .runtime import InferenceRuntime, default_runtime


class InputBuilder(ABC):
    """Own stable model input buffers and update them in place."""

    @property
    @abstractmethod
    def joint_contract(self) -> PolicyJointContract:
        ...

    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, NDArray[np.generic]]:
        ...

    def reset(self, frame: InferenceFrame, joints: JointStateView) -> None:
        pass

    @abstractmethod
    def build_into(
        self,
        frame: InferenceFrame,
        joints: JointStateView,
        dt: float,
        *,
        advance: bool,
    ) -> None:
        ...


class OutputDecoder(ABC):
    """Decode backend tensors into one reusable policy output."""

    @property
    @abstractmethod
    def output(self) -> PolicyOutput:
        ...


class JointPolicy:
    """Small base for hand-written policies with a class-defined joint contract.

    It owns the name-to-index binding and one stable output object. Subclasses
    only build tensors, run their backend and call :meth:`publish_output`.
    """

    joint_contract: PolicyJointContract

    def __init__(self) -> None:
        contract = getattr(type(self), "joint_contract", None)
        if not isinstance(contract, PolicyJointContract):
            raise TypeError(
                f"{type(self).__name__}.joint_contract must be a "
                "PolicyJointContract"
            )
        self._joint_binding = JointInputBinding(contract)
        self._target_buffer = JointTargetBuffer(contract.action)
        self._policy_output = PolicyOutput(joints=self._target_buffer.view)

    @property
    def output(self) -> PolicyOutput:
        return self._policy_output

    def bind_joints(self, frame: InferenceFrame) -> JointStateView:
        return self._joint_binding.bind(frame.joints)

    def publish_output(
        self,
        position: object,
        kp: object | None = None,
        kd: object | None = None,
        *,
        estimated_velocity: NDArray[np.float32] | None = None,
        completed: bool = False,
    ) -> PolicyOutput:
        if (kp is None) != (kd is None):
            raise ValueError("kp and kd must either both be provided or both omitted")
        if kp is None:
            self._target_buffer.update_position(position)
        else:
            self._target_buffer.update(position, kp, kd)
        self._policy_output.estimated_velocity = estimated_velocity
        self._policy_output.completed = bool(completed)
        return self._policy_output

    def reset(self) -> None:
        pass

    @abstractmethod
    def decode_into(self, outputs: Mapping[str, NDArray[np.generic]]) -> None:
        ...


class Policy:
    """Backend-neutral build -> run -> decode template for new policies."""

    def __init__(
        self,
        model: ModelSpec,
        input_builder: InputBuilder,
        output_decoder: OutputDecoder,
        *,
        name: str,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
        self.name = name
        self.input_builder = input_builder
        self.output_decoder = output_decoder
        self.runtime = runtime or default_runtime()
        self.backend = self.runtime.open_backend(model, backend=backend)
        self._joint_binding = JointInputBinding(input_builder.joint_contract)
        if output_decoder.output.joints.layout != input_builder.joint_contract.action:
            raise ValueError(
                "output decoder joint layout does not match policy action contract"
            )

    @property
    def output(self) -> PolicyOutput:
        return self.output_decoder.output

    def reset(self, frame: InferenceFrame) -> None:
        joints = self._joint_binding.bind(frame.joints)
        self.input_builder.reset(frame, joints)
        self.output_decoder.reset()

    def prepare(self, frame: InferenceFrame) -> None:
        self.reset(frame)
        joints = self._joint_binding.bind(frame.joints)
        self.input_builder.build_into(frame, joints, 0.0, advance=False)
        runs = max(1, self.runtime.options.warmup_runs)
        for _ in range(runs):
            outputs = self.backend.run(self.input_builder.inputs)
        self.output_decoder.decode_into(outputs)

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        joints = self._joint_binding.bind(frame.joints)
        self.input_builder.build_into(frame, joints, dt, advance=advance)
        outputs = self.backend.run(self.input_builder.inputs)
        self.output_decoder.decode_into(outputs)
        return self.output_decoder.output

    def close(self) -> None:
        self.backend.close()


__all__ = ["InputBuilder", "JointPolicy", "OutputDecoder", "Policy"]
