"""Simple composition API for new inference policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import time

import numpy as np
from numpy.typing import NDArray

from .api import InferenceFrame, PolicyOutput
from .model import ModelSpec
from .runtime import InferenceRuntime, default_runtime


class InputBuilder(ABC):
    """Own stable model input buffers and update them in place."""

    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, NDArray[np.generic]]:
        ...

    def reset(self, frame: InferenceFrame) -> None:
        pass

    @abstractmethod
    def build_into(
        self,
        frame: InferenceFrame,
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

    @property
    def output(self) -> PolicyOutput:
        return self.output_decoder.output

    def reset(self, frame: InferenceFrame) -> None:
        self.input_builder.reset(frame)
        self.output_decoder.reset()

    def prepare(self, frame: InferenceFrame) -> None:
        self.reset(frame)
        self.input_builder.build_into(frame, 0.0, advance=False)
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
        monitor = self.runtime.options.monitor_enabled
        if monitor:
            started = time.perf_counter_ns()
        self.input_builder.build_into(frame, dt, advance=advance)
        if monitor:
            input_done = time.perf_counter_ns()
        outputs = self.backend.run(self.input_builder.inputs)
        if monitor:
            backend_done = time.perf_counter_ns()
        self.output_decoder.decode_into(outputs)
        if monitor:
            done = time.perf_counter_ns()
            self.runtime.monitor.record(
                self.name,
                input_done - started,
                backend_done - input_done,
                done - backend_done,
                done - started,
            )
        return self.output_decoder.output

    def close(self) -> None:
        self.backend.close()


__all__ = ["InputBuilder", "OutputDecoder", "Policy"]
