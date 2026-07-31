"""Class-defined policy joint contracts and compiled input bindings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bxi_example_py_elf3.framework.joints import (
    CompiledJointMap,
    JointLayout,
    JointStateBuffer,
    JointStateView,
)


@dataclass(frozen=True, slots=True)
class PolicyJointContract:
    observation: JointLayout
    action: JointLayout


class JointInputBinding:
    """Map arbitrary named robot state into one policy observation layout."""

    def __init__(
        self,
        contract: PolicyJointContract,
        *,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self.contract = contract
        self._buffer = JointStateBuffer(contract.observation, dtype=dtype)
        self._source_layout: JointLayout | None = None
        self._mapping: CompiledJointMap | None = None

    @property
    def joints(self) -> JointStateView:
        return self._buffer.view

    def bind(self, source: JointStateView) -> JointStateView:
        if source.layout is self.contract.observation:
            return source
        if self._source_layout is not source.layout:
            self._mapping = CompiledJointMap.compile(
                source.layout,
                self.contract.observation,
            )
            self._source_layout = source.layout
        assert self._mapping is not None
        if self._mapping.is_identity:
            return source
        self._mapping.map_into(source.position, self._buffer.position)
        self._mapping.map_into(source.velocity, self._buffer.velocity)
        self._buffer.view.timestamp_ns = source.timestamp_ns
        return self._buffer.view


__all__ = ["JointInputBinding", "PolicyJointContract"]
