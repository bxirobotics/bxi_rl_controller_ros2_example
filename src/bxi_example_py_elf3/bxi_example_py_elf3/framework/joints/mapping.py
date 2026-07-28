"""Precompiled allocation-free joint order mappings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layout import JointLayout


@dataclass(frozen=True, slots=True)
class CompiledJointMap:
    """Map one-dimensional values from ``source`` into ``target`` order."""

    source: JointLayout
    target: JointLayout
    indices: NDArray[np.intp]
    is_identity: bool

    @classmethod
    def compile(
        cls,
        source: JointLayout,
        target: JointLayout,
        *,
        require_exact: bool = False,
    ) -> "CompiledJointMap":
        missing = target.missing_from(source)
        if missing:
            raise ValueError(
                f"source joint layout is missing joints required by "
                f"'{target.label or 'target'}': {missing}"
            )
        if require_exact and set(source.names) != set(target.names):
            extra = tuple(name for name in source.names if name not in set(target.names))
            raise ValueError(
                f"joint layouts must contain the same names; source-only={extra}"
            )
        indices = np.fromiter(
            (source.index(name) for name in target.names),
            dtype=np.intp,
            count=target.dof_num,
        )
        indices.flags.writeable = False
        identity = source.names == target.names
        return cls(source, target, indices, identity)

    def map_into(self, source_values: object, target_values: object) -> None:
        source_array = np.asarray(source_values)
        target_array = np.asarray(target_values)
        if source_array.shape != (self.source.dof_num,):
            raise ValueError(
                f"source joint values have shape {source_array.shape}, expected "
                f"{(self.source.dof_num,)}"
            )
        if target_array.shape != (self.target.dof_num,):
            raise ValueError(
                f"target joint values have shape {target_array.shape}, expected "
                f"{(self.target.dof_num,)}"
            )
        if self.is_identity:
            np.copyto(target_array, source_array, casting="same_kind")
        else:
            np.take(source_array, self.indices, out=target_array)


__all__ = ["CompiledJointMap"]
