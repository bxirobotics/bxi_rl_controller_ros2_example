"""Allocation-free composition of state-owned joint command sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..joints import JointLayout, JointTargetView
from .frame import MotorFrame


@dataclass(frozen=True, slots=True)
class JointCommandLayer:
    """One persistent named PD target participating in state composition.

    The target may be updated by a policy, trajectory, topic snapshot, IK or any
    other producer. Composition only reads its current arrays. Later layers may
    replace an existing owner only when ``override`` is explicitly enabled.
    """

    name: str
    target: JointTargetView
    override: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("joint command layer name must be a non-empty string")
        if not isinstance(self.target, JointTargetView):
            raise TypeError("joint command layer target must be a JointTargetView")
        if not isinstance(self.override, bool):
            raise TypeError("joint command layer override must be a bool")


@dataclass(frozen=True, slots=True)
class _CompiledLayer:
    layer: JointCommandLayer
    source_layout: JointLayout
    output_indices: NDArray[np.intp]
    is_identity: bool


class JointCommandComposer:
    """Compose persistent command layers into one reusable natural MotorFrame.

    Source ownership and all name mappings are validated and compiled once.
    ``compose()`` performs only array writes into a long-lived output frame.
    """

    __slots__ = ("output_layout", "layers", "_compiled", "_frame")

    def __init__(
        self,
        output_layout: JointLayout,
        layers: Sequence[JointCommandLayer],
    ) -> None:
        if not isinstance(output_layout, JointLayout):
            raise TypeError("composer output_layout must be a JointLayout")
        declared_layers = tuple(layers)
        if not declared_layers:
            raise ValueError("joint command composer requires at least one layer")
        if not all(isinstance(layer, JointCommandLayer) for layer in declared_layers):
            raise TypeError("composer layers must be JointCommandLayer instances")

        layer_names = tuple(layer.name for layer in declared_layers)
        duplicate_layer_names = tuple(
            name for name in dict.fromkeys(layer_names) if layer_names.count(name) > 1
        )
        if duplicate_layer_names:
            raise ValueError(
                f"joint command layer names must be unique: {duplicate_layer_names}"
            )

        output_names = set(output_layout.names)
        owners: dict[str, str] = {}
        compiled: list[_CompiledLayer] = []
        for layer in declared_layers:
            source_layout = layer.target.layout
            outside = tuple(
                name for name in source_layout.names if name not in output_names
            )
            if outside:
                raise ValueError(
                    f"joint command layer '{layer.name}' contains joints outside "
                    f"the composer output layout: {outside}"
                )

            conflicts = tuple(
                (name, owners[name])
                for name in source_layout.names
                if name in owners and not layer.override
            )
            if conflicts:
                details = tuple(
                    f"{name} (owned by {owner})" for name, owner in conflicts
                )
                raise ValueError(
                    f"joint command layer '{layer.name}' has ownership conflicts: "
                    f"{details}; declare override=True only when replacement is "
                    "intentional"
                )

            for name in source_layout.names:
                owners[name] = layer.name
            indices = np.fromiter(
                (output_layout.index(name) for name in source_layout.names),
                dtype=np.intp,
                count=source_layout.dof_num,
            )
            indices.flags.writeable = False
            compiled.append(
                _CompiledLayer(
                    layer=layer,
                    source_layout=source_layout,
                    output_indices=indices,
                    is_identity=source_layout.names == output_layout.names,
                )
            )

        missing = tuple(name for name in output_layout.names if name not in owners)
        if missing:
            raise ValueError(
                "joint command layers do not cover the complete composer output "
                f"layout: {missing}"
            )

        self.output_layout = output_layout
        self.layers = declared_layers
        self._compiled = tuple(compiled)
        self._frame = MotorFrame.empty(output_layout)

    @property
    def frame(self) -> MotorFrame:
        return self._frame

    def compose(self) -> MotorFrame:
        """Write current layer values into and return the reusable output frame."""
        for binding in self._compiled:
            target = binding.layer.target
            if (
                target.layout is not binding.source_layout
                and target.layout.names != binding.source_layout.names
            ):
                raise ValueError(
                    f"joint command layer '{binding.layer.name}' changed layout "
                    "after composer compilation"
                )
            for source, output in (
                (target.position, self._frame.qpos),
                (target.kp, self._frame.kp),
                (target.kd, self._frame.kd),
            ):
                if binding.is_identity:
                    np.copyto(output, source)
                else:
                    output[binding.output_indices] = source
        return self._frame


__all__ = ["JointCommandComposer", "JointCommandLayer"]
