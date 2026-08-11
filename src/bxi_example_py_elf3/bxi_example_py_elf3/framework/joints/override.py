"""Thread-safe mailbox and allocation-stable named motor-command override."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from .layout import JointLayout

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api.frame import MotorFrame


Float32Array = NDArray[np.float32]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class _OverrideCommand:
    robot_names: tuple[str, ...]
    names: tuple[str, ...]
    indices: IndexArray
    qpos: Float32Array
    kp: Float32Array
    kd: Float32Array
    vel: Float32Array
    torque: Float32Array
    received_at: float


class NamedJointCommandOverride:
    """Apply the newest named command after state/policy layout resolution.

    ``submit`` is intended for a ROS callback thread. It validates and copies
    the complete input before atomically replacing one immutable mailbox
    reference. ``apply`` is intended for the control thread and reuses a
    full-layout output frame after the robot layout has been observed once.
    """

    __slots__ = (
        "timeout_sec",
        "release_blend_sec",
        "_mailbox",
        "_next_generation",
        "_seen_generation",
        "_command",
        "_last_applied_command",
        "_output",
        "_was_applied",
        "_release_active",
        "_release_started_at",
        "_release_indices",
        "_release_qpos",
        "_release_kp",
        "_release_kd",
        "_release_vel",
        "_release_torque",
    )

    def __init__(self, *, timeout_sec: float, release_blend_sec: float) -> None:
        timeout_sec = float(timeout_sec)
        release_blend_sec = float(release_blend_sec)
        if not math.isfinite(timeout_sec) or timeout_sec < 0.0:
            raise ValueError("motor override timeout_sec must be finite and non-negative")
        if not math.isfinite(release_blend_sec) or release_blend_sec < 0.0:
            raise ValueError(
                "motor override release_blend_sec must be finite and non-negative"
            )

        self.timeout_sec = timeout_sec
        self.release_blend_sec = release_blend_sec
        self._mailbox: tuple[int, _OverrideCommand | None] = (0, None)
        self._next_generation = 0
        self._seen_generation = -1
        self._command: _OverrideCommand | None = None
        self._last_applied_command: _OverrideCommand | None = None
        self._output: MotorFrame | None = None
        self._was_applied = False
        self._release_active = False
        self._release_started_at = 0.0
        self._release_indices = np.empty(0, dtype=np.intp)
        self._release_qpos = np.empty(0, dtype=np.float32)
        self._release_kp = np.empty(0, dtype=np.float32)
        self._release_kd = np.empty(0, dtype=np.float32)
        self._release_vel = np.empty(0, dtype=np.float32)
        self._release_torque = np.empty(0, dtype=np.float32)

    def submit(
        self,
        robot_layout: JointLayout,
        names: Sequence[str],
        qpos: object,
        kp: object,
        kd: object,
        *,
        vel: object | None = None,
        torque: object | None = None,
        received_at: float | None = None,
    ) -> tuple[str, ...]:
        """Validate and atomically publish one active override command."""
        command_names = tuple(names)
        if not command_names:
            raise ValueError("motor override must contain at least one joint")
        if any(not isinstance(name, str) or not name for name in command_names):
            raise ValueError("motor override joint names must be non-empty strings")
        if len(set(command_names)) != len(command_names):
            duplicates = tuple(
                sorted(
                    {
                        name
                        for name in command_names
                        if command_names.count(name) > 1
                    }
                )
            )
            raise ValueError(f"motor override contains duplicate joints: {duplicates}")

        unknown = tuple(name for name in command_names if name not in robot_layout.names)
        if unknown:
            raise ValueError(f"motor override contains unknown robot joints: {unknown}")

        arrays = tuple(
            np.asarray(values, dtype=np.float32).copy()
            for values in (qpos, kp, kd)
        )
        qpos_array, kp_array, kd_array = arrays
        expected = (len(command_names),)
        for field, array in (
            ("pos", qpos_array),
            ("kp", kp_array),
            ("kd", kd_array),
        ):
            if array.shape != expected:
                raise ValueError(
                    f"motor override {field} has shape {array.shape}, expected {expected}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"motor override {field} must contain only finite values")
        if np.any(kp_array < 0.0) or np.any(kd_array < 0.0):
            raise ValueError("motor override kp and kd must be non-negative")

        mit_arrays: list[Float32Array] = []
        for field, values in (("vel", vel), ("torque", torque)):
            if values is None:
                array = np.zeros(len(command_names), dtype=np.float32)
            else:
                array = np.asarray(values, dtype=np.float32).copy()
            if array.shape != expected:
                raise ValueError(
                    f"motor override {field} has shape {array.shape}, expected {expected}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"motor override {field} must contain only finite values"
                )
            mit_arrays.append(array)
        vel_array, torque_array = mit_arrays

        timestamp = time.monotonic() if received_at is None else float(received_at)
        if not math.isfinite(timestamp):
            raise ValueError("motor override receive time must be finite")

        indices = np.asarray(
            tuple(robot_layout.index(name) for name in command_names),
            dtype=np.intp,
        )
        for array in (
            indices,
            qpos_array,
            kp_array,
            kd_array,
            vel_array,
            torque_array,
        ):
            array.flags.writeable = False

        command = _OverrideCommand(
            robot_names=robot_layout.names,
            names=command_names,
            indices=indices,
            qpos=qpos_array,
            kp=kp_array,
            kd=kd_array,
            vel=vel_array,
            torque=torque_array,
            received_at=timestamp,
        )
        self._next_generation += 1
        self._mailbox = (self._next_generation, command)
        return command_names

    def clear(self) -> None:
        """Atomically request release of the current override."""
        self._next_generation += 1
        self._mailbox = (self._next_generation, None)

    def apply(
        self,
        base: MotorFrame,
        *,
        now: float | None = None,
        permitted: bool = True,
    ) -> MotorFrame:
        """Return the base frame or a reusable frame with override applied."""
        timestamp = time.monotonic() if now is None else float(now)
        if not math.isfinite(timestamp):
            raise ValueError("motor override apply time must be finite")
        if not isinstance(permitted, bool):
            raise TypeError("motor override permitted flag must be a bool")

        self._ensure_output(base.layout)
        generation, submitted = self._mailbox
        if generation != self._seen_generation:
            self._seen_generation = generation
            self._command = submitted
            self._release_active = False

        command = self._command
        fresh = command is not None and (
            self.timeout_sec == 0.0
            or timestamp - command.received_at <= self.timeout_sec
        )
        if fresh and permitted:
            assert command is not None
            if command.robot_names != base.layout.names:
                raise ValueError(
                    "motor override robot layout changed after command submission"
                )
            output = self._copy_base(base)
            output.qpos[command.indices] = command.qpos
            output.kp[command.indices] = command.kp
            output.kd[command.indices] = command.kd
            output.vel[command.indices] = command.vel
            output.torque[command.indices] = command.torque
            self._was_applied = True
            self._last_applied_command = command
            self._release_active = False
            return output

        if not permitted:
            self._was_applied = False
            self._release_active = False
            return base

        if self._was_applied:
            self._start_release(timestamp)
            self._was_applied = False

        if not self._release_active:
            return base

        elapsed = max(0.0, timestamp - self._release_started_at)
        if self.release_blend_sec == 0.0 or elapsed >= self.release_blend_sec:
            self._release_active = False
            self._last_applied_command = None
            return base

        alpha = elapsed / self.release_blend_sec
        output = self._copy_base(base)
        for offset, index in enumerate(self._release_indices):
            output.qpos[index] = self._release_qpos[offset] + alpha * (
                base.qpos[index] - self._release_qpos[offset]
            )
            output.kp[index] = self._release_kp[offset] + alpha * (
                base.kp[index] - self._release_kp[offset]
            )
            output.kd[index] = self._release_kd[offset] + alpha * (
                base.kd[index] - self._release_kd[offset]
            )
            output.vel[index] = self._release_vel[offset] + alpha * (
                base.vel[index] - self._release_vel[offset]
            )
            output.torque[index] = self._release_torque[offset] + alpha * (
                base.torque[index] - self._release_torque[offset]
            )
        return output

    def _ensure_output(self, layout: JointLayout) -> None:
        output = self._output
        if output is None:
            # Imported lazily to keep the semantic joints package independent
            # from mod_api during package initialization.
            from bxi_example_py_elf3.framework.mod_api.frame import MotorFrame

            self._output = MotorFrame.empty(layout)
            return
        if output.layout is not layout and output.layout.names != layout.names:
            raise ValueError("motor override output robot layout changed after startup")

    def _copy_base(self, base: MotorFrame) -> MotorFrame:
        output = self._output
        assert output is not None
        output.update(
            base.qpos,
            base.kp,
            base.kd,
            vel=base.vel,
            torque=base.torque,
        )
        return output

    def _start_release(self, timestamp: float) -> None:
        command = self._last_applied_command
        output = self._output
        if command is None or output is None or self.release_blend_sec == 0.0:
            self._release_active = False
            self._last_applied_command = None
            return
        self._release_indices = command.indices
        self._release_qpos = output.qpos[command.indices].copy()
        self._release_kp = output.kp[command.indices].copy()
        self._release_kd = output.kd[command.indices].copy()
        self._release_vel = output.vel[command.indices].copy()
        self._release_torque = output.torque[command.indices].copy()
        self._release_started_at = timestamp
        self._release_active = True


__all__ = ["NamedJointCommandOverride"]
