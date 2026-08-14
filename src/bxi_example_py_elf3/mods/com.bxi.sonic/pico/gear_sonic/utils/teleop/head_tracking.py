"""Map PICO head orientation to ELF3's two neck joints."""

from __future__ import annotations

import math

import numpy as np


HEAD_JOINT_NAMES = ("head_y_joint", "head_z_joint")


def _normalize_quaternion_wxyz(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("head tracking quaternion must contain four finite WXYZ values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("head tracking quaternion must not have zero length")
    return quaternion / norm


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def _relative_head_roll_pitch(
    spine_quaternion_wxyz: object,
    head_quaternion_wxyz: object,
) -> tuple[float, float]:
    """Return MoCapLab-compatible XYZ roll/pitch for Head relative to Spine3."""

    spine = _normalize_quaternion_wxyz(spine_quaternion_wxyz)
    head = _normalize_quaternion_wxyz(head_quaternion_wxyz)
    inverse_spine = spine.copy()
    inverse_spine[1:] *= -1.0
    relative = _quaternion_multiply_wxyz(inverse_spine, head)
    relative = _normalize_quaternion_wxyz(relative)
    w, x, y, z = relative

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    return roll, pitch


def _wrapped_delta(value: float, origin: float) -> float:
    return math.atan2(math.sin(value - origin), math.cos(value - origin))


class PicoHeadMapper:
    """Center and map relative PICO head orientation into ``(pitch, yaw)``."""

    def __init__(
        self,
        *,
        pitch_limit_rad: float = 0.785,
        yaw_limit_rad: float = 1.57,
    ) -> None:
        if not math.isfinite(pitch_limit_rad) or pitch_limit_rad <= 0.0:
            raise ValueError("head pitch limit must be positive and finite")
        if not math.isfinite(yaw_limit_rad) or yaw_limit_rad <= 0.0:
            raise ValueError("head yaw limit must be positive and finite")
        self.pitch_limit_rad = float(pitch_limit_rad)
        self.yaw_limit_rad = float(yaw_limit_rad)
        self._center: tuple[float, float] | None = None

    def reset(self) -> None:
        self._center = None

    def update(
        self,
        spine_quaternion_wxyz: object,
        head_quaternion_wxyz: object,
    ) -> np.ndarray:
        roll, pitch = _relative_head_roll_pitch(
            spine_quaternion_wxyz,
            head_quaternion_wxyz,
        )
        if self._center is None:
            self._center = (roll, pitch)
        center_roll, center_pitch = self._center

        # Match com.bxi.pico_gmr_motion exactly:
        # robot head_y (pitch) <- negative relative XYZ roll
        # robot head_z (yaw)   <- relative XYZ pitch
        robot_pitch = -_wrapped_delta(roll, center_roll)
        robot_yaw = _wrapped_delta(pitch, center_pitch)
        return np.asarray(
            (
                np.clip(robot_pitch, -self.pitch_limit_rad, self.pitch_limit_rad),
                np.clip(robot_yaw, -self.yaw_limit_rad, self.yaw_limit_rad),
            ),
            dtype=np.float32,
        )


__all__ = ["HEAD_JOINT_NAMES", "PicoHeadMapper"]
