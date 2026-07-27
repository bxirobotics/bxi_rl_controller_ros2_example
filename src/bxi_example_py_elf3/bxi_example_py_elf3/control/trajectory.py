"""Joint trajectory loading, validation and deterministic playback."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elf3 import DOF_NUM, JOINT_NAMES, JOINT_POSITION_MAX, JOINT_POSITION_MIN


def minimum_jerk_progress(progress):
    """Clamp progress to [0, 1] and apply a zero-slope quintic blend."""
    value = min(max(float(progress), 0.0), 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    frame_count: int
    max_step_delta_rad: float
    max_step_joint: str
    max_step_velocity_rad_s: float
    loop_delta_rad: float
    loop_delta_joint: str
    loop_velocity_rad_s: float


class JointTrajectory:
    """Validated in-memory trajectory with pause/resume-friendly indexing."""

    def __init__(self, positions, source_path, control_rate_hz):
        matrix = np.asarray(positions, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != DOF_NUM:
            raise ValueError(
                "trajectory must have shape (frames, %d); got %s"
                % (DOF_NUM, matrix.shape)
            )
        if matrix.shape[0] == 0:
            raise ValueError("trajectory is empty")
        if not np.all(np.isfinite(matrix)):
            row, column = np.argwhere(~np.isfinite(matrix))[0]
            raise ValueError(
                "trajectory contains a non-finite value at frame %d, joint %s"
                % (int(row) + 1, JOINT_NAMES[int(column)])
            )
        violations = np.argwhere(
            (matrix < JOINT_POSITION_MIN) | (matrix > JOINT_POSITION_MAX)
        )
        if violations.size:
            row, column = violations[0]
            raise ValueError(
                "trajectory frame %d commands %s=%.6f rad outside "
                "[%.6f, %.6f]"
                % (
                    int(row) + 1,
                    JOINT_NAMES[int(column)],
                    matrix[row, column],
                    JOINT_POSITION_MIN[column],
                    JOINT_POSITION_MAX[column],
                )
            )
        if float(control_rate_hz) <= 0.0:
            raise ValueError("control_rate_hz must be > 0")
        self.positions = matrix
        self.source_path = str(source_path)
        self.control_rate_hz = float(control_rate_hz)
        self.index = 0

    def next(self):
        position = self.positions[self.index].copy()
        self.index = (self.index + 1) % len(self.positions)
        return position

    def reset(self):
        self.index = 0

    @property
    def next_is_first_frame(self):
        """Whether the next sample starts or wraps to a new trajectory cycle."""
        return self.index == 0

    def diagnostics(self):
        if len(self.positions) == 1:
            deltas = np.zeros((1, DOF_NUM), dtype=np.float64)
        else:
            deltas = np.diff(self.positions, axis=0)
        absolute = np.abs(deltas)
        step_flat = int(np.argmax(absolute))
        _, step_joint = np.unravel_index(step_flat, absolute.shape)
        max_step = float(absolute.flat[step_flat])

        loop = np.abs(self.positions[0] - self.positions[-1])
        loop_joint = int(np.argmax(loop))
        loop_delta = float(loop[loop_joint])
        return TrajectoryDiagnostics(
            frame_count=len(self.positions),
            max_step_delta_rad=max_step,
            max_step_joint=JOINT_NAMES[int(step_joint)],
            max_step_velocity_rad_s=max_step * self.control_rate_hz,
            loop_delta_rad=loop_delta,
            loop_delta_joint=JOINT_NAMES[loop_joint],
            loop_velocity_rad_s=loop_delta * self.control_rate_hz,
        )


def _parse_line(line, line_number):
    clean = line.strip()
    if not clean:
        return None
    if clean.startswith("[") and clean.endswith("]"):
        clean = clean[1:-1].strip()
    tokens = clean.split(",") if "," in clean else clean.split()
    tokens = [token.strip() for token in tokens if token.strip()]
    if len(tokens) != DOF_NUM:
        raise ValueError(
            "trajectory line %d contains %d values; expected %d"
            % (line_number, len(tokens), DOF_NUM)
        )
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(
            "cannot parse trajectory line %d: %s" % (line_number, exc)
        ) from exc


def load_joint_trajectory(path, control_rate_hz):
    source = Path(path).expanduser()
    rows = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                parsed = _parse_line(line, line_number)
                if parsed is not None:
                    rows.append(parsed)
    except OSError as exc:
        raise ValueError("cannot read trajectory %s: %s" % (source, exc)) from exc
    return JointTrajectory(rows, source.resolve(), control_rate_hz)
