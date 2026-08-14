#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence
import communication.msg as bxiMsg


@dataclass
class JointControl:
    """Motor joint command data."""

    p_des: float = 0.0
    v_des: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    t_ff: float = 0.0


@dataclass(frozen=True)
class MotorFeedback:
    """Decoded MIT-mode motor feedback received from one physical CAN bus."""

    motor_id: int
    position: float
    velocity: float
    torque: float
    mos_temperature_c: int
    motor_temperature_c: int
    received_at: float


class CalibrationPhase(str, Enum):
    WAITING_FEEDBACK = "waiting_feedback"
    SETTLING = "settling"
    SEEKING_OPEN = "seeking_open"
    BACKING_OFF_OPEN = "backing_off_open"
    SEEKING_CLOSED = "seeking_closed"
    BACKING_OFF_CLOSED = "backing_off_closed"
    RETURNING_OPEN = "returning_open"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class CalibrationSettings:
    speed_rad_s: float = 0.2
    contact_torque: float = 2.0
    abort_torque: float = 8.0
    contact_confirm_s: float = 0.25
    stopped_velocity_rad_s: float = 0.1
    tracking_error_rad: float = 0.08
    limit_margin_rad: float = 0.15
    minimum_span_rad: float = 1.0
    maximum_search_travel_rad: float = 7.0
    response_timeout_s: float = 1.0
    feedback_timeout_s: float = 0.3
    phase_timeout_s: float = 45.0
    maximum_mos_temperature_c: int = 80
    maximum_motor_temperature_c: int = 80
    settle_time_s: float = 0.5
    position_min_rad: float = -12.5
    position_max_rad: float = 12.5

    def validate(self) -> None:
        finite = (
            self.speed_rad_s,
            self.contact_torque,
            self.abort_torque,
            self.contact_confirm_s,
            self.stopped_velocity_rad_s,
            self.tracking_error_rad,
            self.limit_margin_rad,
            self.minimum_span_rad,
            self.maximum_search_travel_rad,
            self.response_timeout_s,
            self.feedback_timeout_s,
            self.phase_timeout_s,
            self.settle_time_s,
            self.position_min_rad,
            self.position_max_rad,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("gripper calibration settings must be finite")
        positive = finite[:-2]
        if any(value <= 0.0 for value in positive):
            raise ValueError("gripper calibration timing and limits must be positive")
        if self.position_max_rad <= self.position_min_rad:
            raise ValueError("gripper calibration position range is invalid")
        if self.abort_torque <= self.contact_torque:
            raise ValueError("gripper abort torque must exceed contact torque")
        if self.feedback_timeout_s > self.response_timeout_s:
            raise ValueError(
                "gripper feedback timeout must not exceed response timeout"
            )
        if (
            min(
                self.maximum_mos_temperature_c,
                self.maximum_motor_temperature_c,
            )
            <= 0
        ):
            raise ValueError("gripper temperature limits must be positive")


class GripperCalibrator:
    """Non-blocking hard-stop calibration for one gripper motor."""

    _TORQUE_FILTER_ALPHA = 0.2
    _BIAS_FILTER_ALPHA = 0.02

    def __init__(self, label: str, settings: CalibrationSettings) -> None:
        settings.validate()
        self.label = str(label)
        self.settings = settings
        self.phase = CalibrationPhase.WAITING_FEEDBACK
        self.target_position: Optional[float] = None
        self.open_limit: Optional[float] = None
        self.closed_limit: Optional[float] = None
        self.open_position: Optional[float] = None
        self.closed_position: Optional[float] = None
        self.failure_reason: Optional[str] = None
        self._started_at = 0.0
        self._phase_started_at = 0.0
        self._search_started_position = 0.0
        self._filtered_torque: Optional[float] = None
        self._torque_bias = 0.0
        self._contact_since: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.phase is CalibrationPhase.READY

    @property
    def failed(self) -> bool:
        return self.phase is CalibrationPhase.FAILED

    def reset(self, now: float) -> None:
        self.phase = CalibrationPhase.WAITING_FEEDBACK
        self.target_position = None
        self.open_limit = None
        self.closed_limit = None
        self.open_position = None
        self.closed_position = None
        self.failure_reason = None
        self._started_at = float(now)
        self._phase_started_at = float(now)
        self._search_started_position = 0.0
        self._filtered_torque = None
        self._torque_bias = 0.0
        self._contact_since = None

    def fail(self, reason: str) -> None:
        if not self.failed:
            self.failure_reason = str(reason)
            self.phase = CalibrationPhase.FAILED

    def _set_phase(
        self,
        phase: CalibrationPhase,
        now: float,
        feedback: MotorFeedback,
    ) -> None:
        self.phase = phase
        self._phase_started_at = float(now)
        self._search_started_position = float(feedback.position)
        self._contact_since = None
        if self._filtered_torque is not None:
            self._torque_bias = self._filtered_torque

    def _update_torque_filter(self, torque: float) -> None:
        if self._filtered_torque is None:
            self._filtered_torque = float(torque)
        else:
            alpha = self._TORQUE_FILTER_ALPHA
            self._filtered_torque += alpha * (float(torque) - self._filtered_torque)

    def _update_free_torque_bias(self, feedback: MotorFeedback) -> None:
        assert self.target_position is not None
        if abs(self.target_position - feedback.position) <= max(
            self.settings.tracking_error_rad,
            0.15,
        ):
            assert self._filtered_torque is not None
            alpha = self._BIAS_FILTER_ALPHA
            self._torque_bias += alpha * (self._filtered_torque - self._torque_bias)

    def _contact_confirmed(self, feedback: MotorFeedback, now: float) -> bool:
        assert self.target_position is not None
        assert self._filtered_torque is not None
        candidate = (
            abs(self._filtered_torque - self._torque_bias)
            >= self.settings.contact_torque
            and abs(feedback.velocity) <= self.settings.stopped_velocity_rad_s
            and abs(self.target_position - feedback.position)
            >= self.settings.tracking_error_rad
        )
        if not candidate:
            self._contact_since = None
            return False
        if self._contact_since is None:
            self._contact_since = float(now)
            return False
        return now - self._contact_since >= self.settings.contact_confirm_s

    def _settled(self, feedback: MotorFeedback, now: float) -> bool:
        assert self.target_position is not None
        return (
            now - self._phase_started_at >= self.settings.settle_time_s
            and abs(self.target_position - feedback.position)
            <= self.settings.tracking_error_rad
            and abs(feedback.velocity) <= self.settings.stopped_velocity_rad_s
        )

    def _backoff_complete(self, feedback: MotorFeedback, now: float) -> bool:
        """Accept safe-direction overshoot while backing away from a hard stop."""

        assert self.target_position is not None
        if (
            now - self._phase_started_at < self.settings.settle_time_s
            or abs(feedback.velocity) > self.settings.stopped_velocity_rad_s
        ):
            return False
        tolerance = self.settings.tracking_error_rad
        if self.phase is CalibrationPhase.BACKING_OFF_OPEN:
            return feedback.position <= self.target_position + tolerance
        if self.phase is CalibrationPhase.BACKING_OFF_CLOSED:
            return feedback.position >= self.target_position - tolerance
        raise RuntimeError(
            f"backoff completion checked in invalid phase {self.phase.value}"
        )

    def _check_backoff_timeout(self, now: float) -> bool:
        if now - self._phase_started_at >= self.settings.phase_timeout_s:
            self.fail(f"{self.label} gripper backoff phase timed out")
        return self.failed

    def _check_common_faults(
        self,
        feedback: Optional[MotorFeedback],
        now: float,
    ) -> bool:
        if feedback is None:
            if now - self._started_at >= self.settings.response_timeout_s:
                self.fail(f"{self.label} gripper motor is offline: no response frame")
            return self.failed
        if now - feedback.received_at > self.settings.feedback_timeout_s:
            self.fail(f"{self.label} gripper motor is offline: feedback timed out")
        elif feedback.mos_temperature_c >= self.settings.maximum_mos_temperature_c:
            self.fail(
                f"{self.label} gripper MOS temperature is too high: "
                f"{feedback.mos_temperature_c} C"
            )
        elif feedback.motor_temperature_c >= self.settings.maximum_motor_temperature_c:
            self.fail(
                f"{self.label} gripper motor temperature is too high: "
                f"{feedback.motor_temperature_c} C"
            )
        elif (
            self._filtered_torque is not None
            and abs(self._filtered_torque - self._torque_bias)
            >= self.settings.abort_torque
        ):
            self.fail(f"{self.label} gripper torque exceeded the abort limit")
        return self.failed

    def _check_seek_bounds(self, feedback: MotorFeedback, now: float) -> bool:
        assert self.target_position is not None
        if now - self._phase_started_at >= self.settings.phase_timeout_s:
            self.fail(f"{self.label} gripper calibration phase timed out")
        elif (
            abs(feedback.position - self._search_started_position)
            >= self.settings.maximum_search_travel_rad
        ):
            self.fail(f"{self.label} gripper exceeded maximum calibration travel")
        elif not (
            self.settings.position_min_rad
            < self.target_position
            < self.settings.position_max_rad
        ):
            self.fail(f"{self.label} gripper reached the protocol position limit")
        return self.failed

    def update(
        self,
        feedback: Optional[MotorFeedback],
        now: float,
        dt: float,
    ) -> Optional[float]:
        """Advance calibration once and return this cycle's position target."""

        if self.ready or self.failed:
            return self.target_position
        if feedback is not None:
            self._update_torque_filter(feedback.torque)
        if self._check_common_faults(feedback, now):
            return self.target_position
        if feedback is None:
            return None

        if self.phase is CalibrationPhase.WAITING_FEEDBACK:
            self.target_position = float(feedback.position)
            self._torque_bias = float(feedback.torque)
            self._set_phase(CalibrationPhase.SETTLING, now, feedback)
            return self.target_position

        assert self.target_position is not None
        step = self.settings.speed_rad_s * max(float(dt), 0.0)

        if self.phase is CalibrationPhase.SETTLING:
            if now - self._phase_started_at >= self.settings.settle_time_s:
                self._set_phase(CalibrationPhase.SEEKING_OPEN, now, feedback)
            return self.target_position

        if self.phase is CalibrationPhase.SEEKING_OPEN:
            self.target_position += step
            self._update_free_torque_bias(feedback)
            if self._contact_confirmed(feedback, now):
                self.open_limit = float(feedback.position)
                self.open_position = self.open_limit - self.settings.limit_margin_rad
                self.target_position = self.open_position
                self._set_phase(CalibrationPhase.BACKING_OFF_OPEN, now, feedback)
            else:
                self._check_seek_bounds(feedback, now)
            return self.target_position

        if self.phase is CalibrationPhase.BACKING_OFF_OPEN:
            if self._backoff_complete(feedback, now):
                self._set_phase(CalibrationPhase.SEEKING_CLOSED, now, feedback)
            else:
                self._check_backoff_timeout(now)
            return self.target_position

        if self.phase is CalibrationPhase.SEEKING_CLOSED:
            self.target_position -= step
            self._update_free_torque_bias(feedback)
            if self._contact_confirmed(feedback, now):
                self.closed_limit = float(feedback.position)
                self.closed_position = (
                    self.closed_limit + self.settings.limit_margin_rad
                )
                if (
                    self.open_position is None
                    or self.open_position - self.closed_position
                    < self.settings.minimum_span_rad
                ):
                    self.fail(f"{self.label} gripper calibrated span is too small")
                    return self.target_position
                self.target_position = self.closed_position
                self._set_phase(CalibrationPhase.BACKING_OFF_CLOSED, now, feedback)
            else:
                self._check_seek_bounds(feedback, now)
            return self.target_position

        if self.phase is CalibrationPhase.BACKING_OFF_CLOSED:
            if self._backoff_complete(feedback, now):
                self._set_phase(CalibrationPhase.RETURNING_OPEN, now, feedback)
            else:
                self._check_backoff_timeout(now)
            return self.target_position

        if self.phase is CalibrationPhase.RETURNING_OPEN:
            assert self.open_position is not None
            self.target_position = min(
                self.target_position + step,
                self.open_position,
            )
            if self.target_position >= self.open_position and self._settled(
                feedback, now
            ):
                self.phase = CalibrationPhase.READY
            elif now - self._phase_started_at >= self.settings.phase_timeout_s:
                self.fail(f"{self.label} gripper return-to-open phase timed out")
            return self.target_position

        self.fail(f"{self.label} gripper entered an invalid calibration phase")
        return self.target_position


class BxiMotor:
    @staticmethod
    def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
        """Convert float value to unsigned integer with given bit width."""
        if bits <= 0:
            raise ValueError("bits must be positive")

        span = x_max - x_min
        if span <= 0:
            raise ValueError("x_max must be greater than x_min")

        # Clamp
        x = max(min(x, x_max), x_min)

        max_val = (1 << bits) - 1

        # Handle boundaries explicitly to avoid floating point errors
        if x == x_min:
            return 0
        if x == x_max:
            return max_val

        # Linear mapping with rounding
        result = (x - x_min) * max_val / span
        return int(round(result))

    @staticmethod
    def uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
        """Decode the unsigned fixed-width value used by the MIT protocol."""
        if bits <= 0:
            raise ValueError("bits must be positive")
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")
        maximum = (1 << bits) - 1
        if x < 0 or x > maximum:
            raise ValueError(f"value does not fit in {bits} bits")
        return float(x) * (x_max - x_min) / maximum + x_min

    @staticmethod
    def unpack_feedback(
        data: Sequence[int],
        *,
        received_at: float,
        p_range: tuple[float, float] = (-12.5, 12.5),
        v_range: tuple[float, float] = (-45.0, 45.0),
        t_range: tuple[float, float] = (-40.0, 40.0),
    ) -> MotorFeedback:
        """Decode the documented eight-byte motor response payload."""
        if len(data) < 8:
            raise ValueError("motor feedback payload must contain eight bytes")
        values = [int(value) for value in data[:8]]
        if any(value < 0 or value > 0xFF for value in values):
            raise ValueError("motor feedback payload contains a non-byte value")
        position_raw = (values[1] << 8) | values[2]
        velocity_raw = (values[3] << 4) | (values[4] >> 4)
        torque_raw = ((values[4] & 0x0F) << 8) | values[5]
        return MotorFeedback(
            motor_id=values[0],
            position=BxiMotor.uint_to_float(position_raw, *p_range, 16),
            velocity=BxiMotor.uint_to_float(velocity_raw, *v_range, 12),
            torque=BxiMotor.uint_to_float(torque_raw, *t_range, 12),
            mos_temperature_c=values[6],
            motor_temperature_c=values[7],
            received_at=float(received_at),
        )

    @staticmethod
    def fmaxf(x: float, y: float) -> float:
        return max(x, y)

    @staticmethod
    def fminf(x: float, y: float) -> float:
        return min(x, y)

    @staticmethod
    def fmaxf3(x: float, y: float, z: float) -> float:
        return max(x, y, z)

    @staticmethod
    def fminf3(x: float, y: float, z: float) -> float:
        return min(x, y, z)

    @staticmethod
    def limit_norm(x: float, y: float, limit: float) -> tuple[float, float]:
        """Scale vector (x, y) length to be <= limit."""
        norm = math.sqrt(x * x + y * y)
        if norm > limit and norm > 0:
            x = x * limit / norm
            y = y * limit / norm
        return x, y

    @staticmethod
    def zero() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]
        return data

    @staticmethod
    def enter_motor_mode() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
        return data

    @staticmethod
    def exit_motor_mode() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
        return data

    @staticmethod
    def pack_cmd(
        joint: JointControl,
        p_range: tuple[float, float],
        v_range: tuple[float, float],
        t_range: tuple[float, float],
        kp_range: tuple[float, float],
        kd_range: tuple[float, float],
    ) -> list[int]:
        """Pack motor command into 8 bytes.

        Equivalent to the C function:
        pack_cmd(uint8_t *data, joint_control *joint, ...)
        """
        p_min, p_max = p_range
        v_min, v_max = v_range
        t_min, t_max = t_range
        kp_min, kp_max = kp_range
        kd_min, kd_max = kd_range

        p_des = min(max(p_min, joint.p_des), p_max)
        v_des = min(max(v_min, joint.v_des), v_max)
        kp = min(max(kp_min, joint.kp), kp_max)
        kd = min(max(kd_min, joint.kd), kd_max)
        t_ff = min(max(t_min, joint.t_ff), t_max)

        p_int = BxiMotor.float_to_uint(p_des, p_min, p_max, 16)
        v_int = BxiMotor.float_to_uint(v_des, v_min, v_max, 12)
        kp_int = BxiMotor.float_to_uint(kp, kp_min, kp_max, 12)
        kd_int = BxiMotor.float_to_uint(kd, kd_min, kd_max, 12)
        t_int = BxiMotor.float_to_uint(t_ff, t_min, t_max, 12)

        data = [0] * 8
        data[0] = (p_int >> 8) & 0xFF
        data[1] = p_int & 0xFF
        data[2] = (v_int >> 4) & 0xFF
        data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
        data[4] = kp_int & 0xFF
        data[5] = (kd_int >> 4) & 0xFF
        data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
        data[7] = t_int & 0xFF
        return data

    @staticmethod
    def build_motor_packet(bus, canid, data: list[int]):
        packet_type = getattr(
            bxiMsg, "CANFDPacket", getattr(bxiMsg, "CanfdPacket", None)
        )
        if packet_type is None:
            raise RuntimeError("communication.msg.CANFDPacket is unavailable")
        packet = packet_type()
        packet.bus = int(bus)
        packet.frame.can_id = int(canid)
        packet.frame.flags = int(0x01 | 0x04)
        packet.frame.len = len(data)
        values = list(data)
        try:
            packet.frame.data = values
            return packet
        except Exception:
            pass

        padded = values + [0] * (64 - len(values))
        try:
            packet.frame.data = padded
            return packet
        except Exception:
            pass

        for index, value in enumerate(values):
            packet.frame.data[index] = value
        return packet
