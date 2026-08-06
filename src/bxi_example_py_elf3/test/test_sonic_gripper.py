import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic" / "gripper.py"
)
SPEC = importlib.util.spec_from_file_location("sonic_gripper", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
GRIPPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GRIPPER
SPEC.loader.exec_module(GRIPPER)


def _reply_payload(motor_id, position, velocity, torque, mos=35, motor=36):
    p_int = GRIPPER.BxiMotor.float_to_uint(position, -12.5, 12.5, 16)
    v_int = GRIPPER.BxiMotor.float_to_uint(velocity, -45.0, 45.0, 12)
    t_int = GRIPPER.BxiMotor.float_to_uint(torque, -40.0, 40.0, 12)
    return [
        motor_id,
        (p_int >> 8) & 0xFF,
        p_int & 0xFF,
        (v_int >> 4) & 0xFF,
        ((v_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),
        t_int & 0xFF,
        mos,
        motor,
    ]


def _settings():
    return GRIPPER.CalibrationSettings(
        speed_rad_s=2.0,
        contact_torque=0.5,
        abort_torque=5.0,
        contact_confirm_s=0.04,
        stopped_velocity_rad_s=0.05,
        tracking_error_rad=0.05,
        limit_margin_rad=0.1,
        minimum_span_rad=1.0,
        maximum_search_travel_rad=5.0,
        response_timeout_s=0.2,
        feedback_timeout_s=0.1,
        phase_timeout_s=5.0,
        maximum_mos_temperature_c=80,
        maximum_motor_temperature_c=80,
        settle_time_s=0.04,
    )


def test_motor_response_frame_decodes_all_feedback_fields():
    payload = _reply_payload(1, 1.25, -2.5, 3.75, mos=42, motor=43)

    feedback = GRIPPER.BxiMotor.unpack_feedback(payload, received_at=7.0)

    assert feedback.motor_id == 1
    assert feedback.position == pytest.approx(1.25, abs=4.0e-4)
    assert feedback.velocity == pytest.approx(-2.5, abs=2.3e-2)
    assert feedback.torque == pytest.approx(3.75, abs=2.0e-2)
    assert feedback.mos_temperature_c == 42
    assert feedback.motor_temperature_c == 43
    assert feedback.received_at == 7.0


def test_calibrator_fails_when_motor_never_responds():
    calibrator = GRIPPER.GripperCalibrator("left", _settings())
    calibrator.reset(10.0)

    assert calibrator.update(None, 10.19, 0.02) is None
    calibrator.update(None, 10.21, 0.02)

    assert calibrator.failed
    assert "offline" in calibrator.failure_reason
    assert "no response" in calibrator.failure_reason


def test_calibrator_fails_when_feedback_stream_goes_stale():
    calibrator = GRIPPER.GripperCalibrator("right", _settings())
    calibrator.reset(0.0)
    feedback = GRIPPER.MotorFeedback(1, 0.0, 0.0, 0.0, 30, 31, 0.0)
    calibrator.update(feedback, 0.0, 0.02)

    calibrator.update(feedback, 0.11, 0.02)

    assert calibrator.failed
    assert "feedback timed out" in calibrator.failure_reason


def test_calibrator_finds_both_limits_backs_off_and_returns_open():
    settings = _settings()
    calibrator = GRIPPER.GripperCalibrator("left", settings)
    calibrator.reset(0.0)
    closed_limit = -0.5
    open_limit = 2.0
    actual = 0.5
    previous_actual = actual
    target = actual
    dt = 0.02

    for cycle in range(1000):
        now = cycle * dt
        velocity = (actual - previous_actual) / dt
        penetration = target - actual
        torque = penetration * 10.0 if abs(penetration) > 1.0e-9 else 0.0
        feedback = GRIPPER.MotorFeedback(
            motor_id=1,
            position=actual,
            velocity=velocity,
            torque=torque,
            mos_temperature_c=35,
            motor_temperature_c=36,
            received_at=now,
        )
        command = calibrator.update(feedback, now, dt)
        if calibrator.failed:
            pytest.fail(calibrator.failure_reason)
        if calibrator.ready:
            break
        if command is not None:
            target = command
        previous_actual = actual
        maximum_step = 4.0 * dt
        motion = max(min(target - actual, maximum_step), -maximum_step)
        actual = min(max(actual + motion, closed_limit), open_limit)
    else:
        pytest.fail("calibration did not finish")

    assert calibrator.open_limit == pytest.approx(open_limit, abs=1.0e-6)
    assert calibrator.closed_limit == pytest.approx(closed_limit, abs=1.0e-6)
    assert calibrator.open_position == pytest.approx(open_limit - 0.1)
    assert calibrator.closed_position == pytest.approx(closed_limit + 0.1)
    assert calibrator.target_position == pytest.approx(calibrator.open_position)
