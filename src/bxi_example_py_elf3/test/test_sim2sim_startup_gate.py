from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import bxi_example_py_elf3.bxi_example_demo as demo_module
from bxi_example_py_elf3.bxi_example_demo import BxiExample


def _motor_context(*, step):
    frame = tuple(np.ones(3, dtype=np.float32) * value for value in (1, 2, 3))
    calls = []
    ctx = SimpleNamespace(
        step=step,
        motor_target=frame,
        pos_last=np.zeros(3, dtype=np.float32),
        kp_last=np.zeros(3, dtype=np.float32),
        kd_last=np.zeros(3, dtype=np.float32),
        check_control_frame_rate=lambda: calls.append("rate"),
        send_to_motor=lambda *target: calls.append(target),
    )
    return ctx, frame, calls


def test_startup_suspension_never_publishes_motor_target():
    ctx, _, calls = _motor_context(step=1)

    assert not BxiExample.publish_motor_target_if_released(ctx)

    assert calls == []
    np.testing.assert_array_equal(ctx.pos_last, np.zeros(3, dtype=np.float32))


def test_motor_target_is_published_after_release():
    ctx, frame, calls = _motor_context(step=2)

    assert BxiExample.publish_motor_target_if_released(ctx)

    assert calls[0] == "rate"
    for actual, expected in zip(calls[1], frame):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(ctx.pos_last, frame[0])
    np.testing.assert_array_equal(ctx.kp_last, frame[1])
    np.testing.assert_array_equal(ctx.kd_last, frame[2])


def test_step_one_processes_state_events_without_running_control_output():
    calls = []

    def update_state_machine(*args):
        calls.append(("update", args[1]))
        ctx.motor_target = "transition-target"
        return False

    ctx = SimpleNamespace(
        step=1,
        loop_count=10,
        startup_release_delay=1.0,
        dt=0.02,
        state=0,
        state_name_by_id={0: "zero_torque"},
        startup_release_allowed_states={"normal"},
        lock_in=nullcontext(),
        qpos=np.zeros(3),
        qvel=np.zeros(3),
        quat_xyzw=np.zeros(4),
        quat_wxyz=np.zeros(4),
        omega=np.zeros(3),
        raw_cmd_vel=np.zeros(3),
        current_q=np.zeros(3),
        current_dq=np.zeros(3),
        current_quat_xyzw=np.zeros(4),
        current_quat_wxyz=np.zeros(4),
        current_omega=np.zeros(3),
        current_raw_cmd_vel=np.zeros(3),
        current_cmd_vel=np.zeros(3),
        pending_remote_events=deque(["pd_brake_event"]),
        motor_target=None,
        state_machine=SimpleNamespace(
            update=update_state_machine,
            update_current_state=lambda *_: calls.append("update_current_state"),
            current_state_id=0,
        ),
        robot_reset=lambda *_: calls.append("robot_reset"),
        reset_control_rate_monitor=lambda: calls.append("reset_rate"),
        check_hot_reload=lambda *_: calls.append("hot_reload"),
        publish_motor_target_if_released=lambda: calls.append("publish_motor"),
        publish_state_machine_info_if_due=lambda events: calls.append(("info", events)),
    )

    BxiExample.timer_callback(ctx)

    assert ctx.step == 1
    assert ctx.loop_count == 11
    assert calls == [
        ("update", ["pd_brake_event"]),
        ("info", ["pd_brake_event"]),
    ]
    assert ctx.motor_target is None
    assert list(ctx.pending_remote_events) == []


def test_step_one_release_does_not_run_control_until_next_tick():
    calls = []
    ctx = SimpleNamespace(
        step=1,
        loop_count=50,
        startup_release_delay=1.0,
        dt=0.02,
        state=0,
        state_name_by_id={0: "normal"},
        startup_release_allowed_states={"normal"},
        state_machine=SimpleNamespace(
            update=lambda *_: calls.append("update"),
            update_current_state=lambda *_: calls.append("update_current_state"),
            current_state_id=0,
        ),
        robot_reset=lambda *args: calls.append(("robot_reset", args)),
        reset_control_rate_monitor=lambda: calls.append("reset_rate"),
        check_hot_reload=lambda *_: calls.append("hot_reload"),
        publish_motor_target_if_released=lambda: calls.append("publish_motor"),
        publish_state_machine_info_if_due=lambda events: calls.append(("info", events)),
    )

    BxiExample.timer_callback(ctx)

    assert ctx.step == 2
    assert ctx.loop_count == 0
    assert calls == [("robot_reset", (2, True)), "reset_rate"]


def test_startup_wait_is_logged_once_per_state_and_allowlist(capsys):
    ctx = SimpleNamespace(
        startup_release_allowed_states={"normal", "pd_brake"},
        startup_release_wait_signature=None,
    )

    assert BxiExample.log_startup_release_wait(ctx, "zero_torque")
    assert not BxiExample.log_startup_release_wait(ctx, "zero_torque")
    assert BxiExample.log_startup_release_wait(ctx, "idle")

    output = capsys.readouterr().out
    assert output.count("startup release waiting for allowed state") == 2
    assert output.count("current=zero_torque") == 1
    assert output.count("current=idle") == 1
    assert "allowed=['normal', 'pd_brake']" in output


def _control_rate_context(**overrides):
    values = {
        "control_period": 0.02,
        "control_rate_tolerance": 0.005,
        "control_rate_report_period": 1.0,
        "control_rate_warning_log_period": 5.0,
        "control_rate_warn_min_hz": 45.0,
        "control_rate_warn_max_delay": 0.05,
        "control_rate_warn_late_ratio": 0.10,
        "last_control_frame_time": None,
        "control_rate_report_start_time": None,
        "control_rate_frame_count": 0,
        "control_rate_late_count": 0,
        "control_rate_delay_sum": 0.0,
        "control_rate_delay_max": 0.0,
        "control_rate_warning_active": False,
        "last_control_rate_warning_log_time": None,
        "state": 2,
        "state_name_by_id": {2: "pd_brake"},
    }
    values.update(overrides)
    ctx = SimpleNamespace(**values)
    ctx.print_control_rate_summary = lambda *args: (
        BxiExample.print_control_rate_summary(ctx, *args)
    )
    return ctx


def test_healthy_control_rate_is_silent(monkeypatch, capsys):
    times = iter(index * 0.02 for index in range(1501))
    monkeypatch.setattr(demo_module.time, "perf_counter", lambda: next(times))
    ctx = _control_rate_context()

    for _ in range(1501):
        BxiExample.check_control_frame_rate(ctx)

    output = capsys.readouterr().out
    assert output == ""


def test_abnormal_control_rate_warns_immediately_then_throttles(
    monkeypatch, capsys
):
    times = iter(index * 0.1 for index in range(61))
    monkeypatch.setattr(demo_module.time, "perf_counter", lambda: next(times))
    ctx = _control_rate_context(
        control_period=0.1,
        control_rate_tolerance=0.01,
    )

    for _ in range(61):
        BxiExample.check_control_frame_rate(ctx)

    output = capsys.readouterr().out
    assert output.count("[CONTROL RATE WARNING]") == 2
    assert "hz=10.0" in output
    assert "[CONTROL RATE RECOVERED]" not in output


def test_control_rate_recovery_is_reported_once(monkeypatch, capsys):
    times = [0.0]
    times.extend(index * 0.1 for index in range(1, 11))
    times.extend(1.0 + index * 0.02 for index in range(1, 101))
    time_values = iter(times)
    monkeypatch.setattr(
        demo_module.time, "perf_counter", lambda: next(time_values)
    )
    ctx = _control_rate_context()

    for _ in times:
        BxiExample.check_control_frame_rate(ctx)

    output = capsys.readouterr().out
    assert output.count("[CONTROL RATE WARNING]") == 1
    assert output.count("[CONTROL RATE RECOVERED]") == 1
