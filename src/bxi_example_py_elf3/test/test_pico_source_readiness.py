from pathlib import Path

import numpy as np
import pytest

import bxi_example_py_elf3.sonic_pico.pico_pose_to_smpl_ref_bridge as bridge_module
from bxi_example_py_elf3.sonic_pico.pico_pose_to_smpl_ref_bridge import (
    POSE_STREAM_MODE,
    PicoSourceReadinessGate,
    StreamedSmplRefMerger,
    _build_live_smpl_ref_if_ready,
    _extract_wrist_frames,
    _field_scalar,
    _parse_args,
    _parse_incoming_chunk,
    _report_stream_state,
)


STALE_SECONDS = 0.2


def test_bridge_accepts_legacy_log_option_without_periodic_telemetry():
    assert _parse_args(["--log-every", "2"]).log_every == 2.0


def test_bridge_reports_only_stream_state_changes(capsys):
    smpl_ref = {"frame_index": np.array([42], dtype=np.int64)}
    state = _report_stream_state(
        None,
        "waiting",
        received=0,
        skipped=0,
        smpl_ref=None,
        input_age=float("inf"),
    )
    state = _report_stream_state(
        state,
        "waiting",
        received=0,
        skipped=0,
        smpl_ref=None,
        input_age=float("inf"),
    )
    state = _report_stream_state(
        state,
        "streaming",
        received=3,
        skipped=0,
        smpl_ref=smpl_ref,
        input_age=0.01,
    )
    _report_stream_state(
        state,
        "streaming",
        received=4,
        skipped=0,
        smpl_ref=smpl_ref,
        input_age=0.01,
    )

    output = capsys.readouterr().out
    assert output.count("waiting for calibrated") == 1
    assert output.count("stream ready") == 1


def test_vendor_manager_has_no_periodic_healthy_or_waiting_logs():
    manager_path = (
        Path(bridge_module.__file__).parent
        / "vendor/gear_sonic/scripts/pico_manager_thread_server.py"
    )
    source = manager_path.read_text(encoding="utf-8")

    assert "waiting for body data..." not in source
    assert "[PicoReader] dt_ts:" not in source
    assert "PICO RATE WARNING" in source
    assert "PICO RATE RECOVERED" in source
    assert "ELF3_L_WRIST_X_IDX = 19" in source
    assert "ELF3_L_WRIST_Y_IDX = 20" in source
    assert "ELF3_L_WRIST_Z_IDX = 21" in source
    assert "G1_L_WRIST" not in source


def test_bridge_reads_native_elf3_wrist_slots():
    joint_pos = np.arange(29, dtype=np.float32).reshape(1, 29)

    wrist = _extract_wrist_frames(joint_pos)

    np.testing.assert_array_equal(wrist[0], [19, 20, 21, 26, 27, 28])


@pytest.mark.parametrize(
    "value",
    [np.array([np.nan]), np.array([np.inf]), np.array([-np.inf]), ["bad"], []],
)
def test_bridge_rejects_invalid_pico_button_scalars(value):
    assert _field_scalar({"left_trigger": value}, "left_trigger") is None


def _pose_fields(
    frame_start: int,
    *,
    stream_mode: int = POSE_STREAM_MODE,
    calibration_ready: bool = True,
) -> dict[str, np.ndarray]:
    frame_indices = np.arange(frame_start, frame_start + 10, dtype=np.int64)
    frame_count = frame_indices.size
    return {
        "stream_mode": np.array([stream_mode], dtype=np.int32),
        "calibration_ready": np.array([calibration_ready], dtype=bool),
        "frame_index": frame_indices,
        "smpl_joints": np.zeros((frame_count, 24, 3), dtype=np.float32),
        "body_quat_w": np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (frame_count, 1),
        ),
        "joint_pos": np.zeros((frame_count, 29), dtype=np.float32),
    }


def test_gate_rejects_uncalibrated_and_non_pose_messages():
    gate = PicoSourceReadinessGate(required_consecutive=1)

    assert not gate.observe(
        _pose_fields(0, calibration_ready=False),
        now_mono=0.00,
        stale_seconds=STALE_SECONDS,
    )
    assert not gate.is_fresh(0.00, STALE_SECONDS)

    assert not gate.observe(
        _pose_fields(1, stream_mode=0),
        now_mono=0.01,
        stale_seconds=STALE_SECONDS,
    )
    assert not gate.is_fresh(0.01, STALE_SECONDS)


def test_gate_rejects_a_repeated_frame_and_revokes_ready_immediately():
    gate = PicoSourceReadinessGate(required_consecutive=1)
    fields = _pose_fields(10)

    assert gate.observe(fields, now_mono=0.00, stale_seconds=STALE_SECONDS)
    assert gate.is_fresh(0.00, STALE_SECONDS)
    assert not gate.observe(fields, now_mono=0.01, stale_seconds=STALE_SECONDS)
    assert not gate.is_fresh(0.01, STALE_SECONDS)


def test_gate_requires_three_consecutive_progressing_messages():
    gate = PicoSourceReadinessGate()

    assert not gate.observe(
        _pose_fields(20), now_mono=0.00, stale_seconds=STALE_SECONDS
    )
    assert not gate.observe(
        _pose_fields(21), now_mono=0.01, stale_seconds=STALE_SECONDS
    )
    assert gate.observe(
        _pose_fields(22), now_mono=0.02, stale_seconds=STALE_SECONDS
    )
    assert gate.is_fresh(0.02, STALE_SECONDS)


def test_gate_recovers_after_pose_session_frame_counter_resets():
    gate = PicoSourceReadinessGate()

    for index, frame_start in enumerate((1000, 1001, 1002)):
        gate.observe(
            _pose_fields(frame_start),
            now_mono=index * 0.01,
            stale_seconds=STALE_SECONDS,
        )
    assert gate.is_fresh(0.02, STALE_SECONDS)

    assert not gate.observe(
        _pose_fields(0), now_mono=0.03, stale_seconds=STALE_SECONDS
    )
    assert not gate.is_fresh(0.03, STALE_SECONDS)
    assert not gate.observe(
        _pose_fields(1), now_mono=0.04, stale_seconds=STALE_SECONDS
    )
    assert gate.observe(
        _pose_fields(2), now_mono=0.05, stale_seconds=STALE_SECONDS
    )
    assert gate.is_fresh(0.05, STALE_SECONDS)


def test_gate_is_not_fresh_after_stale_timeout():
    gate = PicoSourceReadinessGate()
    for index, frame_start in enumerate((30, 31, 32)):
        gate.observe(
            _pose_fields(frame_start),
            now_mono=index * 0.01,
            stale_seconds=STALE_SECONDS,
        )

    assert gate.is_fresh(0.02 + STALE_SECONDS, STALE_SECONDS)
    assert not gate.is_fresh(0.02 + STALE_SECONDS + 0.001, STALE_SECONDS)


def test_stale_gate_stops_live_output_and_clears_buffered_reference():
    gate = PicoSourceReadinessGate()
    latest_fields = None
    for index, frame_start in enumerate((40, 41, 42)):
        latest_fields = _pose_fields(frame_start)
        gate.observe(
            latest_fields,
            now_mono=index * 0.01,
            stale_seconds=STALE_SECONDS,
        )

    merger = StreamedSmplRefMerger()
    merger.merge(_parse_incoming_chunk(latest_fields))

    live_ref = _build_live_smpl_ref_if_ready(
        gate, merger, now_mono=0.02, stale_seconds=STALE_SECONDS
    )
    assert live_ref is not None
    assert bool(live_ref["source_ready"][0])
    assert int(live_ref["source_stream_mode"][0]) == POSE_STREAM_MODE

    stale_ref = _build_live_smpl_ref_if_ready(
        gate,
        merger,
        now_mono=0.02 + STALE_SECONDS + 0.001,
        stale_seconds=STALE_SECONDS,
    )
    assert stale_ref is None
    assert merger.timesteps == 0
