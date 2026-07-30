from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import zmq

from bxi_example_py_elf3.inference import sonic


IDLE_FRAME_START = 3509


class FakeClock:
    def __init__(self, initial: float = 10.0):
        self.now = initial

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeTensorInfo:
    name: str
    shape: tuple[int, int]


@dataclass
class RecordingOnnxSession:
    action_value: float = 1.0
    calls: list[np.ndarray] = field(default_factory=list)

    def get_inputs(self) -> list[FakeTensorInfo]:
        return [FakeTensorInfo("observations", (1, sonic.MODEL_INPUT_DIM))]

    def get_outputs(self) -> list[FakeTensorInfo]:
        return [FakeTensorInfo("actions", (1, sonic.NUM_JOINTS))]

    def run(self, output_names, feeds):
        assert output_names == ["actions"]
        model_input = np.asarray(feeds["observations"], dtype=np.float32)
        self.calls.append(model_input.copy())
        return [
            np.full(
                (1, sonic.NUM_JOINTS), self.action_value, dtype=np.float32
            )
        ]


class FakeZmqSocket:
    def __init__(self):
        self.messages: list[bytes] = []
        self.closed = False

    def recv(self, flags=0) -> bytes:
        if self.messages:
            return self.messages.pop(0)
        raise zmq.Again()

    def close(self, linger=0) -> None:
        self.closed = True


class FakeZmqPoller:
    def __init__(self, socket: FakeZmqSocket):
        self.socket = socket

    def poll(self, timeout=0):
        if self.socket.messages:
            return [(self.socket, zmq.POLLIN)]
        return []

    def unregister(self, socket) -> None:
        assert socket is self.socket


class FakeZmqContext:
    def __init__(self):
        self.terminated = False

    def term(self) -> None:
        self.terminated = True


class FakeWire:
    def __init__(self):
        self.fields_by_message: dict[bytes, dict[str, np.ndarray]] = {}
        self.next_id = 0

    def push(self, socket: FakeZmqSocket, fields: dict[str, np.ndarray]) -> None:
        message = f"message-{self.next_id}".encode()
        self.next_id += 1
        self.fields_by_message[message] = fields
        socket.messages.append(message)

    def decode(self, message: bytes, topic: str) -> dict[str, np.ndarray]:
        assert topic == "smpl_ref"
        return self.fields_by_message[message]


def _live_fields(
    value: float,
    *,
    source_ready: bool = True,
    frame_index: int = 100,
) -> dict[str, np.ndarray]:
    root_quat = np.zeros((sonic.WINDOW, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    return {
        "source_ready": np.array([source_ready], dtype=bool),
        "source_stream_mode": np.array([1], dtype=np.int32),
        "source_calibration_ready": np.array([True], dtype=bool),
        "frame_index": np.array([frame_index], dtype=np.int64),
        "term1_local": np.full(
            (sonic.WINDOW, 72), value, dtype=np.float32
        ),
        "root_quat": root_quat,
        "wrist": np.full((sonic.WINDOW, 6), value + 0.25, dtype=np.float32),
    }


def _robot_observation():
    return (
        sonic.DEFAULT_DOF_POS.copy(),
        np.zeros(sonic.NUM_JOINTS, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float32),
    )


@pytest.fixture
def policy_harness(tmp_path, monkeypatch):
    frame_count = IDLE_FRAME_START + sonic.WINDOW + 1
    frame_values = np.arange(frame_count, dtype=np.float32)
    term1 = np.repeat(frame_values[:, None], 72, axis=1)
    root_quat = np.zeros((frame_count, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    wrist = np.repeat((frame_values + 0.5)[:, None], 6, axis=1)
    reference_path = tmp_path / "idle_left_reference.npz"
    np.savez(
        reference_path,
        term1_local=term1,
        root_quat=root_quat,
        wrist=wrist,
    )

    clock = FakeClock()
    session = RecordingOnnxSession()
    wire = FakeWire()

    def init_onnx(policy) -> None:
        policy.session = session
        policy.input_info = session.get_inputs()[0]
        policy.output_info = session.get_outputs()[0]
        policy.input_buffer = np.zeros(
            (1, sonic.MODEL_INPUT_DIM), dtype=np.float32
        )

    def init_zmq(policy) -> None:
        policy.zmq_context = FakeZmqContext()
        policy.zmq_socket = FakeZmqSocket()
        policy.zmq_poller = FakeZmqPoller(policy.zmq_socket)

    monkeypatch.setattr(sonic.SonicTeleopPolicy, "_init_onnx", init_onnx)
    monkeypatch.setattr(sonic.SonicTeleopPolicy, "_init_zmq", init_zmq)
    monkeypatch.setattr(sonic, "_decode_packed_message", wire.decode)
    monkeypatch.setattr(sonic.time, "monotonic", clock.monotonic)

    policy = sonic.SonicTeleopPolicy(
        model_onnx_path="mock.onnx",
        stream_reference_npz=str(reference_path),
        use_smpl_ref_zmq=True,
    )
    policy.live_ref_timeout_s = 0.5
    policy.source_blend_duration_s = 0.4

    yield policy, session, wire, clock, term1

    policy.close()


def _assert_idle_reference_input(model_input: np.ndarray, term1: np.ndarray) -> None:
    expected = term1[
        IDLE_FRAME_START : IDLE_FRAME_START + sonic.WINDOW
    ].reshape(-1)
    np.testing.assert_array_equal(model_input[0, :720], expected)


def test_without_pico_runs_onnx_with_fixed_idle_frame_3509(policy_harness):
    policy, session, _, clock, term1 = policy_harness
    observation = _robot_observation()

    policy.inference_step(*observation)
    clock.advance(0.4)
    target = policy.inference_step(*observation)

    assert len(session.calls) == 2
    _assert_idle_reference_input(session.calls[0], term1)
    _assert_idle_reference_input(session.calls[1], term1)
    assert policy.motion_cursor == IDLE_FRAME_START
    assert policy.last_status == "idle_reference"
    np.testing.assert_allclose(
        target,
        sonic.DEFAULT_DOF_POS + sonic.ACTION_SCALE,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_source_ready_false_is_rejected_and_policy_stays_on_idle(policy_harness):
    policy, session, wire, _, term1 = policy_harness
    wire.push(policy.zmq_socket, _live_fields(9000.0, source_ready=False))

    policy.inference_step(*_robot_observation())

    assert policy.latest_live_ref is None
    assert policy.live_sequence == 0
    assert policy.reference_source == "idle"
    assert policy.last_status == "idle_reference"
    _assert_idle_reference_input(session.calls[-1], term1)


def test_ready_live_reference_switches_and_blends_over_point_four_seconds(
    policy_harness,
):
    policy, session, wire, clock, _ = policy_harness
    observation = _robot_observation()

    session.action_value = 1.0
    policy.inference_step(*observation)
    clock.advance(0.4)
    idle_target = policy.inference_step(*observation)
    np.testing.assert_allclose(
        idle_target, sonic.DEFAULT_DOF_POS + sonic.ACTION_SCALE
    )

    session.action_value = 3.0
    wire.push(policy.zmq_socket, _live_fields(9000.0))
    clock.advance(0.01)
    transition_start = policy.inference_step(*observation)

    assert policy.reference_source == "live"
    assert policy.last_status == "live_reference"
    assert policy.latest_live_ref is not None
    np.testing.assert_array_equal(session.calls[-1][0, :720], 9000.0)
    np.testing.assert_allclose(transition_start, idle_target)

    clock.advance(0.2)
    transition_midpoint = policy.inference_step(*observation)
    np.testing.assert_allclose(
        transition_midpoint,
        sonic.DEFAULT_DOF_POS + 2.0 * sonic.ACTION_SCALE,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    # Step just beyond the configured boundary to avoid binary-float rounding
    # keeping a mathematically exact 0.4 s transition active for one cycle.
    clock.advance(0.201)
    transition_end = policy.inference_step(*observation)
    np.testing.assert_allclose(
        transition_end,
        sonic.DEFAULT_DOF_POS + 3.0 * sonic.ACTION_SCALE,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert not policy.source_blend_active


def test_stale_live_reference_blends_back_to_idle_and_reports_transition(
    policy_harness,
):
    policy, session, wire, clock, term1 = policy_harness
    observation = _robot_observation()

    policy.source_blend_duration_s = 0.0
    session.action_value = 3.0
    wire.push(policy.zmq_socket, _live_fields(9000.0))
    live_target = policy.inference_step(*observation)
    assert policy.last_status == "live_reference"

    policy.source_blend_duration_s = 0.4
    session.action_value = 1.0
    clock.advance(policy.live_ref_timeout_s + 0.001)
    stale_transition_start = policy.inference_step(*observation)

    assert policy.latest_live_ref is None
    assert policy.reference_source == "idle"
    assert policy.last_status == "live_stale_to_idle"
    _assert_idle_reference_input(session.calls[-1], term1)
    np.testing.assert_allclose(stale_transition_start, live_target)

    clock.advance(0.2)
    stale_transition_midpoint = policy.inference_step(*observation)
    assert policy.last_status == "live_stale_to_idle"
    np.testing.assert_allclose(
        stale_transition_midpoint,
        sonic.DEFAULT_DOF_POS + 2.0 * sonic.ACTION_SCALE,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    clock.advance(0.201)
    idle_target = policy.inference_step(*observation)
    assert policy.last_status == "idle_reference"
    np.testing.assert_allclose(
        idle_target,
        sonic.DEFAULT_DOF_POS + sonic.ACTION_SCALE,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_reset_clears_live_state_drains_queued_packet_and_does_not_reuse_it(
    policy_harness,
):
    policy, session, wire, _, term1 = policy_harness
    observation = _robot_observation()

    wire.push(policy.zmq_socket, _live_fields(9000.0, frame_index=100))
    policy.inference_step(*observation)
    assert policy.latest_live_ref is not None

    wire.push(policy.zmq_socket, _live_fields(9100.0, frame_index=101))
    policy.reset()

    assert policy.latest_live_ref is None
    assert policy.latest_live_ref_time == 0.0
    assert policy.live_sequence == 0
    assert policy.reference_source is None
    assert not policy.zmq_socket.messages

    policy.inference_step(*observation)
    assert policy.latest_live_ref is None
    assert policy.reference_source == "idle"
    assert policy.last_status == "idle_reference"
    _assert_idle_reference_input(session.calls[-1], term1)
