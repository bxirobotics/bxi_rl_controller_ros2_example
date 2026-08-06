from __future__ import annotations

import importlib
import json
import platform
import sys
from threading import Event
import time
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from bxi_example_py_elf3.framework.inference import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.joints import (
    JointCommandDefaults,
    JointCommandResolver,
    JointLayout,
    JointStateBuffer,
    JointTargetBuffer,
)
from bxi_example_py_elf3.framework.mod_api import MotorFrame
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_JOINTS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = PACKAGE_ROOT / "mods" / "com.bxi.pico_gmr_motion"
_PACKAGE_NAME = "_test_pico_gmr_mod"
if _PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(MOD_ROOT)]
    sys.modules[_PACKAGE_NAME] = package

protocol = importlib.import_module(f"{_PACKAGE_NAME}.protocol")
reference = importlib.import_module(f"{_PACKAGE_NAME}.reference")
head_tracking = importlib.import_module(f"{_PACKAGE_NAME}.head_tracking")
rgmt_policy = importlib.import_module(f"{_PACKAGE_NAME}.rgmt_policy")
state_module = importlib.import_module(f"{_PACKAGE_NAME}.state")
tracking_gate = importlib.import_module(f"{_PACKAGE_NAME}.tracking_gate")
xrt_session = importlib.import_module(f"{_PACKAGE_NAME}.xrt_session")
try:
    gmr = importlib.import_module(f"{_PACKAGE_NAME}.gmr")
except ModuleNotFoundError as error:
    if error.name != "mujoco":
        raise
    gmr = None


def _frame(session: int, sequence: int):
    return protocol.LiveReferenceFrame(
        session_id=session,
        sequence=sequence,
        source_timestamp_ns=sequence * 20_000_000,
        joint_pos=np.arange(29, dtype=np.float32) + sequence,
        head_joint_pos=np.asarray((0.1, -0.2), dtype=np.float32),
        anchor_quat_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        anchor_lin_vel_w=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
        anchor_ang_vel_w=np.asarray((4.0, 5.0, 6.0), dtype=np.float32),
    )


def _bounded_frame(
    session: int,
    sequence: int,
    joint_pos: np.ndarray,
    *,
    source_period_s: float = 0.01,
    head_joint_pos: object = (0.0, 0.0),
):
    return protocol.LiveReferenceFrame(
        session_id=session,
        sequence=sequence,
        source_timestamp_ns=int(sequence * source_period_s * 1.0e9),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        head_joint_pos=np.asarray(head_joint_pos, dtype=np.float32),
        anchor_quat_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        anchor_lin_vel_w=np.asarray((0.5, 0.0, 0.0), dtype=np.float32),
        anchor_ang_vel_w=np.asarray((0.0, 0.0, 0.5), dtype=np.float32),
    )


def test_mod_is_standalone_and_owns_both_rgmt_models():
    manifest = yaml.safe_load((MOD_ROOT / "mod.yaml").read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["requires"]] == [
        "com.bxi.basic_actions"
    ]
    source = manifest["nodes"]["pico_gmr_source"]
    assert source["entrypoint"] == "pico_gmr_launcher.py"
    assert source["lifecycle"] == "state"
    assert source["states"] == ["pico_gmr_motion"]
    assert source["arguments"][source["arguments"].index("--rate-hz") + 1] == "50"
    media_server = manifest["nodes"]["mediamtx_server"]
    assert media_server["entrypoint"] == "mediamtx_launcher.py"
    assert media_server["lifecycle"] == "state"
    assert media_server["states"] == ["pico_gmr_motion"]
    streamer = manifest["nodes"]["head_camera_rtsp"]
    assert streamer["runtime"] == "executable"
    assert streamer["runtime_profile"] == "pico_gmr_host"
    assert streamer["depends_on"] == ["mediamtx_server"]
    assert streamer["params"]["simulation_topic"].startswith("/simulation/")
    assert streamer["params"]["hardware_topic"].startswith("/hardware/")
    assert streamer["params"]["rtsp_url"] == "rtsp://127.0.0.1:2212/video"
    assert streamer["params"]["encoder"] == "libx264"
    assert streamer["scheduling"]["cpu_affinity"] == "background"
    state_params = manifest["states"]["pico_gmr_motion"]["params"]
    assert "startup_timeout_s" not in state_params
    assert state_params["head_pitch_limit_rad"] == pytest.approx(0.5)
    assert state_params["head_yaw_limit_rad"] == pytest.approx(1.0)
    assert "actions" not in manifest
    assert (MOD_ROOT / "assets" / "rgmt.onnx").stat().st_size > 1_000_000
    assert (MOD_ROOT / "assets" / "rgmt.rknn").stat().st_size > 1_000_000
    assert (MOD_ROOT / "tools" / "diagnose_pico_gmr_roundtrip.py").is_file()
    assert (MOD_ROOT / "tools" / "build_rtsp_streamer.py").is_file()
    assert (
        MOD_ROOT / "native" / "rtsp_streamer" / "head_camera_rtsp_node.cpp"
    ).is_file()
    assert (MOD_ROOT / "runtime" / "mediamtx.yml").is_file()
    assert (MOD_ROOT / "vendor" / "licenses" / "MediaMTX.LICENSE").is_file()
    assert (
        MOD_ROOT / "vendor" / "licenses" / "MediaMTX.PROVENANCE.txt"
    ).is_file()
    for runtime_platform in ("linux-x86_64", "linux-aarch64"):
        mediamtx = MOD_ROOT / "runtime" / runtime_platform / "mediamtx"
        assert mediamtx.is_file()
        assert mediamtx.stat().st_mode & 0o111
        assert mediamtx.stat().st_size > 40_000_000
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        streamer_binary = (
            MOD_ROOT / "bin" / "linux-x86_64" / "head_camera_rtsp_node"
        )
        assert streamer_binary.is_file()
        assert streamer_binary.stat().st_mode & 0o111
    for runtime_platform, binding in (
        ("linux-x86_64", "xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so"),
        ("linux-aarch64", "xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so"),
    ):
        service = (
            MOD_ROOT
            / "runtime"
            / runtime_platform
            / "roboticsservice"
            / "RoboticsServiceProcess"
        )
        assert service.is_file()
        assert service.stat().st_mode & 0o111
        bundled_binding = (
            MOD_ROOT
            / "vendor"
            / "python"
            / f"{runtime_platform}-cpython-310"
            / binding
        )
        assert bundled_binding.is_file()


def test_pico_ax_combo_toggles_tracking_only_on_rising_edges():
    gate = tracking_gate.TrackingGate()
    assert not gate.is_enabled()
    assert gate.update(False, False) is None
    assert gate.update(True, True) is True
    assert gate.is_enabled()
    assert gate.update(True, True) is None
    assert gate.update(False, False) is None
    assert gate.update(True, True) is False
    assert not gate.is_enabled()


def test_pico_worker_logs_raw_ax_button_states_before_toggling():
    source = (MOD_ROOT / "pico_gmr_process.py").read_text(encoding="utf-8")
    state_log = source.index("PICO controller buttons:")
    combo_log = source.index("PICO A+X raw combo detected by XRoboToolkit")
    gate_update = source.index("tracking_gate.update(a_pressed, x_pressed)")
    assert state_log < combo_log < gate_update


def test_blocking_xrt_init_does_not_block_reference_loop_startup():
    class BlockingSdk:
        def __init__(self):
            self.entered = Event()
            self.closed = Event()

        def init(self):
            self.entered.set()
            self.closed.wait(timeout=2.0)

        def close(self):
            self.closed.set()

    sdk = BlockingSdk()
    session = xrt_session.XrtBackgroundSession(sdk)
    started = time.monotonic()
    session.start()
    try:
        assert sdk.entered.wait(timeout=0.2)
        assert time.monotonic() - started < 0.5
        assert session.error is None
    finally:
        session.close()


def test_state_uses_rgmt_standing_fallback_until_live_window_exists():
    class Handle:
        status = "ready"

        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class Receiver:
        def __init__(self):
            self.max_age_s = None
            self.value = None

        def clear(self):
            pass

        def snapshot_window(self, *, max_age_s=None, now=None):
            self.max_age_s = max_age_s
            return self.value

    class Policy:
        def __init__(self):
            target = JointTargetBuffer(ELF3_ISAAC_JOINTS)
            target.update(
                np.linspace(-0.2, 0.2, 29, dtype=np.float32),
                np.ones(29, dtype=np.float32),
                np.ones(29, dtype=np.float32),
            )
            self.output = PolicyOutput(target.view)
            self.calls = []
            self.yaw_resets = 0

        def reset(self, _frame):
            pass

        def reset_reference_yaw_alignment(self):
            self.yaw_resets += 1

        def step_with_reference_window(self, _frame, _dt, **kwargs):
            self.calls.append(kwargs)
            return self.output

    receiver = Receiver()
    policy = Policy()
    state = state_module.PicoGmrMotionState(
        "test/pico_gmr_motion",
        1,
        policy=Handle(policy),
        receiver=Handle(receiver),
        params=state_module.PicoGmrMotionParams(stale_timeout_s=0.4),
    )
    joint_state = JointStateBuffer(ELF3_ISAAC_JOINTS, dtype=np.float32).view
    frame = InferenceFrame(
        joint_state,
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    context = types.SimpleNamespace(
        robot_joints=joint_state,
        inference_frame=frame,
        set_motor_target=lambda motor_frame: motor_targets.append(motor_frame),
        request_state=lambda *args, **kwargs: state_requests.append((args, kwargs)),
    )
    motor_targets = []
    state_requests = []
    state._bind_logger(
        types.SimpleNamespace(
            debug=lambda _message: None,
            info=lambda _message: None,
            warning=lambda _message: None,
            error=lambda _message: None,
        )
    )
    state.on_prepare(context, object())
    preview = state.sample_running_frame(context, 0.02, advance=False)
    assert preview.qpos.shape == (31,)
    assert state._reference_source is None
    assert policy.yaw_resets == 0
    state.on_update(context, 0.02)
    assert motor_targets[-1].qpos.shape == (31,)
    np.testing.assert_allclose(motor_targets[-1].qpos[-2:], 0.0)
    assert len(policy.calls) == 2
    assert policy.calls[-1]["reference_joint_pos_window"].shape == (21, 29)
    assert receiver.max_age_s == pytest.approx(0.4)
    assert policy.yaw_resets == 1
    assert state_requests == []

    live_buffer = reference.ReferenceWindowBuffer()
    for sequence in range(21):
        assert live_buffer.accept(
            _bounded_frame(
                9,
                sequence,
                np.zeros(29),
                head_joint_pos=(0.4, -0.6),
            ),
            1.0 + sequence * 0.02,
        )
    receiver.value = live_buffer.snapshot_window()
    state.on_update(context, 0.02)
    assert state._reference_source == "live"
    assert policy.calls[-1]["reference_joint_pos_window"].shape == (21, 29)
    np.testing.assert_allclose(
        policy.calls[-1]["reference_joint_pos_window"][10],
        0.0,
    )
    assert policy.yaw_resets == 2
    np.testing.assert_allclose(motor_targets[-1].qpos[-2:], (0.03, -0.04))

    before_preview = state._head_command.position.copy()
    preview = state.sample_running_frame(context, 0.02, advance=False)
    np.testing.assert_array_equal(state._head_command.position, before_preview)
    np.testing.assert_allclose(preview.qpos[-2:], before_preview)

    receiver.value = None
    state.on_update(context, 0.02)
    np.testing.assert_allclose(motor_targets[-1].qpos[-2:], 0.0, atol=1.0e-7)


def test_private_rgmt_policy_runs_live_window_without_advancing_preview():
    pytest.importorskip("onnxruntime")
    policy = rgmt_policy.RgmtExternalReferencePolicy.for_live_reference(
        str(MOD_ROOT / "assets" / "rgmt.onnx"),
        backend="onnxruntime",
    )
    try:
        joint_state = JointStateBuffer(
            ELF3_ISAAC_JOINTS,
            dtype=np.float32,
        ).view
        inference_frame = InferenceFrame(
            joint_state,
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        joints = np.zeros((21, 29), dtype=np.float32)
        quaternions = np.zeros((21, 4), dtype=np.float32)
        quaternions[:, 0] = 1.0
        velocities = np.zeros((21, 3), dtype=np.float32)
        policy.reset(inference_frame)
        previous_action = policy.action_buffer.copy()
        policy.step_with_reference_window(
            inference_frame,
            0.02,
            reference_joint_pos_window=joints,
            reference_anchor_quat_window_w=quaternions,
            reference_anchor_lin_vel_window_w=velocities,
            reference_anchor_ang_vel_window_w=velocities,
            advance=False,
        )
        np.testing.assert_array_equal(policy.action_buffer, previous_action)
        output = policy.step_with_reference_window(
            inference_frame,
            0.02,
            reference_joint_pos_window=joints,
            reference_anchor_quat_window_w=quaternions,
            reference_anchor_lin_vel_window_w=velocities,
            reference_anchor_ang_vel_window_w=velocities,
            advance=True,
        )
        assert output.joints.position.shape == (29,)
        assert np.isfinite(output.joints.position).all()
    finally:
        policy.close()


def test_reference_protocol_round_trip_and_layout_guard():
    packet = protocol.encode_reference_frame(_frame(11, 7))
    decoded = protocol.decode_reference_frame(packet)
    assert len(packet) == protocol.PACKET_SIZE
    assert decoded.session_id == 11
    assert decoded.sequence == 7
    np.testing.assert_allclose(decoded.joint_pos, np.arange(29) + 7)
    np.testing.assert_allclose(decoded.head_joint_pos, (0.1, -0.2))
    assert len(protocol.REFERENCE_JOINT_NAMES) == 29
    assert protocol.HEAD_JOINT_NAMES == ("head_y_joint", "head_z_joint")
    assert protocol.VERSION == 2

    damaged = bytearray(packet)
    damaged[8] ^= 0x01
    with pytest.raises(ValueError, match="layout"):
        protocol.decode_reference_frame(bytes(damaged))

    invalid = _frame(11, 8)
    object.__setattr__(invalid, "head_joint_pos", np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="head_joint_pos"):
        protocol.encode_reference_frame(invalid)
    object.__setattr__(invalid, "head_joint_pos", np.asarray((np.nan, 0.0)))
    with pytest.raises(ValueError, match="NaN"):
        protocol.encode_reference_frame(invalid)


def test_31_joint_head_output_resolves_by_name_on_29_and_reordered_31_robots():
    source_layout = state_module.PICO_GMR_OUTPUT_JOINTS
    source = MotorFrame.create(
        source_layout,
        np.arange(source_layout.dof_num, dtype=np.float32),
        np.ones(source_layout.dof_num, dtype=np.float32),
        np.full(source_layout.dof_num, 2.0, dtype=np.float32),
    )

    resolver_29 = JointCommandResolver(ELF3_ISAAC_JOINTS, JointCommandDefaults())
    output_29 = MotorFrame.empty(ELF3_ISAAC_JOINTS)
    with pytest.warns(RuntimeWarning, match="head_y_joint"):
        resolver_29.resolve_into(source, output_29)
    np.testing.assert_array_equal(output_29.qpos, np.arange(29, dtype=np.float32))

    reordered = JointLayout(tuple(reversed(source_layout.names)), label="reordered ELF3")
    resolver_31 = JointCommandResolver(reordered, JointCommandDefaults())
    output_31 = MotorFrame.empty(reordered)
    resolver_31.resolve_into(source, output_31)
    for robot_index, name in enumerate(reordered.names):
        assert output_31.qpos[robot_index] == pytest.approx(source_layout.index(name))


def test_reference_window_uses_unique_sequences_and_resets_sessions():
    buffer = reference.ReferenceWindowBuffer()
    first_joint = protocol.REFERENCE_JOINT_NAMES.index("l_shoulder_y_joint")
    for sequence in range(21):
        joints = np.zeros(29, dtype=np.float32)
        joints[first_joint] = sequence * 0.01
        assert buffer.accept(
            _bounded_frame(1, sequence, joints),
            sequence * 0.01,
        )
    window = buffer.snapshot_window()
    assert window is not None
    assert window.latest_sequence == 20
    assert window.joint_pos.shape == (21, 29)
    assert window.head_joint_pos.shape == (21, 2)
    assert window.joint_pos[10, first_joint] == pytest.approx(0.10)
    assert window.joint_pos[9, first_joint] == pytest.approx(0.09)
    assert window.joint_pos[0, first_joint] == pytest.approx(0.0)
    np.testing.assert_allclose(
        np.diff(window.joint_pos[:, first_joint]),
        0.01,
        atol=1.0e-6,
    )

    assert not buffer.accept(_bounded_frame(1, 20, joints), 0.21)
    assert buffer.snapshot_window().latest_sequence == 20

    new_session_joints = np.full(29, 0.05, dtype=np.float32)
    assert buffer.accept(_bounded_frame(2, 0, new_session_joints), 0.22)
    assert buffer.snapshot_window() is None
    for sequence in range(1, 21):
        assert buffer.accept(
            _bounded_frame(2, sequence, new_session_joints),
            0.22 + sequence * 0.02,
        )
    new_window = buffer.snapshot_window()
    assert new_window is not None
    assert new_window.session_id == 2
    np.testing.assert_allclose(
        new_window.joint_pos,
        np.repeat(new_session_joints[None, :], 21, axis=0),
    )


def _axis_quaternion(axis: int, angle: float) -> np.ndarray:
    result = np.zeros(4, dtype=np.float64)
    result[0] = np.cos(angle * 0.5)
    result[axis + 1] = np.sin(angle * 0.5)
    return result


def test_pico_head_mapper_matches_mocaplab_axes_and_recenters_each_session():
    mapper = head_tracking.PicoHeadMapper(pitch_limit_rad=0.5, yaw_limit_rad=1.0)
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))

    np.testing.assert_allclose(mapper.update(identity, identity), 0.0)
    np.testing.assert_allclose(
        mapper.update(identity, _axis_quaternion(0, 0.25)),
        (-0.25, 0.0),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        mapper.update(identity, _axis_quaternion(1, 0.4)),
        (0.0, 0.4),
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        mapper.update(identity, _axis_quaternion(0, 0.8)),
        (-0.5, 0.0),
        atol=1.0e-6,
    )
    mapper.reset()
    np.testing.assert_allclose(
        mapper.update(identity, _axis_quaternion(0, 0.3)),
        0.0,
        atol=1.0e-6,
    )


def test_recorded_pico_frame_centers_without_head_mount_offset():
    payload = json.loads(
        (MOD_ROOT / "tmp" / "pico_gmr_real.json").read_text(encoding="utf-8")
    )
    human = payload["human_data"]
    mapper = head_tracking.PicoHeadMapper()
    result = mapper.update(
        human["Spine3"]["quaternion_wxyz"],
        human["Head"]["quaternion_wxyz"],
    )
    np.testing.assert_allclose(result, 0.0, atol=1.0e-7)


def test_reference_window_waits_for_all_21_unique_frames():
    buffer = reference.ReferenceWindowBuffer()
    for sequence in range(20):
        assert buffer.accept(
            _bounded_frame(1, sequence, np.zeros(29)),
            sequence * 0.02,
        )
    assert buffer.snapshot_window() is None
    assert buffer.accept(_bounded_frame(1, 20, np.zeros(29)), 0.4)
    assert buffer.snapshot_window() is not None


def test_stale_reference_window_is_rejected_without_clearing_session():
    buffer = reference.ReferenceWindowBuffer()
    for sequence in range(21):
        assert buffer.accept(
            _bounded_frame(1, sequence, np.zeros(29)),
            19.8 + sequence * 0.01,
        )
    assert buffer.snapshot_window(max_age_s=0.4, now=20.39) is not None
    assert buffer.snapshot_window(max_age_s=0.4, now=20.41) is None
    assert buffer.snapshot_window() is not None


def test_unity_coordinate_conversion_matches_mocaplab():
    if gmr is None:
        pytest.skip("mujoco is not installed")
    position, quaternion = gmr.unity_pose_to_right_handed(
        (1.0, 2.0, 3.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(position, (1.0, -3.0, 2.0))
    np.testing.assert_allclose(
        quaternion,
        (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0),
    )


def test_arm_orientation_offsets_use_current_elf3_frames():
    if gmr is None:
        pytest.skip("mujoco is not installed")
    config = json.loads(
        (MOD_ROOT / "assets" / "pico_to_elf3.json").read_text(encoding="utf-8")
    )
    expected_frames = {
        "l_shoulder_y_link": np.asarray((np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)),
        "l_elbow_y_link": np.asarray((0.0, 0.0, 1.0, 0.0)),
        "l_wrist_z_link": np.asarray((0.0, 0.0, 1.0, 0.0)),
        "r_shoulder_y_link": np.asarray((0.0, np.sqrt(0.5), 0.0, -np.sqrt(0.5))),
        "r_elbow_y_link": np.asarray((0.0, 1.0, 0.0, 0.0)),
        "r_wrist_z_link": np.asarray((0.0, 1.0, 0.0, 0.0)),
    }
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config[table_name]
        for link_name, expected_frame in expected_frames.items():
            actual_frame = np.asarray(table[link_name][4])
            assert abs(float(actual_frame @ expected_frame)) == pytest.approx(
                1.0,
                abs=1.0e-12,
            )


def test_gmr_reset_uses_named_rgmt_standing_warm_start():
    if gmr is None:
        pytest.skip("mujoco is not installed")
    config = json.loads(
        (MOD_ROOT / "assets" / "pico_to_elf3.json").read_text(encoding="utf-8")
    )
    warm_start = config["warm_start_joint_positions"]
    assert set(warm_start) == set(protocol.REFERENCE_JOINT_NAMES)
    solver = gmr.PicoGmrRetargeter(
        PACKAGE_ROOT / "data" / "mujoco_simulation" / "elf3.xml",
        MOD_ROOT / "assets" / "pico_to_elf3.json",
        protocol.REFERENCE_JOINT_NAMES,
        max_iterations=1,
    )
    expected = np.asarray(
        [warm_start[name] for name in protocol.REFERENCE_JOINT_NAMES]
    )
    solver.data.qpos[solver._qpos_addresses] = 0.0
    solver.reset()
    np.testing.assert_allclose(
        solver.data.qpos[solver._qpos_addresses],
        expected,
        atol=1.0e-12,
    )
    for elbow in ("l_elbow_y_joint", "r_elbow_y_joint"):
        assert warm_start[elbow] == pytest.approx(1.27999997)


def test_portable_box_qp_matches_known_bounded_solution():
    if gmr is None:
        pytest.skip("mujoco is not installed")
    hessian = np.asarray(((4.0, 1.0), (1.0, 2.0)))
    linear = np.asarray((-8.0, 3.0))
    solution = gmr._solve_box_qp(
        hessian,
        linear,
        np.asarray((-0.5, -0.25)),
        np.asarray((0.75, 0.5)),
    )
    np.testing.assert_allclose(solution, (0.75, -0.25), atol=1.0e-12)


def test_gmr_frame_task_cost_matches_mocaplab_squared_cost_convention():
    if gmr is None:
        pytest.skip("mujoco is not installed")
    solver = gmr.PicoGmrRetargeter(
        PACKAGE_ROOT / "data" / "mujoco_simulation" / "elf3.xml",
        MOD_ROOT / "assets" / "pico_to_elf3.json",
        protocol.REFERENCE_JOINT_NAMES,
        max_iterations=1,
    )
    body_id = solver._required_id(gmr.mujoco.mjtObj.mjOBJ_BODY, "l_wrist_z_link")
    position_cost = 7.0
    target = (
        (
            body_id,
            position_cost,
            0.0,
            solver.data.xpos[body_id].copy(),
            solver.data.xquat[body_id].copy(),
        ),
    )
    hessian, linear, error = solver._qp_objective(target)

    jacobian_position = np.zeros((3, solver.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, solver.model.nv), dtype=np.float64)
    gmr.mujoco.mj_jacBody(
        solver.model,
        solver.data,
        jacobian_position,
        jacobian_rotation,
        body_id,
    )
    rotation_world_from_body = solver.data.xmat[body_id].reshape(3, 3)
    body_position_jacobian = (
        rotation_world_from_body.T
        @ jacobian_position[:, solver._solve_dofs]
    )
    expected = (
        solver.damping * np.eye(len(solver._solve_dofs))
        + position_cost**2
        * (body_position_jacobian.T @ body_position_jacobian)
    )
    np.testing.assert_allclose(hessian, expected, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(linear, 0.0, atol=1.0e-12)
    assert error == pytest.approx(0.0, abs=1.0e-12)
    for removed_filter in (
        "continuity_weight",
        "_joint_velocity_limits",
        "shoulder_smoothing",
        "_stabilize_joint_solution",
    ):
        assert not hasattr(solver, removed_filter)


def test_gmr_outputs_named_29_joint_pose_inside_model_limits():
    if gmr is None:
        pytest.skip("mujoco is not installed")
    model_xml = PACKAGE_ROOT / "data" / "mujoco_simulation" / "elf3.xml"
    config_json = MOD_ROOT / "assets" / "pico_to_elf3.json"
    solver = gmr.PicoGmrRetargeter(
        model_xml,
        config_json,
        protocol.REFERENCE_JOINT_NAMES,
        max_iterations=1,
    )
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    positions = {
        "Spine3": (0.0, 0.0, 1.25),
        "Pelvis": (0.0, 0.0, 0.95),
        "Left_Hip": (0.0, 0.12, 0.92),
        "Right_Hip": (0.0, -0.12, 0.92),
        "Left_Knee": (0.0, 0.12, 0.52),
        "Right_Knee": (0.0, -0.12, 0.52),
        "Left_Foot": (0.08, 0.12, 0.08),
        "Right_Foot": (0.08, -0.12, 0.08),
        "Left_Shoulder": (0.0, 0.25, 1.35),
        "Right_Shoulder": (0.0, -0.25, 1.35),
        "Left_Elbow": (0.0, 0.52, 1.2),
        "Right_Elbow": (0.0, -0.52, 1.2),
        "Left_Wrist": (0.0, 0.75, 1.15),
        "Right_Wrist": (0.0, -0.75, 1.15),
    }
    human = {
        name: (np.asarray(position), identity.copy())
        for name, position in positions.items()
    }
    joints, _, root_quaternion = solver.retarget(human)
    assert joints.shape == (29,)
    assert np.isfinite(joints).all()
    assert np.isclose(np.linalg.norm(root_quaternion), 1.0)
    for value, joint_id in zip(joints, solver._joint_ids):
        if solver.model.jnt_limited[joint_id]:
            low, high = solver.model.jnt_range[joint_id]
            assert low - 1.0e-6 <= value <= high + 1.0e-6
