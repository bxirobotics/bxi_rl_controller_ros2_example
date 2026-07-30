from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import bxi_example_py_elf3.bxi_example_demo as demo_module
import bxi_example_py_elf3.robot_states as robot_states_module
from bxi_example_py_elf3.bxi_example_demo import (
    BxiExample,
    dof_num,
    gripper_joint_name,
    joint_name,
)
from bxi_example_py_elf3.robot_states import SonicTeleopState


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _ActuatorCmds:
    def __init__(self):
        self.header = SimpleNamespace()


class _Logger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)


def _motor_context(*, enabled, left=0.0, right=0.0):
    publisher = _Publisher()
    now = SimpleNamespace(to_msg=lambda: "stamp")
    context = SimpleNamespace(
        sim_gripper_enabled=enabled,
        left_gripper_input=left,
        right_gripper_input=right,
        act_pub=publisher,
        get_clock=lambda: SimpleNamespace(now=lambda: now),
    )
    return context, publisher


def _send_motor_command(monkeypatch, *, enabled, left=0.0, right=0.0):
    monkeypatch.setattr(demo_module.bxiMsg, "ActuatorCmds", _ActuatorCmds)
    context, publisher = _motor_context(
        enabled=enabled,
        left=left,
        right=right,
    )
    body = np.arange(dof_num, dtype=np.float32)
    BxiExample.send_to_motor(context, body, body + 100.0, body + 200.0)
    assert len(publisher.messages) == 1
    return publisher.messages[0]


def test_standard_sim_command_remains_body_only(monkeypatch):
    message = _send_motor_command(monkeypatch, enabled=False)

    assert message.actuators_name == list(joint_name)
    assert len(message.pos) == dof_num
    assert len(message.vel) == dof_num
    assert len(message.torque) == dof_num
    assert len(message.kp) == dof_num
    assert len(message.kd) == dof_num


def test_gripper_sim_command_appends_four_clipped_trigger_targets(monkeypatch):
    context = SimpleNamespace(left_gripper_input=0.0, right_gripper_input=0.0)
    BxiExample.left_gripper_callback(context, SimpleNamespace(data=-0.25))
    BxiExample.right_gripper_callback(context, SimpleNamespace(data=1.25))

    message = _send_motor_command(
        monkeypatch,
        enabled=True,
        left=context.left_gripper_input,
        right=context.right_gripper_input,
    )

    assert message.actuators_name == list(joint_name) + list(gripper_joint_name)
    assert len(message.pos) == dof_num + 4
    assert message.pos[-4:] == pytest.approx([0.0, 0.0, 0.055, 0.055])
    assert message.kp[-4:] == [500.0] * 4
    assert message.kd[-4:] == [5.0] * 4


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "bad"])
def test_gripper_callbacks_ignore_non_finite_input(invalid):
    context = SimpleNamespace(left_gripper_input=0.25, right_gripper_input=0.75)

    BxiExample.left_gripper_callback(context, SimpleNamespace(data=invalid))
    BxiExample.right_gripper_callback(context, SimpleNamespace(data=invalid))

    assert context.left_gripper_input == 0.25
    assert context.right_gripper_input == 0.75


def test_gripper_opt_in_is_simulation_only(monkeypatch):
    monkeypatch.setenv("BXI_SIM_GRIPPER_ENABLE", "1")

    assert demo_module._sim_gripper_enabled("simulation/")
    assert not demo_module._sim_gripper_enabled("hardware/")
    assert not demo_module._sim_gripper_enabled("")


def test_state_machine_exposes_explicit_body_and_hardware_gripper_modes():
    config_path = (
        Path(__file__).resolve().parents[1] / "config" / "elf3_state_machine.yaml"
    )
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    body_event = config["remote_events"]["sonic_teleop_event"]
    gripper_event = config["remote_events"]["sonic_teleop_gripper_event"]
    assert body_event == {"slot": "btn_10", "value": 7}
    assert gripper_event == {"slot": "btn_10", "value": 8}

    state_configs = config["states"]
    body = SonicTeleopState(
        "sonic_teleop", 22, **state_configs["sonic_teleop"]["params"]
    )
    gripper = SonicTeleopState(
        "sonic_teleop_gripper",
        23,
        **state_configs["sonic_teleop_gripper"]["params"],
    )
    assert not body.hardware_gripper_requested
    assert gripper.hardware_gripper_requested
    assert list(state_configs)[-1] == "sonic_teleop_gripper"
    assert config["states"]["normal"]["transitions"]["on_event"][
        "sonic_teleop_gripper_event"
    ]["to"] == "sonic_teleop_gripper"


def test_hardware_can_gripper_mode_is_ignored_by_simulation():
    context = SimpleNamespace(
        topic_prefix="simulation/",
        create_subscription=lambda *_args, **_kwargs: pytest.fail(
            "simulation must not subscribe for the hardware CAN gripper"
        ),
        create_publisher=lambda *_args, **_kwargs: pytest.fail(
            "simulation must not publish hardware CAN packets"
        ),
    )
    state = SonicTeleopState(
        "sonic_teleop_gripper", 23, hardware_gripper=True
    )

    state.on_bind(context)

    assert not state.gripper_enabled


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "bad"])
def test_hardware_gripper_callbacks_ignore_non_finite_input(invalid):
    state = SonicTeleopState(
        "sonic_teleop_gripper", 23, hardware_gripper=True
    )
    state._gripper_session_active = True
    state.left_trigger = 0.25
    state.right_trigger = 0.75
    state.left_trigger_received_at = 1.0
    state.right_trigger_received_at = 2.0

    state.left_trigger_callback(SimpleNamespace(data=invalid))
    state.right_trigger_callback(SimpleNamespace(data=invalid))

    assert state.left_trigger == 0.25
    assert state.right_trigger == 0.75
    assert state.left_trigger_received_at == 1.0
    assert state.right_trigger_received_at == 2.0


class _HardwareGripperContext:
    topic_prefix = "hardware/"

    def __init__(self):
        self.logger = _Logger()
        self.publisher = _Publisher()
        self.subscriptions = []
        self.qpos = np.zeros(1)
        self.kp_last = np.zeros(1)
        self.kd_last = np.zeros(1)

    def get_logger(self):
        return self.logger

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = (msg_type, topic, callback, qos)
        self.subscriptions.append(subscription)
        return subscription

    def create_publisher(self, _msg_type, _topic, _qos):
        return self.publisher


def _bound_hardware_gripper(monkeypatch, clock):
    for name, value in {
        "BXI_SONIC_GRIPPER_LEFT_BUS": "5",
        "BXI_SONIC_GRIPPER_RIGHT_BUS": "6",
        "BXI_SONIC_GRIPPER_CAN_ID": "1",
        "BXI_SONIC_GRIPPER_KP": "20",
        "BXI_SONIC_GRIPPER_KD": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        robot_states_module.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        robot_states_module.bxiMsg,
        "CANFDPacket",
        type("CANFDPacket", (), {}),
        raising=False,
    )
    monkeypatch.setattr(
        robot_states_module.BxiMotor,
        "build_motor_packet",
        lambda bus, can_id, data: {
            "bus": bus,
            "can_id": can_id,
            "data": list(data),
        },
    )
    context = _HardwareGripperContext()
    state = SonicTeleopState(
        "sonic_teleop_gripper",
        23,
        hardware_gripper=True,
        gripper_input_timeout_s=0.2,
    )
    state.on_bind(context)
    state._start_gripper_session()
    return state, context


def _is_enter_packet(message):
    return message["data"] == robot_states_module.BxiMotor.enter_motor_mode()


def test_body_only_hardware_mode_never_creates_can_resources(monkeypatch):
    monkeypatch.setenv("BXI_SONIC_GRIPPER_LEFT_BUS", "not-an-integer")
    context = SimpleNamespace(
        topic_prefix="hardware/",
        create_subscription=lambda *_args, **_kwargs: pytest.fail(
            "body-only SONIC must not subscribe to gripper inputs"
        ),
        create_publisher=lambda *_args, **_kwargs: pytest.fail(
            "body-only SONIC must not create a CAN publisher"
        ),
    )
    state = SonicTeleopState("sonic_teleop", 22, hardware_gripper=False)

    state.on_bind(context)

    assert not state.gripper_enabled


def test_gripper_arms_after_fresh_released_inputs_and_enters_once(monkeypatch):
    clock = [10.0]
    state, context = _bound_hardware_gripper(monkeypatch, clock)

    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    assert state.gripper_armed
    first_cycle = context.publisher.messages
    assert [(message["bus"], message["can_id"]) for message in first_cycle] == [
        (5, 1),
        (6, 1),
        (5, 1),
        (6, 1),
    ]
    assert [_is_enter_packet(message) for message in first_cycle] == [
        True,
        True,
        False,
        False,
    ]

    clock[0] += 0.02
    state._update_gripper(context)

    assert sum(map(_is_enter_packet, context.publisher.messages)) == 2
    assert len(context.publisher.messages) == 6
    assert [message["bus"] for message in context.publisher.messages[-2:]] == [5, 6]


def test_gripper_exit_and_reentry_require_new_released_inputs(monkeypatch):
    clock = [15.0]
    state, context = _bound_hardware_gripper(monkeypatch, clock)
    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    clock[0] += 0.02
    state.left_trigger_callback(SimpleNamespace(data=0.8))
    state.right_trigger_callback(SimpleNamespace(data=0.6))
    state._update_gripper(context)
    before_exit = len(context.publisher.messages)
    received_before_exit = (
        state.left_trigger_received_at,
        state.right_trigger_received_at,
    )

    state.on_exit(context)
    state.left_trigger_callback(SimpleNamespace(data=0.1))
    state.right_trigger_callback(SimpleNamespace(data=0.2))
    state._update_gripper(context)

    assert not state._gripper_session_active
    assert not state.gripper_armed
    assert (state.left_trigger, state.right_trigger) == pytest.approx((0.8, 0.6))
    assert (
        state.left_trigger_received_at,
        state.right_trigger_received_at,
    ) == received_before_exit
    assert len(context.publisher.messages) == before_exit

    clock[0] += 0.02
    state._start_gripper_session()
    assert (state.left_trigger, state.right_trigger) == (0.0, 0.0)
    assert state.left_trigger_received_at is None
    assert state.right_trigger_received_at is None

    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)
    assert not state.gripper_armed
    assert len(context.publisher.messages) == before_exit

    state.right_trigger_callback(SimpleNamespace(data=0.4))
    state._update_gripper(context)
    assert not state.gripper_armed
    assert len(context.publisher.messages) == before_exit

    clock[0] += 0.02
    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    reentry_messages = context.publisher.messages[before_exit:]
    assert state.gripper_armed
    reentry_sequence = [
        (message["bus"], _is_enter_packet(message))
        for message in reentry_messages
    ]
    assert reentry_sequence == [
        (5, True),
        (6, True),
        (5, False),
        (6, False),
    ]
    assert sum(map(_is_enter_packet, context.publisher.messages)) == 4


def test_gripper_waits_for_both_triggers_to_be_released(monkeypatch):
    clock = [20.0]
    state, context = _bound_hardware_gripper(monkeypatch, clock)

    state.left_trigger_callback(SimpleNamespace(data=0.4))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    assert not state.gripper_armed
    assert context.publisher.messages == []
    assert context.logger.warnings == [
        "SONIC夹爪等待左右PICO trigger松开；夹爪尚未解锁"
    ]

    clock[0] += 0.02
    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    assert state.gripper_armed
    assert sum(map(_is_enter_packet, context.publisher.messages)) == 2


def test_stale_trigger_holds_last_position_and_logs_edges_once(monkeypatch):
    clock = [30.0]
    state, context = _bound_hardware_gripper(monkeypatch, clock)
    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    clock[0] += 0.05
    state.left_trigger_callback(SimpleNamespace(data=0.8))
    state.right_trigger_callback(SimpleNamespace(data=0.6))
    state._update_gripper(context)
    live_commands = [message["data"] for message in context.publisher.messages[-2:]]

    clock[0] += 0.21
    state._update_gripper(context)
    stale_commands = [message["data"] for message in context.publisher.messages[-2:]]
    state._update_gripper(context)

    assert stale_commands == live_commands
    assert state.left_trigger == pytest.approx(0.8)
    assert state.right_trigger == pytest.approx(0.6)
    assert context.logger.warnings.count(
        "SONIC夹爪PICO trigger断流：left,right；保持最后位置"
    ) == 1

    clock[0] += 0.01
    state.left_trigger_callback(SimpleNamespace(data=0.8))
    state.right_trigger_callback(SimpleNamespace(data=0.6))
    state._update_gripper(context)

    assert context.logger.infos.count(
        "SONIC夹爪PICO trigger已恢复：left,right"
    ) == 1


def test_stale_and_recovery_edges_are_reported_per_side(monkeypatch):
    clock = [40.0]
    state, context = _bound_hardware_gripper(monkeypatch, clock)
    state.left_trigger_callback(SimpleNamespace(data=0.0))
    state.right_trigger_callback(SimpleNamespace(data=0.0))
    state._update_gripper(context)

    clock[0] += 0.05
    state.left_trigger_callback(SimpleNamespace(data=0.8))
    state.right_trigger_callback(SimpleNamespace(data=0.6))
    state._update_gripper(context)
    live_commands = [message["data"] for message in context.publisher.messages[-2:]]

    clock[0] += 0.21
    state.right_trigger_callback(SimpleNamespace(data=0.6))
    state._update_gripper(context)
    held_commands = [
        message["data"] for message in context.publisher.messages[-2:]
    ]
    assert held_commands == live_commands
    state._update_gripper(context)

    clock[0] += 0.21
    state.left_trigger_callback(SimpleNamespace(data=0.8))
    state._update_gripper(context)
    state._update_gripper(context)

    clock[0] += 0.01
    state.right_trigger_callback(SimpleNamespace(data=0.6))
    state._update_gripper(context)

    assert context.logger.warnings.count(
        "SONIC夹爪PICO trigger断流：left；保持最后位置"
    ) == 1
    assert context.logger.warnings.count(
        "SONIC夹爪PICO trigger断流：right；保持最后位置"
    ) == 1
    assert context.logger.infos.count("SONIC夹爪PICO trigger已恢复：left") == 1
    assert context.logger.infos.count("SONIC夹爪PICO trigger已恢复：right") == 1


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gripper_release_threshold_rejects_non_finite_values(invalid):
    with pytest.raises(ValueError, match="gripper_release_threshold must be finite"):
        SonicTeleopState(
            "sonic_teleop_gripper",
            23,
            hardware_gripper=True,
            gripper_release_threshold=invalid,
        )


@pytest.mark.parametrize(
    ("variable", "invalid"),
    [
        ("BXI_SONIC_GRIPPER_LEFT_BUS", "not-an-integer"),
        ("BXI_SONIC_GRIPPER_RIGHT_BUS", "-1"),
        ("BXI_SONIC_GRIPPER_CAN_ID", "-1"),
        ("BXI_SONIC_GRIPPER_KP", "nan"),
        ("BXI_SONIC_GRIPPER_KD", "not-a-number"),
    ],
)
def test_invalid_hardware_gripper_config_disables_only_gripper(
    monkeypatch, variable, invalid
):
    for name, value in {
        "BXI_SONIC_GRIPPER_LEFT_BUS": "5",
        "BXI_SONIC_GRIPPER_RIGHT_BUS": "6",
        "BXI_SONIC_GRIPPER_CAN_ID": "1",
        "BXI_SONIC_GRIPPER_KP": "20",
        "BXI_SONIC_GRIPPER_KD": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(variable, invalid)
    context = _HardwareGripperContext()
    state = SonicTeleopState(
        "sonic_teleop_gripper", 23, hardware_gripper=True
    )

    state.on_bind(context)
    state._start_gripper_session()
    state._update_gripper(context)

    assert not state.gripper_enabled
    assert not state._gripper_session_active
    assert context.subscriptions == []
    assert context.publisher.messages == []
    assert not hasattr(state, "gripper_control_pub")
    assert len(context.logger.errors) == 1
    assert context.logger.errors[0].startswith(
        "SONIC gripper disabled: invalid config:"
    )


def test_mujoco_gripper_model_only_adds_the_expected_actuators_and_geometry():
    mujoco = pytest.importorskip("mujoco")
    data_dir = Path(__file__).resolve().parents[1] / "data/mujoco_simulation"
    standard = mujoco.MjModel.from_xml_path(str(data_dir / "elf3.xml"))
    gripper = mujoco.MjModel.from_xml_path(str(data_dir / "elf3_gripper.xml"))

    def names(model, object_type, count):
        return [
            mujoco.mj_id2name(model, object_type, index)
            for index in range(count)
        ]

    standard_actuators = names(
        standard, mujoco.mjtObj.mjOBJ_ACTUATOR, standard.nu
    )
    gripper_actuators = names(
        gripper, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper.nu
    )
    assert standard.nu == 29
    assert gripper.nu == 33
    assert set(gripper_actuators) == set(standard_actuators) | set(gripper_joint_name)

    for name in standard_actuators:
        standard_id = mujoco.mj_name2id(
            standard, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        gripper_id = mujoco.mj_name2id(
            gripper, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        np.testing.assert_allclose(
            gripper.actuator_ctrlrange[gripper_id],
            standard.actuator_ctrlrange[standard_id],
        )
        np.testing.assert_allclose(
            gripper.actuator_gear[gripper_id],
            standard.actuator_gear[standard_id],
        )

    standard_geoms = set(
        names(standard, mujoco.mjtObj.mjOBJ_GEOM, standard.ngeom)
    )
    gripper_geoms = set(names(gripper, mujoco.mjtObj.mjOBJ_GEOM, gripper.ngeom))
    assert standard_geoms <= gripper_geoms
    assert "head_collision" in gripper_geoms

    for name in gripper_joint_name:
        joint_id = mujoco.mj_name2id(gripper, mujoco.mjtObj.mjOBJ_JOINT, name)
        np.testing.assert_allclose(gripper.jnt_range[joint_id], [0.0, 0.055])


def _feedback_context():
    logger = _Logger()
    return SimpleNamespace(
        qpos=np.full(dof_num, -1.0, dtype=np.float64),
        qvel=np.full(dof_num, -2.0, dtype=np.float64),
        lock_in=nullcontext(),
        get_logger=lambda: logger,
    ), logger


def test_shuffled_feedback_is_restored_to_body_joint_order():
    names = list(reversed(joint_name + gripper_joint_name))
    positions = [float(index) for index in range(len(names))]
    velocities = [float(index + 100) for index in range(len(names))]
    message = SimpleNamespace(
        name=names,
        position=positions,
        velocity=velocities,
    )
    context, logger = _feedback_context()

    BxiExample.actuator_callback(context, message)

    expected_positions = [positions[names.index(name)] for name in joint_name]
    expected_velocities = [velocities[names.index(name)] for name in joint_name]
    np.testing.assert_array_equal(context.qpos, expected_positions)
    np.testing.assert_array_equal(context.qvel, expected_velocities)
    assert logger.errors == []


def test_feedback_length_mismatch_is_rejected_without_partial_update():
    message = SimpleNamespace(
        name=list(joint_name),
        position=[0.0] * dof_num,
        velocity=[0.0] * (dof_num - 1),
    )
    context, logger = _feedback_context()

    BxiExample.actuator_callback(context, message)

    np.testing.assert_array_equal(context.qpos, np.full(dof_num, -1.0))
    np.testing.assert_array_equal(context.qvel, np.full(dof_num, -2.0))
    assert len(logger.errors) == 1
    assert "velocities" in logger.errors[0]
