import numpy as np

from bxi_example_py_elf3.robot_states import NormalState, SonicTeleopState


class SonicContext:
    current_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def __init__(self):
        self.motor_target = None
        self.orientation_checks = 0
        self.requested_states = []

    def is_orientation_unsafe(self, _quat):
        self.orientation_checks += 1
        raise AssertionError("sonic_teleop must not apply the global 60-degree gate")

    def request_state(self, state_name, trigger):
        self.requested_states.append((state_name, trigger))

    def set_motor_target(self, *frame):
        self.motor_target = frame


class UnsafeNormalContext:
    current_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def __init__(self):
        self.requested_states = []

    def is_orientation_unsafe(self, _quat):
        return True

    def request_state(self, state_name, trigger):
        self.requested_states.append((state_name, trigger))


def test_sonic_teleop_does_not_apply_the_global_60_degree_gate():
    state = SonicTeleopState("sonic_teleop", 23)
    ctx = SonicContext()
    frame = tuple(np.full(29, value, dtype=np.float32) for value in (1.0, 2.0, 3.0))
    state.get_motor_frame = lambda _ctx, _dt, _transition: frame
    state._update_gripper = lambda _ctx: None

    state.on_update(ctx, 0.02)

    assert ctx.orientation_checks == 0
    assert ctx.requested_states == []
    assert ctx.motor_target is not None
    for actual, expected in zip(ctx.motor_target, frame):
        np.testing.assert_array_equal(actual, expected)


def test_normal_state_keeps_global_orientation_gate():
    state = NormalState("normal", 1)
    ctx = UnsafeNormalContext()
    state.get_motor_frame = lambda *_args: (_ for _ in ()).throw(
        AssertionError("unsafe normal state must return before inference")
    )

    state.on_update(ctx, 0.02)

    assert ctx.requested_states == [("zero_torque", "safety")]
