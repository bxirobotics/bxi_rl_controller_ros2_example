import inspect
from pathlib import Path

import numpy as np

from bxi_example_py_elf3.control.elf3 import (
    DOF_NUM,
    JOINT_KD,
    JOINT_KP,
    JOINT_NAMES,
    JOINT_NOMINAL_POS,
    JOINT_POSITION_MAX,
    JOINT_POSITION_MIN,
    SUSPENDED_RUN_NOMINAL_POS,
    position_limit_violations,
)
from bxi_example_py_elf3.control.remote import RemoteButtonEdge
from bxi_example_py_elf3.control.limb_sequence import (
    WHOLE_BODY_TEST_GROUPS,
    build_safe_ranges,
    compact_posture,
    full_range_waypoints,
    velocity_limited_duration,
)
from bxi_example_py_elf3.control.trajectory import (
    JointTrajectory,
    minimum_jerk_progress,
)
from bxi_example_py_elf3.suspended_states import (
    SuspendedLimbTestState,
    SuspendedRunningState,
    SuspendedVibrationState,
    create_button_states,
)


def test_shared_joint_configuration_is_consistent():
    assert DOF_NUM == 29
    assert len(JOINT_NAMES) == DOF_NUM
    for vector in (
        JOINT_KP,
        JOINT_KD,
        JOINT_NOMINAL_POS,
        JOINT_POSITION_MIN,
        JOINT_POSITION_MAX,
        SUSPENDED_RUN_NOMINAL_POS,
    ):
        assert vector.shape == (DOF_NUM,)
        assert np.all(np.isfinite(vector))
    assert not position_limit_violations(JOINT_NOMINAL_POS)
    assert SUSPENDED_RUN_NOMINAL_POS[16] == 0.2
    assert SUSPENDED_RUN_NOMINAL_POS[23] == -0.2


def test_toggle_button_ignores_initial_state_and_activates_on_each_change():
    button = RemoteButtonEdge("toggle", resync_sec=0.5)
    assert not button.update(1, 1.0)
    assert not button.update(1, 1.1)
    assert button.update(0, 1.2)
    assert not button.update(0, 1.3)
    assert button.update(1, 1.4)


def test_momentary_button_only_activates_on_rising_edge():
    button = RemoteButtonEdge("momentary", resync_sec=0.5)
    assert not button.update(0, 1.0)
    assert button.update(1, 1.1)
    assert not button.update(1, 1.2)
    assert not button.update(0, 1.3)


def test_button_resynchronizes_after_publisher_gap():
    button = RemoteButtonEdge("toggle", resync_sec=0.5)
    assert not button.update(0, 1.0)
    assert button.update(1, 1.1)
    assert not button.update(0, 2.0)
    assert button.update(1, 2.1)


def test_x_and_y_use_independent_toggle_edges():
    run_button = RemoteButtonEdge("toggle", resync_sec=0.5)
    vibration_button = RemoteButtonEdge("toggle", resync_sec=0.5)
    assert not run_button.update(0, 1.0)
    assert not vibration_button.update(0, 1.0)
    assert run_button.update(1, 1.1)
    assert not vibration_button.update(0, 1.1)
    assert not run_button.update(1, 1.2)
    assert vibration_button.update(1, 1.2)


def test_suspended_button_states_have_independent_remote_fields():
    states = create_button_states()
    assert isinstance(states[0], SuspendedRunningState)
    assert isinstance(states[1], SuspendedVibrationState)
    assert isinstance(states[2], SuspendedLimbTestState)
    assert [state.button for state in states] == ["X", "Y", "A"]
    assert [state.message_field for state in states] == [
        "btn_9",
        "btn_10",
        "btn_7",
    ]
    assert len({state.name for state in states}) == len(states)
    assert len({state.message_field for state in states}) == len(states)
    state_file = Path(inspect.getsourcefile(SuspendedRunningState))
    assert "mods" in state_file.parts
    assert "com.bxi.suspended_tests" in state_file.parts


def test_minimum_jerk_blend_has_clamped_endpoints_and_is_monotonic():
    samples = [minimum_jerk_progress(index / 20.0) for index in range(21)]
    assert minimum_jerk_progress(-1.0) == 0.0
    assert minimum_jerk_progress(2.0) == 1.0
    assert samples == sorted(samples)


def test_whole_body_full_range_order_and_mirrored_motion():
    groups = WHOLE_BODY_TEST_GROUPS
    assert [group.category for group in groups].count("arms") == 7
    assert [group.category for group in groups].count("torso") == 3
    assert [group.category for group in groups].count("legs") == 6
    assert groups[0].joint_names == (
        "l_wrist_z_joint",
        "r_wrist_z_joint",
    )
    assert groups[-1].joint_names == (
        "l_hip_y_joint",
        "r_hip_y_joint",
    )
    safe_ranges = build_safe_ranges()
    motion_names, wrist_waypoints = full_range_waypoints(
        JOINT_NOMINAL_POS,
        groups[0],
        safe_ranges,
    )
    left = JOINT_NAMES.index("l_wrist_z_joint")
    right = JOINT_NAMES.index("r_wrist_z_joint")
    assert motion_names == groups[0].joint_names
    assert np.rad2deg(wrist_waypoints[0][left]) == -40.0
    assert np.rad2deg(wrist_waypoints[0][right]) == 40.0
    assert np.rad2deg(wrist_waypoints[1][left]) == 40.0
    assert np.rad2deg(wrist_waypoints[1][right]) == -40.0


def test_full_range_shoulder_motion_uses_compact_elbows_and_unfolds():
    safe_ranges = build_safe_ranges()
    shoulder_group = WHOLE_BODY_TEST_GROUPS[6]
    motion_names, waypoints = full_range_waypoints(
        np.zeros(DOF_NUM), shoulder_group, safe_ranges
    )
    left_elbow = JOINT_NAMES.index("l_elbow_y_joint")
    right_elbow = JOINT_NAMES.index("r_elbow_y_joint")
    assert compact_posture(shoulder_group.joint_names) == {
        "l_elbow_y_joint": np.deg2rad(-45.0),
        "r_elbow_y_joint": np.deg2rad(-45.0),
    }
    assert "l_elbow_y_joint" in motion_names
    assert np.rad2deg(waypoints[0][left_elbow]) == -45.0
    assert np.rad2deg(waypoints[0][right_elbow]) == -45.0
    np.testing.assert_allclose(waypoints[-1], np.zeros(DOF_NUM))


def test_full_range_duration_limits_minimum_jerk_peak_speed():
    start = np.zeros(DOF_NUM)
    target = start.copy()
    target[JOINT_NAMES.index("waist_z_joint")] = np.deg2rad(160.0)
    duration = velocity_limited_duration(
        start,
        target,
        ("waist_z_joint",),
        minimum_move_sec=1.5,
        range_speed_deg_s=20.0,
    )
    assert duration == 15.0


def test_trajectory_rejects_position_limit_violation():
    positions = JOINT_NOMINAL_POS.reshape(1, -1).copy()
    positions[0, 0] = JOINT_POSITION_MAX[0] + 0.01
    try:
        JointTrajectory(positions, "test", 50.0)
    except ValueError as exc:
        assert JOINT_NAMES[0] in str(exc)
    else:
        raise AssertionError("position-limit violation was accepted")


def test_trajectory_playback_wraps_without_aliasing_source_data():
    positions = np.vstack(
        [JOINT_NOMINAL_POS, JOINT_NOMINAL_POS + 0.001]
    )
    trajectory = JointTrajectory(positions, "test", 50.0)
    assert trajectory.next_is_first_frame
    first = trajectory.next()
    assert not trajectory.next_is_first_frame
    first[0] = 99.0
    np.testing.assert_allclose(trajectory.next(), positions[1])
    assert trajectory.next_is_first_frame
    np.testing.assert_allclose(trajectory.next(), positions[0])
