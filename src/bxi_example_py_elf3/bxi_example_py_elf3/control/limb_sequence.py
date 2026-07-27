"""Collision-margined full-range whole-body joint inspection planning.

Arm and leg ranges, mirrored directions and compact elbow/knee postures are
adapted from ``~/BXI/robot_limb_testing/bxi_rl_controller_ros2_example``.
The three waist joints are inserted between the arm and leg groups.
"""

from dataclasses import dataclass

import numpy as np

from .elf3 import (
    JOINT_NAMES,
    JOINT_POSITION_MAX,
    JOINT_POSITION_MIN,
    validate_joint_vector,
)


MODEL_COLLISION_FREE_RANGE_DEG = {
    "l_hip_y_joint": (-116.9, 155.8),
    "l_hip_x_joint": (-17.0, 142.0),
    "l_hip_z_joint": (-165.0, 165.0),
    "l_knee_y_joint": (-5.0, 150.0),
    "l_ankle_y_joint": (-48.2, 45.0),
    "l_ankle_x_joint": (-20.0, 20.0),
    "r_hip_y_joint": (-116.9, 155.8),
    "r_hip_x_joint": (-142.0, 17.0),
    "r_hip_z_joint": (-165.0, 165.0),
    "r_knee_y_joint": (-5.0, 150.0),
    "r_ankle_y_joint": (-48.2, 45.0),
    "r_ankle_x_joint": (-20.0, 20.0),
    "l_shoulder_y_joint": (-165.0, 165.0),
    "l_shoulder_x_joint": (-5.9, 175.0),
    "l_shoulder_z_joint": (-30.0, 165.0),
    "l_elbow_y_joint": (-55.0, 67.9),
    "l_wrist_x_joint": (-165.0, 165.0),
    "l_wrist_y_joint": (-75.0, 75.0),
    "l_wrist_z_joint": (-45.0, 45.0),
    "r_shoulder_y_joint": (-165.0, 165.0),
    "r_shoulder_x_joint": (-175.0, 5.9),
    "r_shoulder_z_joint": (-165.0, 30.0),
    "r_elbow_y_joint": (-55.0, 67.9),
    "r_wrist_x_joint": (-165.0, 165.0),
    "r_wrist_y_joint": (-75.0, 75.0),
    "r_wrist_z_joint": (-45.0, 45.0),
    # Waist ranges use the model mechanical range as the scan boundary. The
    # configured collision reserve shrinks both ends, and every command is
    # still checked online by the MuJoCo collision guard.
    "waist_y_joint": (-30.0, 30.0),
    "waist_x_joint": (-15.0, 15.0),
    "waist_z_joint": (-165.0, 165.0),
}

COMPACT_POSTURE_DEG = {
    "shoulder_y_joint": ("elbow_y_joint", -45.0),
    "shoulder_x_joint": ("elbow_y_joint", -45.0),
    "shoulder_z_joint": ("elbow_y_joint", -15.0),
    "hip_y_joint": ("knee_y_joint", 30.0),
    "hip_x_joint": ("knee_y_joint", 135.0),
    "hip_z_joint": ("knee_y_joint", 5.0),
}

MIRRORED_PAIR_LOWER_BOUND_DEG = {
    "hip_x_joint": -8.0,
    "hip_z_joint": -60.5,
}


@dataclass(frozen=True)
class JointMotionGroup:
    category: str
    label: str
    joint_names: tuple


def _bilateral_group(category, label, suffix):
    return JointMotionGroup(
        category,
        label,
        ("l_" + suffix, "r_" + suffix),
    )


WHOLE_BODY_TEST_GROUPS = (
    _bilateral_group("arms", "左右腕旋转", "wrist_z_joint"),
    _bilateral_group("arms", "左右腕俯仰", "wrist_y_joint"),
    _bilateral_group("arms", "左右腕侧摆", "wrist_x_joint"),
    _bilateral_group("arms", "左右肘俯仰", "elbow_y_joint"),
    _bilateral_group("arms", "左右肩旋转", "shoulder_z_joint"),
    _bilateral_group("arms", "左右肩侧摆", "shoulder_x_joint"),
    _bilateral_group("arms", "左右肩俯仰", "shoulder_y_joint"),
    JointMotionGroup("torso", "腰部旋转", ("waist_z_joint",)),
    JointMotionGroup("torso", "腰部侧摆", ("waist_x_joint",)),
    JointMotionGroup("torso", "腰部俯仰", ("waist_y_joint",)),
    _bilateral_group("legs", "左右踝侧摆", "ankle_x_joint"),
    _bilateral_group("legs", "左右踝俯仰", "ankle_y_joint"),
    _bilateral_group("legs", "左右膝俯仰", "knee_y_joint"),
    _bilateral_group("legs", "左右髋旋转", "hip_z_joint"),
    _bilateral_group("legs", "左右髋侧摆", "hip_x_joint"),
    _bilateral_group("legs", "左右髋俯仰", "hip_y_joint"),
)


def joint_suffix(joint_name):
    return joint_name[2:] if joint_name[:2] in ("l_", "r_") else joint_name


def uses_opposite_bilateral_sign(joint_name):
    suffix = joint_suffix(joint_name)
    return "_x_joint" in suffix or "_z_joint" in suffix


def compact_posture(active_joint_names):
    result = {}
    for name in active_joint_names:
        configuration = COMPACT_POSTURE_DEG.get(joint_suffix(name))
        if configuration is None:
            continue
        companion_suffix, angle_deg = configuration
        result[name[:2] + companion_suffix] = float(np.deg2rad(angle_deg))
    return result


def safe_joint_range(
    joint_name,
    collision_margin_deg=5.0,
    mechanical_margin_deg=2.0,
):
    index = JOINT_NAMES.index(joint_name)
    model_low, model_high = MODEL_COLLISION_FREE_RANGE_DEG[joint_name]
    mechanical_low = JOINT_POSITION_MIN[index] + np.deg2rad(
        mechanical_margin_deg
    )
    mechanical_high = JOINT_POSITION_MAX[index] - np.deg2rad(
        mechanical_margin_deg
    )
    collision_low = np.deg2rad(model_low + collision_margin_deg)
    collision_high = np.deg2rad(model_high - collision_margin_deg)
    low = max(float(mechanical_low), float(collision_low))
    high = min(float(mechanical_high), float(collision_high))
    if high <= low:
        raise ValueError("%s collision-safe range is empty" % joint_name)
    return low, high


def build_safe_ranges(
    groups=WHOLE_BODY_TEST_GROUPS,
    collision_margin_deg=5.0,
    mechanical_margin_deg=2.0,
):
    result = {
        name: safe_joint_range(
            name,
            collision_margin_deg,
            mechanical_margin_deg,
        )
        for group in groups
        for name in group.joint_names
    }
    for group in groups:
        if len(group.joint_names) != 2:
            continue
        left, right = group.joint_names
        left_low, left_high = result[left]
        right_low, right_high = result[right]
        if uses_opposite_bilateral_sign(left):
            low = max(left_low, -right_high)
            high = min(left_high, -right_low)
            boundary = MIRRORED_PAIR_LOWER_BOUND_DEG.get(joint_suffix(left))
            if boundary is not None:
                low = max(
                    low,
                    np.deg2rad(boundary + collision_margin_deg),
                )
            if high <= low:
                raise ValueError("%s mirrored safe range is empty" % left)
            result[left] = (float(low), float(high))
            result[right] = (float(-high), float(-low))
        else:
            low = max(left_low, right_low)
            high = min(left_high, right_high)
            if high <= low:
                raise ValueError("%s bilateral safe range is empty" % left)
            result[left] = result[right] = (float(low), float(high))
    return result


def full_range_waypoints(center_positions, group, target_ranges):
    """Return compact-posture, low, high, center and unfold waypoints."""
    center = validate_joint_vector("joint test center", center_positions)
    posture = compact_posture(group.joint_names)
    motion_names = tuple(dict.fromkeys(group.joint_names + tuple(posture)))
    folded = center.copy()
    for name, value in posture.items():
        folded[JOINT_NAMES.index(name)] = value

    result = []
    if posture:
        result.append(folded.copy())

    low_target = folded.copy()
    high_target = folded.copy()
    if (
        len(group.joint_names) == 2
        and uses_opposite_bilateral_sign(group.joint_names[0])
    ):
        left, right = group.joint_names
        low_target[JOINT_NAMES.index(left)] = target_ranges[left][0]
        low_target[JOINT_NAMES.index(right)] = target_ranges[right][1]
        high_target[JOINT_NAMES.index(left)] = target_ranges[left][1]
        high_target[JOINT_NAMES.index(right)] = target_ranges[right][0]
    else:
        for name in group.joint_names:
            low_target[JOINT_NAMES.index(name)] = target_ranges[name][0]
            high_target[JOINT_NAMES.index(name)] = target_ranges[name][1]
    result.extend((low_target, high_target, folded.copy()))
    if posture:
        result.append(center.copy())
    return motion_names, tuple(result)


def velocity_limited_duration(
    start_positions,
    target_positions,
    motion_names,
    minimum_move_sec=1.5,
    range_speed_deg_s=20.0,
):
    start = validate_joint_vector("segment start", start_positions)
    target = validate_joint_vector("segment target", target_positions)
    indices = [JOINT_NAMES.index(name) for name in motion_names]
    travel_deg = max(
        (abs(float(np.rad2deg(target[i] - start[i]))) for i in indices),
        default=0.0,
    )
    # A quintic minimum-jerk trajectory has peak normalized slope 1.875.
    return max(
        float(minimum_move_sec),
        1.875 * travel_deg / float(range_speed_deg_s),
    )
