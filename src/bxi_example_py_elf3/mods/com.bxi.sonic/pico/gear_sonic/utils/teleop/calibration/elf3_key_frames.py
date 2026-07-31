"""Display-independent ELF3 key-frame forward kinematics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as sRot


ELF3_ANCHOR_FRAME = "waist_z_link"
ELF3_LEFT_WRIST_FRAME = "l_wrist_z_link"
ELF3_RIGHT_WRIST_FRAME = "r_wrist_z_link"

# Offsets are expressed in each frame's local coordinates. Keep them explicit
# even while zero so a future, authoritative ELF3 end-effector contract has a
# single well-defined place to supply wrist/tool offsets.
ELF3_KEY_FRAME_OFFSETS = {
    "left_wrist": np.zeros(3, dtype=np.float64),
    "right_wrist": np.zeros(3, dtype=np.float64),
    "torso": np.zeros(3, dtype=np.float64),
}


def _se3_to_pose_dict(placement, local_offset: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert a Pinocchio SE3 to the provider pose schema."""
    position = placement.translation + placement.rotation @ local_offset
    quat_xyzw = sRot.from_matrix(placement.rotation).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return {
        "position": np.asarray(position, dtype=np.float64).copy(),
        "orientation_xyzw": np.asarray(quat_xyzw, dtype=np.float64).copy(),
        "orientation_wxyz": np.asarray(quat_wxyz, dtype=np.float64).copy(),
    }


def get_elf3_key_frame_poses(
    robot_model,
    q: np.ndarray | None = None,
    apply_offset: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return ELF3 wrists relative to ``waist_z_link``.

    Args:
        robot_model: Shared ``RobotModel`` instance built from the ELF3 URDF.
        q: Full 31-DoF Pinocchio configuration. ``None`` uses the model default.
        apply_offset: Apply the local offsets in ``ELF3_KEY_FRAME_OFFSETS``.

    Returns:
        ``left_wrist``, ``right_wrist``, and ``torso`` pose dictionaries. All
        poses are expressed in the anchor-local coordinate system. ``torso``
        represents the selected anchor itself and is therefore identity.
    """
    if q is None:
        q = robot_model.default_body_pose
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (robot_model.num_dofs,):
        raise ValueError(
            f"ELF3 FK expected full q shape ({robot_model.num_dofs},), got {q.shape}"
        )
    if not np.all(np.isfinite(q)):
        raise ValueError("ELF3 FK configuration contains non-finite values")

    robot_model.cache_forward_kinematics(q, auto_clip=False)
    world_anchor = robot_model.frame_placement(ELF3_ANCHOR_FRAME)
    world_left = robot_model.frame_placement(ELF3_LEFT_WRIST_FRAME)
    world_right = robot_model.frame_placement(ELF3_RIGHT_WRIST_FRAME)

    anchor_left = world_anchor.inverse() * world_left
    anchor_right = world_anchor.inverse() * world_right
    anchor_identity = world_anchor.inverse() * world_anchor

    zero = np.zeros(3, dtype=np.float64)
    offsets = ELF3_KEY_FRAME_OFFSETS if apply_offset else {
        "left_wrist": zero,
        "right_wrist": zero,
        "torso": zero,
    }
    return {
        "left_wrist": _se3_to_pose_dict(anchor_left, offsets["left_wrist"]),
        "right_wrist": _se3_to_pose_dict(anchor_right, offsets["right_wrist"]),
        "torso": _se3_to_pose_dict(anchor_identity, offsets["torso"]),
    }

