"""NumPy/SciPy SMPL geometry used by the headless PICO manager.

The live manager only needs small rotation conversions and forward kinematics;
it does not run a Torch model.  Keeping this path NumPy-native makes the PICO
runtime substantially smaller and avoids a process-wide Torch dependency.
Quaternion arrays exposed by this module use scalar-first ``[w, x, y, z]``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


FloatArray = NDArray[np.float32]
_OUTPUT_JOINTS = np.concatenate((np.arange(22), np.array([39, 54])))
_SMPL_BASE_INVERSE_WXYZ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
_Y_UP_TO_Z_UP = Rotation.from_rotvec(
    np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64)
)


def compute_from_body_poses(
    parent_indices: list[int],
    body_poses: NDArray[np.generic],
) -> dict[str, FloatArray]:
    """Convert one XRT 24-joint body frame to the SONIC SMPL representation."""

    poses = np.asarray(body_poses)
    if poses.shape != (24, 7):
        raise ValueError(f"PICO body poses must have shape (24, 7), got {poses.shape}")
    if len(parent_indices) != 24:
        raise ValueError(
            f"PICO parent index list must contain 24 entries, got {len(parent_indices)}"
        )
    if not np.all(np.isfinite(poses)):
        raise ValueError("PICO body poses contain non-finite values")

    positions = poses[:, :3]
    # XRT stores [x, y, z, qx, qy, qz, qw].
    global_quats_xyzw = poses[:, [3, 4, 5, 6]]
    global_rotations = Rotation.from_quat(global_quats_xyzw)
    global_rotations = global_rotations * Rotation.from_euler("y", 180, degrees=True)

    local_rotvecs = np.empty((24, 3), dtype=np.float32)
    for index, parent in enumerate(parent_indices):
        rotation = (
            global_rotations[index]
            if parent == -1
            else global_rotations[parent].inv() * global_rotations[index]
        )
        local_rotvecs[index] = rotation.as_rotvec().astype(np.float32)

    body_pose = local_rotvecs[1:].reshape(1, -1)
    global_orient = local_rotvecs[0].reshape(1, 3)
    translation = positions[0].astype(np.float32, copy=False).reshape(1, 3)
    return process_smpl_joints(body_pose, global_orient, translation)


def process_smpl_joints(
    body_pose: NDArray[np.generic],
    global_orient: NDArray[np.generic],
    translation: NDArray[np.generic],
) -> dict[str, FloatArray]:
    """Compute the same values previously produced by the Torch helper path."""

    pose = np.asarray(body_pose, dtype=np.float32)
    orient = np.asarray(global_orient, dtype=np.float32)
    transl = np.asarray(translation, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 63:
        raise ValueError(f"SMPL body pose must have shape (N, >=63), got {pose.shape}")
    if orient.shape != (pose.shape[0], 3):
        raise ValueError(
            f"SMPL global orientation must have shape {(pose.shape[0], 3)}, "
            f"got {orient.shape}"
        )
    if transl.shape != (pose.shape[0], 3):
        raise ValueError(
            f"SMPL translation must have shape {(pose.shape[0], 3)}, got {transl.shape}"
        )

    root_y_up = Rotation.from_rotvec(orient)
    root_z_up = _Y_UP_TO_Z_UP * root_y_up
    root_z_up_rotvec = root_z_up.as_rotvec().astype(np.float32)
    joints = compute_human_joints(pose[:, :63], root_z_up_rotvec)

    base_inverse = Rotation.from_quat(_SMPL_BASE_INVERSE_WXYZ, scalar_first=True)
    root_robot = root_z_up * base_inverse
    root_robot_wxyz = root_robot.as_quat(scalar_first=True).astype(np.float32)

    local_joints = np.empty_like(joints)
    for batch_index in range(joints.shape[0]):
        local_joints[batch_index] = (
            root_robot[batch_index].inv().apply(joints[batch_index]).astype(np.float32)
        )
    root_matrix = root_robot.as_matrix().astype(np.float32)

    return {
        "smpl_pose": pose,
        "joints": joints,
        "smpl_joints_local": local_joints,
        "global_orient_quat": root_robot_wxyz,
        "global_orient_6d": root_matrix[:, :, :2].reshape(pose.shape[0], 6),
        "adjusted_transl": transl,
    }


def compute_human_joints(
    body_pose: NDArray[np.generic],
    global_orient: NDArray[np.generic],
) -> FloatArray:
    """Run batched SMPL forward kinematics for the 24 joints SONIC consumes."""

    pose = np.asarray(body_pose, dtype=np.float32)
    orient = np.asarray(global_orient, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 63:
        raise ValueError(f"SMPL body pose must have shape (N, 63), got {pose.shape}")
    if orient.shape != (pose.shape[0], 3):
        raise ValueError(
            f"SMPL global orientation must have shape {(pose.shape[0], 3)}, "
            f"got {orient.shape}"
        )

    rest_joints, parents = _human_joint_info()
    zeros = np.zeros((pose.shape[0], 99), dtype=np.float32)
    full_pose = np.concatenate((orient, pose, zeros), axis=-1)
    rotation_matrices = (
        Rotation.from_rotvec(full_pose.reshape(-1, 3))
        .as_matrix()
        .astype(np.float32)
        .reshape(pose.shape[0], 55, 3, 3)
    )

    relative_joints = np.broadcast_to(
        rest_joints, (pose.shape[0], *rest_joints.shape)
    ).copy()
    relative_joints[:, 1:] -= rest_joints[parents[1:]]

    local_transforms = np.zeros((pose.shape[0], 55, 4, 4), dtype=np.float32)
    local_transforms[:, :, :3, :3] = rotation_matrices
    local_transforms[:, :, :3, 3] = relative_joints
    local_transforms[:, :, 3, 3] = 1.0

    global_transforms = np.empty_like(local_transforms)
    global_transforms[:, 0] = local_transforms[:, 0]
    for index in range(1, len(parents)):
        global_transforms[:, index] = np.matmul(
            global_transforms[:, parents[index]],
            local_transforms[:, index],
        )
    return np.take(global_transforms[:, :, :3, 3], _OUTPUT_JOINTS, axis=1)


@lru_cache(maxsize=1)
def _human_joint_info() -> tuple[FloatArray, NDArray[np.int64]]:
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "human" / "human_joints_info.npz"
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"SONIC human joint data is missing: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        joints = np.asarray(data["J"], dtype=np.float32)
        parents = np.asarray(data["parents"], dtype=np.int64)
    if joints.shape != (55, 3) or parents.shape != (55,):
        raise ValueError(
            "SONIC human joint data has invalid shapes: "
            f"J={joints.shape}, parents={parents.shape}"
        )
    if parents[0] != -1 or np.any(parents[1:] < 0):
        raise ValueError("SONIC human joint parent indices are invalid")
    joints.setflags(write=False)
    parents.setflags(write=False)
    return joints, parents


__all__ = ["compute_from_body_poses", "compute_human_joints", "process_smpl_joints"]
