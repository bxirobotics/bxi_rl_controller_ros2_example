"""ELF3-specific supplemental robot metadata for the 31-DoF URDF."""

from dataclasses import dataclass

import numpy as np

from gear_sonic.data.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)


@dataclass
class Elf3SupplementalInfo(RobotSupplementalInfo):
    """Joint ordering, groups, limits, and frame names for ELF3.

    The ELF3 URDF has 31 joints. The SONIC policy/controller contract contains
    the 29 waist, leg, and arm joints listed in ``body_actuated_joints``. The
    two head joints are deliberately excluded from that vector and therefore
    remain at the URDF neutral position when a 29-element body vector is
    expanded by ``RobotModel.get_configuration_from_actuated_joints``.
    """

    def __init__(self):
        name = "ELF3_31DoF"

        # This order is the authoritative 29-DoF SONIC ELF3 policy order.
        body_actuated_joints = [
            # Waist
            "waist_y_joint",
            "waist_x_joint",
            "waist_z_joint",
            # Left leg
            "l_hip_y_joint",
            "l_hip_x_joint",
            "l_hip_z_joint",
            "l_knee_y_joint",
            "l_ankle_y_joint",
            "l_ankle_x_joint",
            # Right leg
            "r_hip_y_joint",
            "r_hip_x_joint",
            "r_hip_z_joint",
            "r_knee_y_joint",
            "r_ankle_y_joint",
            "r_ankle_x_joint",
            # Left arm
            "l_shoulder_y_joint",
            "l_shoulder_x_joint",
            "l_shoulder_z_joint",
            "l_elbow_y_joint",
            "l_wrist_x_joint",
            "l_wrist_y_joint",
            "l_wrist_z_joint",
            # Right arm
            "r_shoulder_y_joint",
            "r_shoulder_x_joint",
            "r_shoulder_z_joint",
            "r_elbow_y_joint",
            "r_wrist_x_joint",
            "r_wrist_y_joint",
            "r_wrist_z_joint",
        ]

        # The selected ELF3 model has no actuated finger/hand joints.
        left_hand_actuated_joints = []
        right_hand_actuated_joints = []

        # Limits copied from model_data/elf3/elf3-dof31.urdf.
        joint_limits = {
            "waist_y_joint": [-0.5236, 0.5236],
            "waist_x_joint": [-0.2618, 0.2618],
            "waist_z_joint": [-2.8798, 2.8798],
            "l_hip_y_joint": [-2.8798, 2.8798],
            "l_hip_x_joint": [-0.48869, 3.0543],
            "l_hip_z_joint": [-2.8798, 2.8798],
            "l_knee_y_joint": [-0.087266, 2.618],
            "l_ankle_y_joint": [-0.87266, 0.7854],
            "l_ankle_x_joint": [-0.34907, 0.34907],
            "r_hip_y_joint": [-2.8798, 2.8798],
            "r_hip_x_joint": [-3.0543, 0.48869],
            "r_hip_z_joint": [-2.8798, 2.8798],
            "r_knee_y_joint": [-0.087266, 2.618],
            "r_ankle_y_joint": [-0.87266, 0.7854],
            "r_ankle_x_joint": [-0.34907, 0.34907],
            "l_shoulder_y_joint": [-2.8798, 2.8798],
            "l_shoulder_x_joint": [-0.34907, 3.0543],
            "l_shoulder_z_joint": [-2.8798, 2.8798],
            "l_elbow_y_joint": [-0.95993, 1.6581],
            "l_wrist_x_joint": [-2.8798, 2.8798],
            "l_wrist_y_joint": [-1.309, 1.309],
            "l_wrist_z_joint": [-0.7854, 0.7854],
            "r_shoulder_y_joint": [-2.8798, 2.8798],
            "r_shoulder_x_joint": [-3.0543, 0.34907],
            "r_shoulder_z_joint": [-2.8798, 2.8798],
            "r_elbow_y_joint": [-0.95993, 1.6581],
            "r_wrist_x_joint": [-2.8798, 2.8798],
            "r_wrist_y_joint": [-1.309, 1.309],
            "r_wrist_z_joint": [-0.7854, 0.7854],
            "head_z_joint": [-1.57, 1.57],
            "head_y_joint": [-0.785, 0.785],
        }

        # Subgroups are defined before composite groups because RobotModel
        # resolves nested groups in insertion order during initialization.
        joint_groups = {
            "waist": {
                "joints": ["waist_y_joint", "waist_x_joint", "waist_z_joint"],
                "groups": [],
            },
            "left_leg": {
                "joints": [
                    "l_hip_y_joint",
                    "l_hip_x_joint",
                    "l_hip_z_joint",
                    "l_knee_y_joint",
                    "l_ankle_y_joint",
                    "l_ankle_x_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "r_hip_y_joint",
                    "r_hip_x_joint",
                    "r_hip_z_joint",
                    "r_knee_y_joint",
                    "r_ankle_y_joint",
                    "r_ankle_x_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            "left_arm": {
                "joints": [
                    "l_shoulder_y_joint",
                    "l_shoulder_x_joint",
                    "l_shoulder_z_joint",
                    "l_elbow_y_joint",
                    "l_wrist_x_joint",
                    "l_wrist_y_joint",
                    "l_wrist_z_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "r_shoulder_y_joint",
                    "r_shoulder_x_joint",
                    "r_shoulder_z_joint",
                    "r_elbow_y_joint",
                    "r_wrist_x_joint",
                    "r_wrist_y_joint",
                    "r_wrist_z_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            "head": {
                "joints": ["head_z_joint", "head_y_joint"],
                "groups": [],
            },
            "left_hand": {"joints": [], "groups": []},
            "right_hand": {"joints": [], "groups": []},
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
            "whole_robot": {"joints": [], "groups": ["body", "head"]},
        }

        # Generic semantic names used by RobotModel default/calibration poses.
        # ELF3 axis suffixes name the physical rotation axes: X=roll, Y=pitch,
        # Z=yaw.
        joint_name_mapping = {
            "waist_pitch": "waist_y_joint",
            "waist_roll": "waist_x_joint",
            "waist_yaw": "waist_z_joint",
            "shoulder_pitch": {
                "left": "l_shoulder_y_joint",
                "right": "r_shoulder_y_joint",
            },
            "shoulder_roll": {
                "left": "l_shoulder_x_joint",
                "right": "r_shoulder_x_joint",
            },
            "shoulder_yaw": {
                "left": "l_shoulder_z_joint",
                "right": "r_shoulder_z_joint",
            },
            "elbow_pitch": {
                "left": "l_elbow_y_joint",
                "right": "r_elbow_y_joint",
            },
            "wrist_roll": {
                "left": "l_wrist_x_joint",
                "right": "r_wrist_x_joint",
            },
            "wrist_pitch": {
                "left": "l_wrist_y_joint",
                "right": "r_wrist_y_joint",
            },
            "wrist_yaw": {
                "left": "l_wrist_z_joint",
                "right": "r_wrist_z_joint",
            },
            "head_yaw": "head_z_joint",
            "head_pitch": "head_y_joint",
        }

        root_frame_name = "torso_link"
        hand_frame_names = {
            "left": "l_wrist_z_link",
            "right": "r_wrist_z_link",
        }

        calibration_joint_q = {
            "elbow_pitch": {"left": 0.0, "right": 0.0},
        }

        # No ELF3-specific hand-tracking correction is established yet. Keep
        # the model metadata neutral rather than inheriting another robot's correction.
        hand_rotation_correction = np.eye(3, dtype=np.float64)

        # Keep RobotModel's default configuration at the URDF neutral pose.
        # XYAB calibration also explicitly supplies a zero 29-DoF body vector.
        default_joint_q = {}
        teleop_upper_body_motion_scale = 1.0

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            calibration_joint_q=calibration_joint_q,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )

