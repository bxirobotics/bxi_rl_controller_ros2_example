"""Joint layouts owned by the built-in ELF3 policies."""

from bxi_example_py_elf3.framework.joints import JointLayout


ELF3_POLICY_JOINTS = JointLayout(
    (
        "waist_y_joint",
        "waist_x_joint",
        "waist_z_joint",
        "l_hip_y_joint",
        "l_hip_x_joint",
        "l_hip_z_joint",
        "l_knee_y_joint",
        "l_ankle_y_joint",
        "l_ankle_x_joint",
        "r_hip_y_joint",
        "r_hip_x_joint",
        "r_hip_z_joint",
        "r_knee_y_joint",
        "r_ankle_y_joint",
        "r_ankle_x_joint",
        "l_shoulder_y_joint",
        "l_shoulder_x_joint",
        "l_shoulder_z_joint",
        "l_elbow_y_joint",
        "l_wrist_x_joint",
        "l_wrist_y_joint",
        "l_wrist_z_joint",
        "r_shoulder_y_joint",
        "r_shoulder_x_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
        "r_wrist_x_joint",
        "r_wrist_y_joint",
        "r_wrist_z_joint",
    ),
    label="ELF3 29-joint policy",
)

ELF3_LOWER_BODY_JOINTS = ELF3_POLICY_JOINTS.select(
    ELF3_POLICY_JOINTS.names[:15],
    label="ELF3 waist and legs policy",
)

ELF3_ISAAC_JOINTS = ELF3_POLICY_JOINTS.select(
    (
        "l_shoulder_y_joint",
        "r_shoulder_y_joint",
        "waist_y_joint",
        "l_shoulder_x_joint",
        "r_shoulder_x_joint",
        "waist_x_joint",
        "l_shoulder_z_joint",
        "r_shoulder_z_joint",
        "waist_z_joint",
        "l_elbow_y_joint",
        "r_elbow_y_joint",
        "l_hip_y_joint",
        "r_hip_y_joint",
        "l_wrist_x_joint",
        "r_wrist_x_joint",
        "l_hip_x_joint",
        "r_hip_x_joint",
        "l_wrist_y_joint",
        "r_wrist_y_joint",
        "l_hip_z_joint",
        "r_hip_z_joint",
        "l_wrist_z_joint",
        "r_wrist_z_joint",
        "l_knee_y_joint",
        "r_knee_y_joint",
        "l_ankle_y_joint",
        "r_ankle_y_joint",
        "l_ankle_x_joint",
        "r_ankle_x_joint",
    ),
    label="ELF3 Isaac policy",
)

__all__ = [
    "ELF3_ISAAC_JOINTS",
    "ELF3_LOWER_BODY_JOINTS",
    "ELF3_POLICY_JOINTS",
]
