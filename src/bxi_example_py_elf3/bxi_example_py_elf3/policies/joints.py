"""Named joint layouts and parameters owned by the built-in ELF3 policies."""

from bxi_example_py_elf3.framework.joints import JointLayout, JointParameterSet


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


# Every row is: (joint name, default position, kp, kd, action scale).
# Keeping the name beside its values makes the model order reviewable without
# translating between several parallel numeric arrays.
ELF3_ISAAC_PARAMETERS = JointParameterSet.from_rows(
    ELF3_ISAAC_JOINTS,
    (
        ("l_shoulder_y_joint", 0.2, 54.224, 3.452, 0.231),
        ("r_shoulder_y_joint", 0.2, 54.224, 3.452, 0.231),
        ("waist_y_joint", 0.0, 108.448, 6.904, 0.231),
        ("l_shoulder_x_joint", 0.2, 54.224, 3.452, 0.231),
        ("r_shoulder_x_joint", -0.2, 54.224, 3.452, 0.231),
        ("waist_x_joint", 0.0, 162.672, 10.356, 0.154),
        ("l_shoulder_z_joint", 0.0, 16.747, 1.066, 0.373),
        ("r_shoulder_z_joint", 0.0, 16.747, 1.066, 0.373),
        ("waist_z_joint", 0.0, 176.421, 11.231, 0.213),
        ("l_elbow_y_joint", 0.6, 54.224, 3.452, 0.231),
        ("r_elbow_y_joint", 0.6, 54.224, 3.452, 0.231),
        ("l_hip_y_joint", -0.3, 176.421, 11.231, 0.213),
        ("r_hip_y_joint", -0.3, 176.421, 11.231, 0.213),
        ("l_wrist_x_joint", 0.0, 16.747, 1.066, 0.373),
        ("r_wrist_x_joint", 0.0, 16.747, 1.066, 0.373),
        ("l_hip_x_joint", 0.0, 176.421, 11.231, 0.213),
        ("r_hip_x_joint", 0.0, 176.421, 11.231, 0.213),
        ("l_wrist_y_joint", 0.0, 16.747, 1.066, 0.373),
        ("r_wrist_y_joint", 0.0, 16.747, 1.066, 0.373),
        ("l_hip_z_joint", 0.0, 54.224, 3.452, 0.231),
        ("r_hip_z_joint", 0.0, 54.224, 3.452, 0.231),
        ("l_wrist_z_joint", 0.0, 16.747, 1.066, 0.373),
        ("r_wrist_z_joint", 0.0, 16.747, 1.066, 0.373),
        ("l_knee_y_joint", 0.6, 176.421, 11.231, 0.213),
        ("r_knee_y_joint", 0.6, 176.421, 11.231, 0.213),
        ("l_ankle_y_joint", -0.3, 33.493, 2.132, 0.373),
        ("r_ankle_y_joint", -0.3, 33.493, 2.132, 0.373),
        ("l_ankle_x_joint", 0.0, 21.771, 1.386, 0.230),
        ("r_ankle_x_joint", 0.0, 21.771, 1.386, 0.230),
    ),
)

__all__ = [
    "ELF3_ISAAC_JOINTS",
    "ELF3_ISAAC_PARAMETERS",
    "ELF3_LOWER_BODY_JOINTS",
    "ELF3_POLICY_JOINTS",
]
