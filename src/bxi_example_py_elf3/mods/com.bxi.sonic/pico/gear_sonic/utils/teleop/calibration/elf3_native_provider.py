"""ELF3-native calibration provider using the shared RobotModel stack."""

from __future__ import annotations

import numpy as np

from gear_sonic.data.robot_model.instantiation import instantiate_elf3_robot_model
from gear_sonic.utils.teleop.calibration.elf3_key_frames import (
    get_elf3_key_frame_poses,
)


class Elf3NativeCalibrationProvider:
    """Expose ELF3 anchor-local wrist FK behind the calibration provider API."""

    name = "elf3_native"
    dof = 29

    def __init__(self, robot_model=None):
        if robot_model is None:
            robot_model = instantiate_elf3_robot_model()
        self.robot_model = robot_model

        if self.robot_model.num_dofs != 31:
            raise ValueError(
                f"ELF3 native calibration requires a 31-DoF model, "
                f"got {self.robot_model.num_dofs}"
            )
        body_joints = self.robot_model.supplemental_info.body_actuated_joints
        if len(body_joints) != self.dof:
            raise ValueError(
                f"ELF3 native calibration requires {self.dof} body joints, "
                f"got {len(body_joints)}"
            )

    @classmethod
    def from_package_data(cls) -> "Elf3NativeCalibrationProvider":
        """Compatibility factory using the packaged ELF3 RobotModel assets."""
        return cls()

    def get_key_frame_poses(
        self,
        body_q: np.ndarray | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return ELF3 wrist/anchor poses for a 29-DoF SONIC body vector."""
        if body_q is None:
            robot_q = None
        else:
            body_q = np.asarray(body_q, dtype=np.float64)
            if body_q.shape != (self.dof,):
                raise ValueError(
                    f"ELF3 native calibration expected body_q shape ({self.dof},), "
                    f"got {body_q.shape}"
                )
            if not np.all(np.isfinite(body_q)):
                raise ValueError("ELF3 native calibration body_q contains non-finite values")

            robot_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q
            )

        return get_elf3_key_frame_poses(self.robot_model, q=robot_q)

