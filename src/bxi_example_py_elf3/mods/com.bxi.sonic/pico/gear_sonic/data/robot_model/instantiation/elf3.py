"""Factory function to instantiate the 31-DoF ELF3 RobotModel from URDF."""

from pathlib import Path

from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.data.robot_model.supplemental_info.elf3 import Elf3SupplementalInfo


def instantiate_elf3_robot_model() -> RobotModel:
    """Instantiate ELF3 with the 29-DoF SONIC body-joint contract.

    The underlying URDF contains 31 joints. ``Elf3SupplementalInfo`` exposes
    the 29 waist/leg/arm joints as body actuators while leaving the two head
    joints at their neutral positions when body configurations are expanded.
    """
    model_data_dir = Path(__file__).resolve().parent.parent / "model_data" / "elf3"
    urdf_path = model_data_dir / "elf3-dof31.urdf"

    if not urdf_path.is_file():
        raise FileNotFoundError(f"ELF3 URDF not found: {urdf_path}")

    robot_model_supplemental_info = Elf3SupplementalInfo()
    return RobotModel(
        str(urdf_path),
        str(model_data_dir),
        supplemental_info=robot_model_supplemental_info,
    )

