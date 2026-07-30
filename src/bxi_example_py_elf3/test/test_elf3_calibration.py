from pathlib import Path

import numpy as np
import pytest

# Importing the vendored namespace adds its private root to sys.path, matching
# the packaged manager entry point without allowing it to shadow globally.
import bxi_example_py_elf3.sonic_pico.vendor  # noqa: F401
from bxi_example_py_elf3.sonic_pico import vendor as vendor_package
from gear_sonic.utils.teleop.calibration import create_calibration_provider


def test_elf3_native_calibration_fk_smoke():
    provider = create_calibration_provider()

    poses = provider.get_key_frame_poses(np.zeros(29, dtype=np.float64))

    assert provider.name == "elf3_native"
    assert provider.robot_model.num_dofs == 31
    assert set(poses) == {"left_wrist", "right_wrist", "torso"}
    for pose in poses.values():
        assert set(pose) == {
            "position",
            "orientation_xyzw",
            "orientation_wxyz",
        }
        assert pose["position"].shape == (3,)
        assert pose["orientation_xyzw"].shape == (4,)
        assert pose["orientation_wxyz"].shape == (4,)
        assert np.all(np.isfinite(pose["position"]))
        assert np.all(np.isfinite(pose["orientation_wxyz"]))
        assert np.linalg.norm(pose["orientation_wxyz"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "body_q",
    [
        np.zeros(28, dtype=np.float64),
        np.full(29, np.nan, dtype=np.float64),
    ],
)
def test_elf3_native_calibration_rejects_invalid_body_vectors(body_q):
    provider = create_calibration_provider()

    with pytest.raises(ValueError):
        provider.get_key_frame_poses(body_q)


def test_vendored_runtime_contains_no_alternate_robot_model_files():
    vendor_root = Path(vendor_package.__file__).parent / "gear_sonic"
    forbidden = [
        path
        for path in vendor_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in {".py", ".urdf"}
            and "g1" in path.name.lower()
        )
    ]

    assert forbidden == []
