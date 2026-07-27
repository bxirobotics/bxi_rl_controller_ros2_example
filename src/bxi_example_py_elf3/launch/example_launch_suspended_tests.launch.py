"""Combined suspended running (X) and vibration (Y) simulation launch."""

import importlib.util

from ament_index_python.packages import get_package_share_path


def generate_launch_description():
    launch_path = (
        get_package_share_path("bxi_example_py_elf3")
        / "launch/example_launch_vibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "combined_simulation_launch_base",
        launch_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description(
        controller_executable_default=(
            "bxi_example_py_elf3_suspended_tests"
        ),
        controller_name_default="bxi_example_py_elf3_suspended_tests",
        joint_test_required_default="false",
        allow_hardware_without_joint_test_default="true",
    )
