"""Canonical simulation launch; keeps example_launch_vibration.py compatible."""

import importlib.util
from pathlib import Path


def generate_launch_description():
    compatibility_launch = Path(__file__).with_name("example_launch_vibration.py")
    spec = importlib.util.spec_from_file_location(
        "example_launch_vibration_compat",
        compatibility_launch,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % compatibility_launch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()
