from pathlib import Path

import yaml

from bxi_depth_camera.manager import (
    CONFIG_PARAMETER_DEFAULTS,
    MANAGER_PARAMETER_DEFAULTS,
)


def test_default_yaml_matches_all_manager_defaults():
    path = Path(__file__).parents[1] / "config" / "default.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = document["/depth_camera_manager"]["ros__parameters"]
    expected = {**MANAGER_PARAMETER_DEFAULTS, **CONFIG_PARAMETER_DEFAULTS}

    assert parameters == expected
