from pathlib import Path

import yaml

from bxi_depth_camera.manager import (
    CONFIG_PARAMETER_DEFAULTS,
    MANAGER_PARAMETER_DEFAULTS,
)


def test_default_yaml_matches_manager_defaults_and_deployment_profiles():
    path = Path(__file__).parents[1] / "config" / "default.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = document["/depth_camera_manager"]["ros__parameters"]
    expected = {**MANAGER_PARAMETER_DEFAULTS, **CONFIG_PARAMETER_DEFAULTS}

    deployment_profiles = {
        "depth_module.depth_profile": "848x480x30",
        "rgb_camera.color_profile": "1920,1080,30",
    }

    assert parameters.keys() == expected.keys()
    assert {
        name: value
        for name, value in parameters.items()
        if name not in deployment_profiles
    } == {
        name: value
        for name, value in expected.items()
        if name not in deployment_profiles
    }
    for name, value in deployment_profiles.items():
        assert parameters[name] == value
        assert expected[name] == "0,0,0"
