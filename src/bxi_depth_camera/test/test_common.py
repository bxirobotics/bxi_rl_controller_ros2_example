import numpy as np
import pytest

from bxi_depth_camera.common import (
    ImageRectifier,
    camera_base_topic,
    camera_name_token,
    parse_profile,
    topic_token,
)
from bxi_depth_camera.orbbec import _depth_mm_from_sdk_data


def test_parse_profile_supports_realsense_style_and_auto():
    assert parse_profile("0,0,0", "profile").automatic
    profile = parse_profile("640x480x30", "profile")
    assert (profile.width, profile.height, profile.fps) == (640, 480, 30)


@pytest.mark.parametrize("value", ["640x480", "640x0x30", "bad"])
def test_parse_profile_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_profile(value, "profile")


def test_topic_token_is_stable_for_vendor_serials():
    assert topic_token("831612073525") == "SN_831612073525"
    assert topic_token("CP0F4630000L") == "SN_CP0F4630000L"


def test_topic_token_does_not_rewrite_serials():
    with pytest.raises(ValueError, match="letters, digits, or underscores"):
        topic_token("CP1-2.3")


def test_logical_camera_name_matches_simulation_name():
    assert camera_name_token("head_depth_camera") == "head_depth_camera"
    assert camera_base_topic("hardware", "head_depth_camera") == "/hardware/head_depth_camera"
    with pytest.raises(ValueError, match="logical camera name"):
        camera_name_token("1_head_depth_camera")


def test_orbbec_depth_is_converted_to_millimeters():
    raw = np.array([[0, 1, 123, 65535]], dtype=np.uint16)
    converted = _depth_mm_from_sdk_data(raw.tobytes(), 4, 1, 0.5)
    np.testing.assert_array_equal(
        converted, np.array([[0, 0, 62, 32768]], dtype=np.uint16)
    )


def test_zero_distortion_rectification_is_an_identity():
    rectifier = ImageRectifier()
    image = np.arange(48, dtype=np.uint16).reshape(6, 8)

    result = rectifier.rectify(
        image,
        fx=8.0,
        fy=8.0,
        cx=3.5,
        cy=2.5,
        distortion=(0.0,) * 5,
        distortion_model="brown_conrady",
        nearest=True,
    )

    np.testing.assert_array_equal(result, image)
    assert result.flags.c_contiguous


@pytest.mark.parametrize(
    "model", ["brown_conrady", "inverse_brown_conrady"]
)
def test_depth_rectification_preserves_integer_samples_and_caches_maps(model):
    rectifier = ImageRectifier()
    image = np.arange(192, dtype=np.uint16).reshape(12, 16)
    parameters = dict(
        fx=15.0,
        fy=15.0,
        cx=7.5,
        cy=5.5,
        distortion=(0.15, -0.02, 0.001, -0.001, 0.0),
        distortion_model=model,
        nearest=True,
    )

    first = rectifier.rectify(image, **parameters)
    second = rectifier.rectify(image, **parameters)

    assert first.shape == image.shape
    assert first.dtype == np.uint16
    assert set(np.unique(first)).issubset({0, *np.unique(image)})
    np.testing.assert_array_equal(second, first)
    assert len(rectifier._maps) == 1
