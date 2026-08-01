import math

import pytest

from bxi_depth_camera.inspect import CameraIdentity, Intrinsics, _matching


def test_intrinsics_compute_horizontal_and_vertical_fov():
    intrinsics = Intrinsics(
        fx=320.0,
        fy=240.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    )

    assert intrinsics.hfov_deg == pytest.approx(90.0)
    assert intrinsics.vfov_deg == pytest.approx(90.0)


def test_invalid_intrinsics_do_not_produce_non_finite_fov():
    intrinsics = Intrinsics(0.0, math.nan, 0.0, 0.0, 640, 480)

    assert intrinsics.hfov_deg == 0.0
    assert intrinsics.vfov_deg == 0.0


def test_serial_filter_keeps_backend_identity():
    first = CameraIdentity("realsense", "123", "D435", "usb-1")
    second = CameraIdentity("orbbec", "ABC", "Gemini 335", "usb-2")
    devices = {first.key: first, second.key: second}

    assert _matching(devices, "123") == {first.key: first}
    assert _matching(devices, "") == devices
