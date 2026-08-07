from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time

from bxi_depth_camera.common import (
    CameraNode,
    ImageRectifier,
    camera_base_topic,
    camera_name_token,
    parse_profile,
    topic_token,
)
from bxi_depth_camera.orbbec import OrbbecCamera, _depth_mm_from_sdk_data
from bxi_depth_camera.realsense import RealSenseCamera


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


def _orbbec_alignment_camera(process):
    camera = OrbbecCamera.__new__(OrbbecCamera)
    camera._align_to_color = SimpleNamespace(process=process)
    camera._pub_aligned_depth = object()
    camera._pub_aligned_depth_info = object()
    camera.descriptor = SimpleNamespace(serial="CP0F4630000L")
    camera.alignment_logs = []
    camera.log_throttled = lambda *args, **kwargs: camera.alignment_logs.append(
        (args, kwargs)
    )
    return camera


def test_orbbec_alignment_waits_for_complete_frameset():
    called = []
    camera = _orbbec_alignment_camera(lambda frame: called.append(frame))
    frameset = SimpleNamespace(
        get_depth_frame=lambda: object(),
        get_color_frame=lambda: None,
    )

    camera._publish_aligned_depth(frameset, object())

    assert called == []
    assert camera.alignment_logs[0][0][0] == "align-incomplete-frameset"


def test_orbbec_alignment_no_output_does_not_fail_whole_camera_frame():
    frame = SimpleNamespace(get_width=lambda: 848, get_height=lambda: 480)
    color = SimpleNamespace(get_width=lambda: 1920, get_height=lambda: 1080)
    camera = _orbbec_alignment_camera(lambda _frameset: None)
    frameset = SimpleNamespace(
        get_depth_frame=lambda: frame,
        get_color_frame=lambda: color,
    )

    camera._publish_aligned_depth(frameset, object())

    assert camera.alignment_logs[0][0][0] == "align-no-output"


def test_unordered_pointcloud_drops_invalid_and_untextured_points():
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
        dtype=np.uint8,
    )

    message = CameraNode.pointcloud_message(
        vertices,
        width=2,
        height=2,
        frame_id="camera_depth_optical_frame",
        stamp=Time(),
        ordered=False,
        allow_no_texture_points=False,
        colors=colors,
        texture_valid=np.array([True, True, False, True]),
    )

    assert message.width == 2
    assert message.height == 1
    assert message.point_step == 16
    assert message.is_dense
    assert [field.name for field in message.fields] == ["x", "y", "z", "rgb"]
    packed = np.frombuffer(
        message.data,
        dtype=np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
        ),
    )
    np.testing.assert_allclose(packed["x"], [1.0, 7.0])
    np.testing.assert_array_equal(
        packed["rgb"].view(np.uint32),
        [(10 << 16) | (20 << 8) | 30, (100 << 16) | (110 << 8) | 120],
    )


def test_ordered_pointcloud_preserves_grid_and_uses_nan_for_invalid_depth():
    vertices = np.array(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    colors = np.full((4, 3), 255, dtype=np.uint8)

    message = CameraNode.pointcloud_message(
        vertices,
        width=2,
        height=2,
        frame_id="camera_depth_optical_frame",
        stamp=Time(),
        ordered=True,
        allow_no_texture_points=True,
        colors=colors,
        texture_valid=np.array([True, True, False, True]),
    )

    assert message.width == 2
    assert message.height == 2
    assert not message.is_dense
    packed = np.frombuffer(
        message.data,
        dtype=np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
        ),
    )
    assert np.isnan(packed["z"][1])
    assert packed["z"][2] == pytest.approx(6.0)
    assert packed["rgb"].view(np.uint32)[2] == 0


def test_realsense_pointcloud_maps_bgr_texture_to_rgb():
    points = SimpleNamespace(
        get_vertices=lambda: np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        ),
        get_texture_coordinates=lambda: np.array(
            [[0.25, 0.25], [0.75, 0.75]], dtype=np.float32
        ),
    )
    mapped = []
    pointcloud = SimpleNamespace(
        map_to=lambda frame: mapped.append(frame),
        calculate=lambda _frame: points,
    )
    depth = SimpleNamespace(get_width=lambda: 2, get_height=lambda: 1)
    color_data = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    color = SimpleNamespace(get_data=lambda: color_data)
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera._pointcloud = pointcloud
    published = []
    camera.publish_pointcloud = lambda *args, **kwargs: published.append(
        (args, kwargs)
    )

    camera._publish_pointcloud(
        SimpleNamespace(get_depth_frame=lambda: depth),
        SimpleNamespace(get_color_frame=lambda: color),
        Time(),
    )

    assert mapped == [color]
    np.testing.assert_array_equal(
        published[0][1]["colors"],
        np.array([[3, 2, 1], [12, 11, 10]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(published[0][1]["texture_valid"], [True, True])


def test_orbbec_rgb_pointcloud_uses_meters_and_six_column_layout():
    calls = {}

    class _PointCloudFilter:
        def set_create_point_format(self, value):
            calls["format"] = value

        def set_position_data_scaled(self, value):
            calls["scale"] = value

        def calculate(self, source):
            calls["source"] = source
            return np.array(
                [[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], dtype=np.float32
            )

    depth = SimpleNamespace(
        get_depth_scale=lambda: 1.0,
        get_width=lambda: 1,
        get_height=lambda: 1,
    )
    color = object()
    frameset = SimpleNamespace(
        get_depth_frame=lambda: depth,
        get_color_frame=lambda: color,
    )
    camera = OrbbecCamera.__new__(OrbbecCamera)
    camera._pointcloud_filter = _PointCloudFilter()
    camera._ob = SimpleNamespace(
        OBFormat=SimpleNamespace(RGB_POINT="rgb", POINT="xyz")
    )
    published = []
    camera.publish_pointcloud = lambda *args, **kwargs: published.append(
        (args, kwargs)
    )

    camera._publish_pointcloud(frameset, Time())

    assert calls == {"format": "rgb", "scale": 0.001, "source": frameset}
    np.testing.assert_array_equal(
        published[0][1]["colors"], np.array([[10, 20, 30]], dtype=np.uint8)
    )
    np.testing.assert_allclose(published[0][0][0], [[1.0, 2.0, 3.0]])


def test_publisher_gate_accepts_either_image_or_camera_info_subscriber():
    zero = SimpleNamespace(get_subscription_count=lambda: 0)
    one = SimpleNamespace(get_subscription_count=lambda: 1)

    assert not CameraNode.publishers_requested(None, zero)
    assert CameraNode.publishers_requested(zero, one)


def test_video_consumer_gate_includes_pointcloud_and_camera_info():
    zero = SimpleNamespace(get_subscription_count=lambda: 0)
    one = SimpleNamespace(get_subscription_count=lambda: 1)
    camera = CameraNode.__new__(CameraNode)
    for name in (
        "_pub_depth",
        "_pub_depth_info",
        "_pub_color",
        "_pub_color_info",
        "_pub_aligned_depth",
        "_pub_aligned_depth_info",
        "_pub_infra1",
        "_pub_infra1_info",
        "_pub_infra2",
        "_pub_infra2_info",
        "_pub_pointcloud",
    ):
        setattr(camera, name, zero)

    assert not camera.video_consumers_requested()
    camera._pub_depth_info = one
    assert camera.video_consumers_requested()
    camera._pub_depth_info = zero
    camera._pub_pointcloud = one
    assert camera.video_consumers_requested()


def test_realsense_sdk_callback_does_not_wake_executor_without_consumers():
    camera = RealSenseCamera.__new__(RealSenseCamera)
    marked = []
    camera.mark_frame = lambda: marked.append(True)
    camera.video_consumers_requested = lambda: False
    camera._frame_guard = SimpleNamespace(
        trigger=lambda: pytest.fail("executor should not be woken")
    )
    frame = SimpleNamespace(
        is_frameset=lambda: True,
        as_frameset=lambda: pytest.fail("frame should not be retained"),
    )

    camera._on_frame(frame)

    assert marked == [True]


def test_orbbec_sdk_callback_does_not_wake_executor_without_consumers():
    camera = OrbbecCamera.__new__(OrbbecCamera)
    marked = []
    camera.mark_frame = lambda: marked.append(True)
    camera.video_consumers_requested = lambda: False
    camera._frame_guard = SimpleNamespace(
        trigger=lambda: pytest.fail("executor should not be woken")
    )

    camera._on_video_frames(object())

    assert marked == [True]


def test_orbbec_imu_callback_does_not_decode_without_consumers():
    camera = OrbbecCamera.__new__(OrbbecCamera)
    marked = []
    camera.mark_frame = lambda: marked.append(True)
    camera._pub_accel = object()
    camera._pub_gyro = object()
    camera.publishers_requested = lambda *_publishers: False
    frameset = SimpleNamespace(
        get_accel_frame=lambda: pytest.fail("accel should not be decoded"),
        get_gyro_frame=lambda: pytest.fail("gyro should not be decoded"),
    )

    camera._on_imu_frames(frameset)

    assert marked == [True]


def _disable_all_camera_consumers(camera):
    camera.depth_requested = lambda: False
    camera.color_requested = lambda: False
    camera.aligned_depth_requested = lambda: False
    camera.infra1_requested = lambda: False
    camera.infra2_requested = lambda: False
    camera.pointcloud_requested = lambda: False


def test_realsense_skips_entire_video_hot_path_without_subscribers():
    camera = RealSenseCamera.__new__(RealSenseCamera)
    _disable_all_camera_consumers(camera)
    camera.get_clock = lambda: pytest.fail("clock should not be read")
    camera._process_depth_frameset = lambda _frame: pytest.fail(
        "depth filters should not run"
    )

    camera._publish_frameset(object())


def test_orbbec_skips_entire_video_hot_path_without_subscribers():
    camera = OrbbecCamera.__new__(OrbbecCamera)
    _disable_all_camera_consumers(camera)
    camera.get_clock = lambda: pytest.fail("clock should not be read")
    camera._publish_aligned_depth = lambda *_args: pytest.fail(
        "alignment should not run"
    )

    camera._publish_frameset(object())


def test_realsense_alignment_subscriber_runs_required_depth_processing_only():
    camera = RealSenseCamera.__new__(RealSenseCamera)
    _disable_all_camera_consumers(camera)
    camera.aligned_depth_requested = lambda: True
    camera.config = SimpleNamespace(enable_depth=True)
    camera.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time())
    )
    filtered = object()
    camera._process_depth_frameset = lambda _frame: filtered
    published = []
    camera._publish_aligned_depth = lambda *args: published.append(args)
    source = object()

    camera._publish_frameset(source)

    assert published == [(filtered, source, Time())]


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
