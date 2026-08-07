from types import SimpleNamespace

import pytest
from rclpy.context import Context
from rclpy.parameter import Parameter

from bxi_depth_camera.common import DeviceDescriptor
from bxi_depth_camera.manager import CameraManager, CameraSupervisor


class _Logger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _Executor:
    def __init__(self):
        self.removed = []

    def remove_node(self, worker):
        self.removed.append(worker)


class _Worker:
    def __init__(self, stale=False, logical_name="head_depth_camera"):
        self.stale = stale
        self.logical_name = logical_name
        self.destroyed = False

    def is_stale(self, _now):
        return self.stale

    def destroy_node(self):
        self.destroyed = True


def _supervisor(discovered):
    supervisor = CameraSupervisor.__new__(CameraSupervisor)
    logger = _Logger()
    supervisor.manager = SimpleNamespace(
        retry_interval=2.0,
        get_logger=lambda: logger,
        take_pending_restarts=lambda: set(),
        camera_name_for=lambda _serial: "head_depth_camera",
    )
    supervisor.executor = _Executor()
    supervisor.workers = {}
    supervisor.failures = {}
    supervisor.discover = lambda: discovered
    return supervisor, logger


def test_reconcile_removes_disconnected_worker():
    supervisor, _logger = _supervisor({})
    key = ("realsense", "123")
    worker = _Worker()
    supervisor.workers[key] = worker

    supervisor.reconcile()

    assert worker.destroyed
    assert supervisor.executor.removed == [worker]


def test_reconcile_retries_stale_worker_without_immediate_restart():
    descriptor = DeviceDescriptor("orbbec", "ABC", "Gemini 335")
    supervisor, _logger = _supervisor({descriptor.key: descriptor})
    worker = _Worker(stale=True)
    supervisor.workers[descriptor.key] = worker
    started = []
    supervisor._start = lambda item, now: started.append((item, now))

    supervisor.reconcile()

    assert worker.destroyed
    assert descriptor.key in supervisor.failures
    assert started == []


def test_reconcile_starts_newly_discovered_camera():
    descriptor = DeviceDescriptor("realsense", "123", "D435")
    supervisor, _logger = _supervisor({descriptor.key: descriptor})
    started = []
    supervisor._start = lambda item, now: started.append((item, now))

    supervisor.reconcile()

    assert started and started[0][0] == descriptor


def test_reconcile_restarts_only_requested_camera():
    first = DeviceDescriptor("realsense", "123", "D435")
    second = DeviceDescriptor("orbbec", "ABC", "Gemini 335")
    supervisor, _logger = _supervisor({first.key: first, second.key: second})
    first_worker = _Worker(logical_name="head_depth_camera")
    second_worker = _Worker(logical_name="rear_cam")
    supervisor.workers = {
        first.key: first_worker,
        second.key: second_worker,
    }
    supervisor.manager.take_pending_restarts = lambda: {"head_depth_camera"}
    supervisor.manager.camera_name_for = lambda serial: {
        "123": "head_depth_camera",
        "ABC": "rear_cam",
    }[serial]
    started = []
    supervisor._start = lambda item, now: started.append((item, now))

    supervisor.reconcile()

    assert first_worker.destroyed
    assert not second_worker.destroyed
    assert [item[0] for item in started] == [first]


def test_reconcile_restarts_worker_when_single_camera_fallback_changes():
    descriptor = DeviceDescriptor("realsense", "123", "D435")
    supervisor, _logger = _supervisor({descriptor.key: descriptor})
    worker = _Worker(logical_name="SN_123")
    supervisor.workers[descriptor.key] = worker
    started = []
    supervisor._start = lambda item, now: started.append((item, now))

    supervisor.reconcile()

    assert worker.destroyed
    assert [item[0] for item in started] == [descriptor]


def test_single_unmapped_camera_uses_head_camera_name():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        manager.update_discovered_serials(["349422070502"])

        assert manager.camera_name_for("349422070502") == "head_depth_camera"
    finally:
        manager.destroy_node()
        context.shutdown()


def test_multiple_unmapped_cameras_keep_serial_fallback_names():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        manager.update_discovered_serials(["349422070502", "CP0F4630000L"])

        assert manager.camera_name_for("349422070502") == "SN_349422070502"
        assert manager.camera_name_for("CP0F4630000L") == "SN_CP0F4630000L"
    finally:
        manager.destroy_node()
        context.shutdown()


def test_explicit_mapping_wins_over_single_camera_fallback():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(
        context=context,
        parameter_overrides=[
            Parameter("cameras.rear_cam.serial_no", value="349422070502")
        ],
    )
    try:
        manager.update_discovered_serials(["349422070502"])

        assert manager.camera_name_for("349422070502") == "rear_cam"
    finally:
        manager.destroy_node()
        context.shutdown()


def test_per_camera_profile_override_is_dynamic():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        name = "cameras.body_depth_camera.depth_module.depth_profile"
        result = manager.set_parameters_atomically(
            [
                Parameter(
                    "cameras.body_depth_camera.serial_no",
                    value="CP0F4630000L",
                ),
                Parameter(
                    name,
                    value="848x480x30",
                ),
            ]
        )

        assert result.successful, result.reason
        profile = manager.config_for("CP0F4630000L").depth_profile
        assert (profile.width, profile.height, profile.fps) == (848, 480, 30)
        assert manager.camera_name_for("CP0F4630000L") == "body_depth_camera"
        assert manager.config_for("349422070502").depth_profile.automatic
        assert manager.camera_name_for("349422070502") == "SN_349422070502"
        assert manager.take_pending_restarts() == {"*", "body_depth_camera"}

        deleted = manager.set_parameters_atomically(
            [Parameter(name, type_=Parameter.Type.NOT_SET)]
        )
        assert deleted.successful, deleted.reason
        assert manager.config_for("CP0F4630000L").depth_profile.automatic
        assert manager.take_pending_restarts() == {"body_depth_camera"}
    finally:
        manager.destroy_node()
        context.shutdown()


def test_invalid_per_camera_parameter_is_rejected():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        result = manager.set_parameters_atomically(
            [Parameter("cameras.body_depth_camera.depth_module.unknown", value="bad")]
        )

        assert not result.successful
        assert "invalid per-camera parameter" in result.reason
    finally:
        manager.destroy_node()
        context.shutdown()


def test_rectification_can_be_enabled_per_stream_and_camera():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        result = manager.set_parameters_atomically(
            [
                Parameter(
                    "cameras.body_depth_camera.depth_module.rectification.enable",
                    value=True,
                ),
                Parameter(
                    "cameras.body_depth_camera.infra2.rectification.enable",
                    value=True,
                ),
            ]
        )

        assert result.successful, result.reason
        config = manager.config_for("unmapped_serial")
        assert not config.rectify_depth
        mapped = manager.set_parameters_atomically(
            [
                Parameter(
                    "cameras.body_depth_camera.serial_no",
                    value="ABC",
                )
            ]
        )
        assert mapped.successful, mapped.reason
        config = manager.config_for("ABC")
        assert config.rectify_depth
        assert config.rectify_infra2
        assert config.rectify_color
        assert not config.rectify_infra1
    finally:
        manager.destroy_node()
        context.shutdown()


def test_depth_alignment_can_be_enabled_per_camera():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        result = manager.set_parameters_atomically(
            [
                Parameter(
                    "cameras.head_depth_camera.serial_no",
                    value="ABC",
                ),
                Parameter(
                    "cameras.head_depth_camera.align_depth.enable",
                    value=True,
                ),
            ]
        )

        assert result.successful, result.reason
        assert manager.config_for("ABC").align_depth
        assert not manager.config_for("unmapped_serial").align_depth
        assert manager.take_pending_restarts() == {"*", "head_depth_camera"}
    finally:
        manager.destroy_node()
        context.shutdown()


def test_depth_alignment_requires_depth_and_color_streams():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        result = manager.set_parameters_atomically(
            [
                Parameter("align_depth.enable", value=True),
                Parameter("enable_color", value=False),
            ]
        )

        assert not result.successful
        assert "requires both enable_depth and enable_color" in result.reason
        assert not manager.config.align_depth
        assert manager.config.enable_color
    finally:
        manager.destroy_node()
        context.shutdown()


def test_pointcloud_can_be_enabled_and_limited_per_camera():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        result = manager.set_parameters_atomically(
            [
                Parameter("cameras.head_depth_camera.serial_no", value="ABC"),
                Parameter(
                    "cameras.head_depth_camera.pointcloud.enable", value=True
                ),
                Parameter(
                    "cameras.head_depth_camera.pointcloud.ordered_pc", value=True
                ),
                Parameter(
                    "cameras.head_depth_camera.pointcloud.max_fps", value=5.0
                ),
            ]
        )

        assert result.successful, result.reason
        config = manager.config_for("ABC")
        assert config.pointcloud_enabled
        assert config.pointcloud_ordered
        assert config.pointcloud_max_fps == pytest.approx(5.0)
        assert not manager.config_for("unmapped_serial").pointcloud_enabled
    finally:
        manager.destroy_node()
        context.shutdown()


def test_pointcloud_requires_depth_and_positive_rate():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        no_depth = manager.set_parameters_atomically(
            [
                Parameter("pointcloud.enable", value=True),
                Parameter("enable_depth", value=False),
            ]
        )
        assert not no_depth.successful
        assert "requires enable_depth" in no_depth.reason

        bad_rate = manager.set_parameters_atomically(
            [Parameter("pointcloud.max_fps", value=0.0)]
        )
        assert not bad_rate.successful
        assert "greater than zero" in bad_rate.reason
    finally:
        manager.destroy_node()
        context.shutdown()


def test_one_serial_cannot_be_assigned_to_two_logical_cameras():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(context=context)
    try:
        result = manager.set_parameters_atomically(
            [
                Parameter("cameras.body_depth_camera.serial_no", value="ABC"),
                Parameter("cameras.rear_cam.serial_no", value="ABC"),
            ]
        )

        assert not result.successful
        assert "assigned to both" in result.reason
    finally:
        manager.destroy_node()
        context.shutdown()


def test_deployment_overrides_apply_before_camera_start():
    context = Context()
    context.init(initialize_logging=False)
    manager = CameraManager(
        context=context,
        parameter_overrides=[
            Parameter("cameras.body_depth_camera.serial_no", value="349422070502"),
            Parameter(
                "cameras.body_depth_camera.depth_module.depth_profile",
                value="640x480x30",
            ),
        ],
    )
    try:
        assert manager.camera_name_for("349422070502") == "body_depth_camera"
        profile = manager.config_for("349422070502").depth_profile
        assert (profile.width, profile.height, profile.fps) == (640, 480, 30)
        assert manager.take_pending_restarts() == set()
    finally:
        manager.destroy_node()
        context.shutdown()
