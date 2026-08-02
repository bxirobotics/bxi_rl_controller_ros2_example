from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .common import CameraConfig, CameraNode, DeviceDescriptor


def discover_realsense(rs) -> tuple[DeviceDescriptor, ...]:
    descriptors: list[DeviceDescriptor] = []
    context = rs.context()
    for device in context.query_devices():
        try:
            serial = str(device.get_info(rs.camera_info.serial_number)).strip()
            name = str(device.get_info(rs.camera_info.name)).strip()
            uid = str(device.get_info(rs.camera_info.physical_port)).strip()
        except Exception:
            continue
        if serial:
            descriptors.append(DeviceDescriptor("realsense", serial, name, uid))
    return tuple(descriptors)


class RealSenseCamera(CameraNode):
    def __init__(
        self,
        descriptor: DeviceDescriptor,
        logical_name: str,
        config: CameraConfig,
        rs,
    ) -> None:
        super().__init__(descriptor, logical_name, config)
        self._rs = rs
        self._pipeline = rs.pipeline()
        self._pipeline_started = False
        self._frame_lock = Lock()
        self._latest_frameset = None
        self._motion_samples: deque[tuple[object, float, float, float]] = deque(
            maxlen=2048
        )
        self._pub_gyro = (
            self.create_publisher(
                Imu, self.topic("gyro/sample"), qos_profile_sensor_data
            )
            if config.enable_gyro
            else None
        )
        self._pub_accel = (
            self.create_publisher(
                Imu, self.topic("accel/sample"), qos_profile_sensor_data
            )
            if config.enable_accel
            else None
        )
        self._frame_guard = self.create_guard_condition(self._publish_latest)

        try:
            profile = self._start_pipeline()
            self._initialize_depth_processing(profile)
        except BaseException:
            self._stop_pipeline()
            super().destroy_node()
            raise

    def _start_pipeline(self):
        rs = self._rs
        config = rs.config()
        config.enable_device(self.descriptor.serial)
        profile = self.config.depth_profile
        if self.config.enable_depth:
            if profile.automatic:
                config.enable_stream(rs.stream.depth)
            else:
                config.enable_stream(
                    rs.stream.depth,
                    profile.width,
                    profile.height,
                    rs.format.z16,
                    profile.fps,
                )
        color = self.config.color_profile
        if self.config.enable_color:
            if color.automatic:
                config.enable_stream(rs.stream.color)
            else:
                config.enable_stream(
                    rs.stream.color,
                    color.width,
                    color.height,
                    rs.format.bgr8,
                    color.fps,
                )
        if self.config.enable_infra1:
            if profile.automatic:
                config.enable_stream(rs.stream.infrared, 1)
            else:
                config.enable_stream(
                    rs.stream.infrared,
                    1,
                    profile.width,
                    profile.height,
                    rs.format.y8,
                    profile.fps,
                )
        if self.config.enable_infra2:
            if profile.automatic:
                config.enable_stream(rs.stream.infrared, 2)
            else:
                config.enable_stream(
                    rs.stream.infrared,
                    2,
                    profile.width,
                    profile.height,
                    rs.format.y8,
                    profile.fps,
                )
        if self.config.enable_gyro:
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
        if self.config.enable_accel:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)

        try:
            active = self._pipeline.start(config, self._on_frame)
        except Exception as exc:
            raise RuntimeError(
                f"RealSense {self.descriptor.serial} pipeline start failed: {exc}"
            ) from exc
        self._pipeline_started = True
        return active

    def _initialize_depth_processing(self, profile) -> None:
        rs = self._rs
        device = profile.get_device()
        self._configure_device(device)
        self._decimation_filter = rs.decimation_filter()
        self._decimation_filter.set_option(
            rs.option.filter_magnitude,
            float(self.config.decimation_magnitude),
        )
        self._spatial_filter = rs.spatial_filter()
        self._spatial_filter.set_option(
            rs.option.filter_smooth_alpha, self.config.spatial_alpha
        )
        self._spatial_filter.set_option(
            rs.option.filter_smooth_delta, self.config.spatial_delta
        )
        self._spatial_filter.set_option(
            rs.option.holes_fill, float(self.config.spatial_holes_fill)
        )
        self._temporal_filter = rs.temporal_filter()
        self._temporal_filter.set_option(
            rs.option.filter_smooth_alpha, self.config.temporal_alpha
        )
        self._temporal_filter.set_option(
            rs.option.filter_smooth_delta, self.config.temporal_delta
        )
        self._temporal_filter.set_option(
            rs.option.holes_fill, float(self.config.temporal_holes_fill)
        )
        self._hole_filter = rs.hole_filling_filter(self.config.hole_filling_mode)
        self._second_hole_filter = rs.hole_filling_filter(
            self.config.second_hole_filling_mode
        )

        enabled: list[str] = []
        if self.config.decimation_enabled:
            enabled.append("decimation")
        if self.config.spatial_enabled:
            enabled.append("spatial")
        if self.config.temporal_enabled:
            enabled.append("temporal")
        if self.config.hole_filling_enabled:
            enabled.append("hole_filling")
        if self.config.second_hole_filling_enabled:
            enabled.append("second_hole_filling")
        identity = f"{self.descriptor.name} serial={self.descriptor.serial}"
        self.get_logger().info(
            f"started RealSense {identity}; filters={','.join(enabled) or 'none'}"
        )

    def _configure_device(self, device) -> None:
        rs = self._rs
        for sensor in device.query_sensors():
            try:
                if sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(rs.option.emitter_enabled, 1.0)
                if sensor.supports(rs.option.laser_power):
                    limits = sensor.get_option_range(rs.option.laser_power)
                    sensor.set_option(rs.option.laser_power, limits.max)
                if sensor.supports(rs.option.visual_preset):
                    sensor.set_option(
                        rs.option.visual_preset,
                        float(int(rs.rs400_visual_preset.high_accuracy)),
                    )
            except Exception as exc:
                self.log_throttled(
                    "device-options",
                    f"cannot configure RealSense {self.descriptor.serial}: {exc}",
                    error=False,
                )

    def _on_frame(self, frame) -> None:
        try:
            if frame.is_frameset():
                with self._frame_lock:
                    self._latest_frameset = frame.as_frameset()
                self.mark_frame()
                self._frame_guard.trigger()
                return
            if frame.is_motion_frame():
                motion_frame = frame.as_motion_frame()
                motion = motion_frame.get_motion_data()
                with self._frame_lock:
                    self._motion_samples.append(
                        (
                            motion_frame.get_profile().stream_type(),
                            float(motion.x),
                            float(motion.y),
                            float(motion.z),
                        )
                    )
                self.mark_frame()
                self._frame_guard.trigger()
        except Exception as exc:
            self.log_throttled(
                "callback",
                f"RealSense {self.descriptor.serial} callback failed: {exc}",
            )

    def _publish_latest(self) -> None:
        with self._frame_lock:
            frameset = self._latest_frameset
            self._latest_frameset = None
            motion = tuple(self._motion_samples)
            self._motion_samples.clear()
        if frameset is not None:
            try:
                self._publish_frameset(frameset)
            except Exception as exc:
                self.log_throttled(
                    "frameset",
                    f"RealSense {self.descriptor.serial} frame failed: {exc}",
                )
        if motion:
            self._publish_motion(motion)

    def _publish_frameset(self, frameset) -> None:
        stamp = self.get_clock().now().to_msg()
        if self._pub_depth is not None:
            frame = frameset.get_depth_frame()
            if frame:
                self._publish_depth(frame, stamp)
        if self._pub_color is not None:
            frame = frameset.get_color_frame()
            if frame:
                self._publish_video_frame(
                    frame,
                    stamp,
                    self._pub_color,
                    self._pub_color_info,
                    self.config.rectify_color,
                    "bgr8",
                    self.color_frame_id,
                )
        if self._pub_infra1 is not None:
            frame = frameset.get_infrared_frame(1)
            if frame:
                self._publish_video_frame(
                    frame,
                    stamp,
                    self._pub_infra1,
                    self._pub_infra1_info,
                    self.config.rectify_infra1,
                    "mono8",
                    self.infra1_frame_id,
                )
        if self._pub_infra2 is not None:
            frame = frameset.get_infrared_frame(2)
            if frame:
                self._publish_video_frame(
                    frame,
                    stamp,
                    self._pub_infra2,
                    self._pub_infra2_info,
                    self.config.rectify_infra2,
                    "mono8",
                    self.infra2_frame_id,
                )

    def _publish_depth(self, depth_frame, stamp) -> None:
        frame = depth_frame
        if self.config.decimation_enabled:
            frame = self._decimation_filter.process(frame)
        if self.config.spatial_enabled:
            frame = self._spatial_filter.process(frame)
        if self.config.temporal_enabled:
            frame = self._temporal_filter.process(frame)
        if self.config.hole_filling_enabled:
            frame = self._hole_filter.process(frame)
        if self.config.second_hole_filling_enabled:
            frame = self._second_hole_filter.process(frame)

        depth = np.asanyarray(frame.get_data())
        intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
        assert self._pub_depth_info is not None
        self.publish_calibrated_image(
            depth,
            "16UC1",
            self.depth_frame_id,
            stamp,
            self._pub_depth,
            self._pub_depth_info,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.ppx),
            cy=float(intrinsics.ppy),
            distortion=tuple(float(value) for value in intrinsics.coeffs),
            distortion_model=intrinsics.model,
            rectify=self.config.rectify_depth,
            depth=True,
        )

    def _publish_video_frame(
        self,
        frame,
        stamp,
        image_publisher,
        info_publisher,
        rectify,
        encoding,
        frame_id,
    ) -> None:
        intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.publish_calibrated_image(
            np.asanyarray(frame.get_data()),
            encoding,
            frame_id,
            stamp,
            image_publisher,
            info_publisher,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.ppx),
            cy=float(intrinsics.ppy),
            distortion=tuple(float(value) for value in intrinsics.coeffs),
            distortion_model=intrinsics.model,
            rectify=rectify,
        )

    def _publish_motion(self, samples) -> None:
        rs = self._rs
        for stream_type, x, y, z in samples:
            message = Imu()
            message.header.stamp = self.get_clock().now().to_msg()
            if stream_type == rs.stream.gyro and self._pub_gyro is not None:
                message.header.frame_id = self.gyro_frame_id
                message.angular_velocity.x = x
                message.angular_velocity.y = y
                message.angular_velocity.z = z
                self._pub_gyro.publish(message)
            elif stream_type == rs.stream.accel and self._pub_accel is not None:
                message.header.frame_id = self.accel_frame_id
                message.linear_acceleration.x = x
                message.linear_acceleration.y = y
                message.linear_acceleration.z = z
                self._pub_accel.publish(message)

    def _stop_pipeline(self) -> None:
        if not self._pipeline_started:
            return
        self._pipeline_started = False
        try:
            self._pipeline.stop()
        except KeyboardInterrupt:
            pass
        except BaseException as exc:
            self.log_throttled(
                "stop",
                f"RealSense {self.descriptor.serial} stop failed: {exc}",
                error=False,
            )

    def destroy_node(self):
        self._stop_pipeline()
        return super().destroy_node()


__all__ = ["RealSenseCamera", "discover_realsense"]
