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
        self._align_to_color = None
        self._pointcloud = None
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
            self.start_pointcloud_worker(self._publish_pointcloud)
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
        if self.config.align_depth:
            self._align_to_color = rs.align(rs.stream.color)
        if self.config.pointcloud_enabled:
            self._pointcloud = rs.pointcloud()

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
                self.mark_frame()
                if not self.video_consumers_requested():
                    return
                with self._frame_lock:
                    self._latest_frameset = frame.as_frameset()
                self._frame_guard.trigger()
                return
            if frame.is_motion_frame():
                motion_frame = frame.as_motion_frame()
                stream_type = motion_frame.get_profile().stream_type()
                self.mark_frame()
                if stream_type == self._rs.stream.gyro:
                    if not self.publishers_requested(self._pub_gyro):
                        return
                elif stream_type == self._rs.stream.accel:
                    if not self.publishers_requested(self._pub_accel):
                        return
                else:
                    return
                motion = motion_frame.get_motion_data()
                with self._frame_lock:
                    self._motion_samples.append(
                        (
                            stream_type,
                            float(motion.x),
                            float(motion.y),
                            float(motion.z),
                        )
                    )
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
        depth_requested = self.depth_requested()
        color_requested = self.color_requested()
        aligned_depth_requested = self.aligned_depth_requested()
        infra1_requested = self.infra1_requested()
        infra2_requested = self.infra2_requested()
        pointcloud_requested = self.pointcloud_requested()
        if not any(
            (
                depth_requested,
                color_requested,
                aligned_depth_requested,
                infra1_requested,
                infra2_requested,
                pointcloud_requested,
            )
        ):
            return
        stamp = self.get_clock().now().to_msg()
        processed_depth_requested = (
            depth_requested or aligned_depth_requested or pointcloud_requested
        )
        processed_frameset = (
            self._process_depth_frameset(frameset)
            if self.config.enable_depth and processed_depth_requested
            else frameset
        )
        if depth_requested:
            frame = processed_frameset.get_depth_frame()
            if frame:
                self._publish_depth(frame, stamp)
        if color_requested:
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
        if pointcloud_requested:
            self.queue_pointcloud(processed_frameset, frameset, stamp)
        if aligned_depth_requested:
            self._publish_aligned_depth(processed_frameset, frameset, stamp)
        if infra1_requested:
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
        if infra2_requested:
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

    def _process_depth_frameset(self, frameset):
        frame = frameset
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

        if hasattr(frame, "as_frameset"):
            frame = frame.as_frameset()
        if not hasattr(frame, "get_depth_frame"):
            raise RuntimeError("RealSense depth filters did not return a frameset")
        return frame

    def _publish_depth(self, frame, stamp) -> None:
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

    def _publish_pointcloud(self, filtered_frameset, original_frameset, stamp) -> None:
        assert self._pointcloud is not None
        depth_frame = filtered_frameset.get_depth_frame()
        if not depth_frame:
            return
        color_frame = original_frameset.get_color_frame()
        if color_frame:
            self._pointcloud.map_to(color_frame)
        points = self._pointcloud.calculate(depth_frame)
        width = int(depth_frame.get_width())
        height = int(depth_frame.get_height())
        vertices = (
            np.asanyarray(points.get_vertices())
            .view(np.float32)
            .reshape(-1, 3)
        )

        colors = None
        texture_valid = None
        if color_frame:
            texture = (
                np.asanyarray(points.get_texture_coordinates())
                .view(np.float32)
                .reshape(-1, 2)
            )
            color = np.asanyarray(color_frame.get_data())
            color_height, color_width = color.shape[:2]
            texture_valid = (
                np.isfinite(texture).all(axis=1)
                & (texture[:, 0] >= 0.0)
                & (texture[:, 0] < 1.0)
                & (texture[:, 1] >= 0.0)
                & (texture[:, 1] < 1.0)
            )
            colors = np.zeros((vertices.shape[0], 3), dtype=np.uint8)
            if texture_valid.any():
                pixels_x = np.floor(
                    texture[texture_valid, 0] * color_width
                ).astype(np.intp)
                pixels_y = np.floor(
                    texture[texture_valid, 1] * color_height
                ).astype(np.intp)
                colors[texture_valid] = color[pixels_y, pixels_x, :3][:, ::-1]

        self.publish_pointcloud(
            vertices,
            width=width,
            height=height,
            stamp=stamp,
            colors=colors,
            texture_valid=texture_valid,
        )

    def _publish_aligned_depth(
        self, filtered_frameset, original_frameset, stamp
    ) -> None:
        assert self._align_to_color is not None
        assert self._pub_aligned_depth is not None
        assert self._pub_aligned_depth_info is not None
        aligned = self._align_to_color.process(filtered_frameset)
        if hasattr(aligned, "as_frameset"):
            aligned = aligned.as_frameset()
        if not hasattr(aligned, "get_depth_frame"):
            raise RuntimeError("RealSense align filter did not return a frameset")
        depth_frame = aligned.get_depth_frame()
        color_frame = original_frameset.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth = np.asanyarray(depth_frame.get_data())
        color_profile = color_frame.get_profile().as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        color_width = int(color_frame.get_width())
        color_height = int(color_frame.get_height())
        if depth.shape != (color_height, color_width):
            raise RuntimeError(
                "RealSense aligned depth dimensions do not match color: "
                f"depth={depth.shape[1]}x{depth.shape[0]}, "
                f"color={color_width}x{color_height}"
            )
        self.publish_calibrated_image(
            depth,
            "16UC1",
            self.color_frame_id,
            stamp,
            self._pub_aligned_depth,
            self._pub_aligned_depth_info,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.ppx),
            cy=float(intrinsics.ppy),
            distortion=tuple(float(value) for value in intrinsics.coeffs),
            distortion_model=intrinsics.model,
            rectify=self.config.rectify_color,
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
        gyro_requested = self.publishers_requested(self._pub_gyro)
        accel_requested = self.publishers_requested(self._pub_accel)
        if not gyro_requested and not accel_requested:
            return
        for stream_type, x, y, z in samples:
            if stream_type == rs.stream.gyro and gyro_requested:
                message = Imu()
                message.header.stamp = self.get_clock().now().to_msg()
                message.header.frame_id = self.gyro_frame_id
                message.angular_velocity.x = x
                message.angular_velocity.y = y
                message.angular_velocity.z = z
                self._pub_gyro.publish(message)
            elif stream_type == rs.stream.accel and accel_requested:
                message = Imu()
                message.header.stamp = self.get_clock().now().to_msg()
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
        self.stop_pointcloud_worker()
        self._stop_pipeline()
        return super().destroy_node()


__all__ = ["RealSenseCamera", "discover_realsense"]
