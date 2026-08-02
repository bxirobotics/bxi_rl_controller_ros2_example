from __future__ import annotations

from collections import deque
import math
from threading import Lock

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .common import CameraConfig, CameraNode, DeviceDescriptor


ORBBEC_VENDOR_ID = 0x2BC5
GEMINI_335_PRODUCT_ID = 0x0800


def discover_orbbec(ob, context=None) -> tuple[DeviceDescriptor, ...]:
    sdk_context = context or ob.Context()
    devices = sdk_context.query_devices()
    descriptors: list[DeviceDescriptor] = []
    for index in range(devices.get_count()):
        try:
            info = devices.get_device_by_index(index).get_device_info()
            if (
                int(info.get_vid()) != ORBBEC_VENDOR_ID
                or int(info.get_pid()) != GEMINI_335_PRODUCT_ID
            ):
                continue
            serial = str(info.get_serial_number()).strip()
            if serial:
                descriptors.append(
                    DeviceDescriptor(
                        "orbbec",
                        serial,
                        str(info.get_name()).strip(),
                        str(info.get_uid()).strip(),
                    )
                )
        except Exception:
            continue
    return tuple(descriptors)


def _depth_mm_from_sdk_data(data, width: int, height: int, scale_mm: float):
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid Orbbec depth dimensions: {width}x{height}")
    if not math.isfinite(scale_mm) or scale_mm <= 0.0:
        raise ValueError(f"invalid Orbbec depth scale: {scale_mm}")
    required = width * height * 2
    if memoryview(data).nbytes < required:
        raise ValueError(
            f"malformed Orbbec depth frame: expected {required} bytes, "
            f"received {memoryview(data).nbytes}"
        )
    raw = np.frombuffer(data, dtype=np.uint16, count=width * height).reshape(
        height, width
    )
    millimeters = np.rint(raw.astype(np.float64) * scale_mm)
    return np.ascontiguousarray(np.clip(millimeters, 0.0, 65535.0), dtype=np.uint16)


class OrbbecCamera(CameraNode):
    def __init__(
        self,
        descriptor: DeviceDescriptor,
        logical_name: str,
        config: CameraConfig,
        ob,
    ) -> None:
        super().__init__(descriptor, logical_name, config)
        self._ob = ob
        self._sdk_context = None
        self._device = None
        self._pipeline = None
        self._pipeline_started = False
        self._imu_pipeline = None
        self._imu_pipeline_started = False
        self._frame_lock = Lock()
        self._latest_frameset = None
        self._motion_samples: deque[tuple[str, float, float, float]] = deque(
            maxlen=2048
        )
        self._depth_filters: list[object] = []
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
            self._start_sdk()
        except BaseException:
            self._stop_sdk()
            super().destroy_node()
            raise

    def _start_sdk(self) -> None:
        ob = self._ob
        ob.Context.set_logger_to_console(ob.OBLogLevel.WARNING)
        self._sdk_context = ob.Context()
        devices = self._sdk_context.query_devices()
        try:
            self._device = devices.get_device_by_serial_number(self.descriptor.serial)
        except Exception as exc:
            raise RuntimeError(
                f"Orbbec {self.descriptor.serial} is no longer available: {exc}"
            ) from exc

        self._pipeline = ob.Pipeline(self._device)
        config = ob.Config()
        depth_profile = None
        if self.config.enable_depth:
            profiles = self._pipeline.get_stream_profile_list(
                ob.OBSensorType.DEPTH_SENSOR
            )
            requested = self.config.depth_profile
            if requested.automatic:
                depth_profile = profiles.get_default_video_stream_profile()
            else:
                depth_profile = profiles.get_video_stream_profile(
                    requested.width,
                    requested.height,
                    ob.OBFormat.Y16,
                    requested.fps,
                )
            config.enable_stream(depth_profile)

        self._configure_color_stream(config)
        self._configure_ir_streams(config)
        self._initialize_filters()
        try:
            self._pipeline.start(config, self._on_video_frames)
        except Exception as exc:
            raise RuntimeError(
                f"Orbbec {self.descriptor.serial} pipeline start failed: {exc}"
            ) from exc
        self._pipeline_started = True
        if self.config.enable_gyro or self.config.enable_accel:
            self._start_imu_pipeline()

        profile_text = (
            f"{depth_profile.get_width()}x{depth_profile.get_height()}"
            f"@{depth_profile.get_fps()}"
            if depth_profile is not None
            else "disabled"
        )
        names = [item.get_name() for item in self._depth_filters]
        self.get_logger().info(
            f"started Orbbec {self.descriptor.name} serial={self.descriptor.serial}; "
            f"depth={profile_text}; filters={','.join(names) or 'none'}"
        )

    def _configure_color_stream(self, config) -> None:
        if not self.config.enable_color:
            return
        ob = self._ob
        profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        requested = self.config.color_profile
        if requested.automatic:
            config.enable_stream(profiles.get_default_video_stream_profile())
            return
        for frame_format in (ob.OBFormat.BGR, ob.OBFormat.RGB, ob.OBFormat.MJPG):
            try:
                profile = profiles.get_video_stream_profile(
                    requested.width,
                    requested.height,
                    frame_format,
                    requested.fps,
                )
                config.enable_stream(profile)
                return
            except Exception:
                continue
        raise RuntimeError(
            f"Orbbec color profile unavailable: {requested.width}x"
            f"{requested.height}@{requested.fps}"
        )

    def _configure_ir_streams(self, config) -> None:
        ob = self._ob
        selections = (
            (self.config.enable_infra1, ob.OBSensorType.LEFT_IR_SENSOR),
            (self.config.enable_infra2, ob.OBSensorType.RIGHT_IR_SENSOR),
        )
        enabled = 0
        for requested, sensor_type in selections:
            if not requested:
                continue
            try:
                profiles = self._pipeline.get_stream_profile_list(sensor_type)
                profile = self._select_ir_profile(profiles)
                config.enable_stream(profile)
                enabled += 1
            except Exception:
                continue
        if (self.config.enable_infra1 or self.config.enable_infra2) and enabled == 0:
            profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.IR_SENSOR)
            config.enable_stream(self._select_ir_profile(profiles))

    def _select_ir_profile(self, profiles):
        requested = self.config.depth_profile
        if requested.automatic:
            return profiles.get_default_video_stream_profile()
        return profiles.get_video_stream_profile(
            requested.width,
            requested.height,
            self._ob.OBFormat.Y8,
            requested.fps,
        )

    def _initialize_filters(self) -> None:
        if not self.config.enable_depth:
            return
        ob = self._ob
        sensor = self._device.get_sensor(ob.OBSensorType.DEPTH_SENSOR)
        mandatory: list[object] = []
        optional: list[object] = []
        for depth_filter in sensor.get_recommended_filters():
            required = depth_filter.is_disparity_transform_filter()
            selected = self.config.orbbec_enable_sdk_filters and (
                depth_filter.is_noise_removal_filter()
                or depth_filter.is_spatial_advanced_filter()
                or depth_filter.is_temporal_filter()
                or depth_filter.is_hole_filling_filter()
            )
            depth_filter.enable(required or selected)
            if required:
                mandatory.append(depth_filter)
            elif selected:
                optional.append(depth_filter)
        self._depth_filters = mandatory + optional

    def _start_imu_pipeline(self) -> None:
        ob = self._ob
        self._imu_pipeline = ob.Pipeline(self._device)
        config = ob.Config()
        if self.config.enable_accel:
            config.enable_accel_stream()
        if self.config.enable_gyro:
            config.enable_gyro_stream()
        try:
            self._imu_pipeline.start(config, self._on_imu_frames)
        except Exception as exc:
            raise RuntimeError(
                f"Orbbec {self.descriptor.serial} IMU start failed: {exc}"
            ) from exc
        self._imu_pipeline_started = True

    def _on_video_frames(self, frameset) -> None:
        if frameset is None:
            return
        with self._frame_lock:
            self._latest_frameset = frameset
        self.mark_frame()
        self._frame_guard.trigger()

    def _on_imu_frames(self, frameset) -> None:
        if frameset is None:
            return
        try:
            samples: list[tuple[str, float, float, float]] = []
            accel = frameset.get_accel_frame()
            if accel and self.config.enable_accel:
                samples.append(
                    (
                        "accel",
                        float(accel.get_x()),
                        float(accel.get_y()),
                        float(accel.get_z()),
                    )
                )
            gyro = frameset.get_gyro_frame()
            if gyro and self.config.enable_gyro:
                samples.append(
                    (
                        "gyro",
                        float(gyro.get_x()),
                        float(gyro.get_y()),
                        float(gyro.get_z()),
                    )
                )
            if samples:
                with self._frame_lock:
                    self._motion_samples.extend(samples)
                self.mark_frame()
                self._frame_guard.trigger()
        except Exception as exc:
            self.log_throttled(
                "imu-callback",
                f"Orbbec {self.descriptor.serial} IMU callback failed: {exc}",
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
                    f"Orbbec {self.descriptor.serial} frame failed: {exc}",
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
                image, encoding = self._color_array(frame)
                self._publish_video(
                    frame,
                    image,
                    encoding,
                    self.color_frame_id,
                    stamp,
                    self._pub_color,
                    self._pub_color_info,
                    self.config.rectify_color,
                )
        left = frameset.get_left_ir_frame() if self._pub_infra1 is not None else None
        right = frameset.get_right_ir_frame() if self._pub_infra2 is not None else None
        if self._pub_infra1 is not None and not left:
            left = frameset.get_ir_frame()
        if left and self._pub_infra1 is not None:
            image, encoding = self._ir_array(left)
            self._publish_video(
                left,
                image,
                encoding,
                self.infra1_frame_id,
                stamp,
                self._pub_infra1,
                self._pub_infra1_info,
                self.config.rectify_infra1,
            )
        if right and self._pub_infra2 is not None:
            image, encoding = self._ir_array(right)
            self._publish_video(
                right,
                image,
                encoding,
                self.infra2_frame_id,
                stamp,
                self._pub_infra2,
                self._pub_infra2_info,
                self.config.rectify_infra2,
            )

    def _publish_depth(self, depth_frame, stamp) -> None:
        frame = depth_frame
        for depth_filter in self._depth_filters:
            frame = depth_filter.process(frame)
            if frame is None:
                raise RuntimeError(
                    f"filter {depth_filter.get_name()} returned no frame"
                )
        if hasattr(frame, "as_depth_frame"):
            frame = frame.as_depth_frame()
        width = int(frame.get_width())
        height = int(frame.get_height())
        depth = _depth_mm_from_sdk_data(
            frame.get_data(), width, height, float(frame.get_depth_scale())
        )
        fx, fy, cx, cy, distortion = self._video_calibration(frame, width, height)
        assert self._pub_depth_info is not None
        self.publish_calibrated_image(
            depth,
            "16UC1",
            self.depth_frame_id,
            stamp,
            self._pub_depth,
            self._pub_depth_info,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=distortion,
            distortion_model="brown_conrady",
            rectify=self.config.rectify_depth,
            depth=True,
        )

    def _publish_video(
        self,
        frame,
        array,
        encoding,
        frame_id,
        stamp,
        image_publisher,
        info_publisher,
        rectify,
    ) -> None:
        fx, fy, cx, cy, distortion = self._video_calibration(
            frame, int(array.shape[1]), int(array.shape[0])
        )
        self.publish_calibrated_image(
            array,
            encoding,
            frame_id,
            stamp,
            image_publisher,
            info_publisher,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=distortion,
            distortion_model="brown_conrady",
            rectify=rectify,
        )

    def _video_calibration(self, frame, width: int, height: int):
        try:
            profile = frame.get_stream_profile().as_video_stream_profile()
            intrinsic = profile.get_intrinsic()
            distortion = profile.get_distortion()
            scale_x = float(width) / float(intrinsic.width)
            scale_y = float(height) / float(intrinsic.height)
            return (
                float(intrinsic.fx) * scale_x,
                float(intrinsic.fy) * scale_y,
                float(intrinsic.cx) * scale_x,
                float(intrinsic.cy) * scale_y,
                (
                    float(distortion.k1),
                    float(distortion.k2),
                    float(distortion.p1),
                    float(distortion.p2),
                    float(distortion.k3),
                    float(distortion.k4),
                    float(distortion.k5),
                    float(distortion.k6),
                ),
            )
        except Exception as exc:
            self.log_throttled(
                "calibration",
                f"Orbbec {self.descriptor.serial} calibration unavailable: {exc}",
                error=False,
            )
            fx, fy, cx, cy = self.fallback_intrinsics(
                width,
                height,
                self.config.orbbec_fallback_hfov,
                self.config.orbbec_fallback_vfov,
            )
            return fx, fy, cx, cy, (0.0,) * 5

    def _color_array(self, frame):
        ob = self._ob
        width = int(frame.get_width())
        height = int(frame.get_height())
        frame_format = frame.get_format()
        data = frame.get_data()
        if frame_format == ob.OBFormat.RGB:
            return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), "rgb8"
        if frame_format == ob.OBFormat.BGR:
            return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), "bgr8"
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                f"OpenCV is required for Orbbec color format {frame_format}"
            ) from exc
        encoded = np.frombuffer(data, dtype=np.uint8)
        if frame_format == ob.OBFormat.MJPG:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        elif frame_format == ob.OBFormat.YUYV:
            decoded = cv2.cvtColor(
                encoded.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
            )
        elif frame_format == ob.OBFormat.UYVY:
            decoded = cv2.cvtColor(
                encoded.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY
            )
        else:
            raise ValueError(f"unsupported Orbbec color format: {frame_format}")
        if decoded is None:
            raise ValueError(f"cannot decode Orbbec color format: {frame_format}")
        return decoded, "bgr8"

    def _ir_array(self, frame):
        width = int(frame.get_width())
        height = int(frame.get_height())
        if frame.get_format() == self._ob.OBFormat.Y8:
            return (
                np.frombuffer(frame.get_data(), dtype=np.uint8).reshape(height, width),
                "mono8",
            )
        return (
            np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(height, width),
            "mono16",
        )

    def _publish_motion(self, samples) -> None:
        for kind, x, y, z in samples:
            message = Imu()
            message.header.stamp = self.get_clock().now().to_msg()
            if kind == "gyro" and self._pub_gyro is not None:
                message.header.frame_id = self.gyro_frame_id
                message.angular_velocity.x = x
                message.angular_velocity.y = y
                message.angular_velocity.z = z
                self._pub_gyro.publish(message)
            elif kind == "accel" and self._pub_accel is not None:
                message.header.frame_id = self.accel_frame_id
                message.linear_acceleration.x = x
                message.linear_acceleration.y = y
                message.linear_acceleration.z = z
                self._pub_accel.publish(message)

    def _stop_sdk(self) -> None:
        if self._imu_pipeline_started and self._imu_pipeline is not None:
            self._imu_pipeline_started = False
            try:
                self._imu_pipeline.stop()
            except KeyboardInterrupt:
                pass
            except BaseException as exc:
                self.log_throttled("imu-stop", str(exc), error=False)
        if self._pipeline_started and self._pipeline is not None:
            self._pipeline_started = False
            try:
                self._pipeline.stop()
            except KeyboardInterrupt:
                pass
            except BaseException as exc:
                self.log_throttled("stop", str(exc), error=False)
        self._depth_filters.clear()
        self._imu_pipeline = None
        self._pipeline = None
        self._device = None
        self._sdk_context = None

    def destroy_node(self):
        self._stop_sdk()
        return super().destroy_node()


__all__ = [
    "GEMINI_335_PRODUCT_ID",
    "ORBBEC_VENDOR_ID",
    "OrbbecCamera",
    "_depth_mm_from_sdk_data",
    "discover_orbbec",
]
