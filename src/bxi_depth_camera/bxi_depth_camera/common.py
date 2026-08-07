from __future__ import annotations

from array import array as byte_array
from dataclasses import dataclass
import math
import re
from threading import Condition, Thread
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


@dataclass(frozen=True)
class StreamProfile:
    width: int = 0
    height: int = 0
    fps: int = 0

    @property
    def automatic(self) -> bool:
        return self.width == self.height == self.fps == 0


def parse_profile(value: str, name: str) -> StreamProfile:
    text = value.strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must look like '640x480x30' or '0,0,0'")
    try:
        width, height, fps = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-integer value: {value!r}") from exc
    if width == height == fps == 0:
        return StreamProfile()
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"{name} dimensions and FPS must all be positive")
    return StreamProfile(width, height, fps)


def topic_token(serial: str) -> str:
    token = serial.strip()
    if not token or re.fullmatch(r"[A-Za-z0-9_]+", token) is None:
        raise ValueError(
            "camera serial must contain only letters, digits, or underscores: "
            f"{serial!r}"
        )
    return "SN_" + token


def node_token(serial: str) -> str:
    return f"camera_{topic_token(serial)}"


def camera_name_token(name: str) -> str:
    token = name.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token) is None:
        raise ValueError(
            "logical camera name must start with a letter and contain only "
            f"letters, digits, or underscores: {name!r}"
        )
    return token


def camera_base_topic(namespace: str, logical_name: str) -> str:
    prefix = namespace.strip("/")
    if not prefix:
        raise ValueError("camera namespace must not be empty")
    return "/" + "/".join((prefix, camera_name_token(logical_name)))


@dataclass(frozen=True)
class DeviceDescriptor:
    backend: str
    serial: str
    name: str
    uid: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.backend, self.serial


@dataclass(frozen=True)
class CameraConfig:
    camera_namespace: str
    depth_profile: StreamProfile
    color_profile: StreamProfile
    enable_depth: bool
    enable_color: bool
    enable_infra1: bool
    enable_infra2: bool
    enable_gyro: bool
    enable_accel: bool
    align_depth: bool
    pointcloud_enabled: bool
    pointcloud_ordered: bool
    pointcloud_allow_no_texture_points: bool
    pointcloud_max_fps: float
    rectify_depth: bool
    rectify_color: bool
    rectify_infra1: bool
    rectify_infra2: bool
    device_timeout_sec: float
    decimation_enabled: bool
    decimation_magnitude: int
    spatial_enabled: bool
    spatial_alpha: float
    spatial_delta: float
    spatial_holes_fill: int
    temporal_enabled: bool
    temporal_alpha: float
    temporal_delta: float
    temporal_holes_fill: int
    hole_filling_enabled: bool
    hole_filling_mode: int
    second_hole_filling_enabled: bool
    second_hole_filling_mode: int
    orbbec_enable_sdk_filters: bool
    orbbec_fallback_hfov: float
    orbbec_fallback_vfov: float

    def __post_init__(self) -> None:
        if self.align_depth and not (self.enable_depth and self.enable_color):
            raise ValueError(
                "align_depth.enable requires both enable_depth and enable_color"
            )
        if self.pointcloud_enabled and not self.enable_depth:
            raise ValueError("pointcloud.enable requires enable_depth")
        if (
            not math.isfinite(self.pointcloud_max_fps)
            or self.pointcloud_max_fps <= 0
        ):
            raise ValueError("pointcloud.max_fps must be greater than zero")


class ImageRectifier:
    """Lazily build and cache OpenCV remap tables for camera streams."""

    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "image rectification requires the python3-opencv package"
            ) from exc
        self._cv2 = cv2
        self._maps: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _model_name(model: object) -> str:
        return str(model).strip().lower().rsplit(".", 1)[-1]

    def _create_maps(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion: tuple[float, ...],
        model: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        cv2 = self._cv2
        matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        coefficients = np.asarray(distortion, dtype=np.float64)
        size = (width, height)

        if model in {"brown_conrady", "modified_brown_conrady", "plumb_bob"}:
            return cv2.initUndistortRectifyMap(
                matrix,
                coefficients,
                None,
                matrix,
                size,
                cv2.CV_32FC1,
            )
        if model == "inverse_brown_conrady":
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            points = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
            source = cv2.undistortPoints(
                points,
                matrix,
                coefficients,
                P=matrix,
            ).reshape(height, width, 2)
            return (
                np.ascontiguousarray(source[:, :, 0], dtype=np.float32),
                np.ascontiguousarray(source[:, :, 1], dtype=np.float32),
            )
        if model in {"kannala_brandt4", "equidistant"}:
            if coefficients.size < 4:
                raise ValueError(
                    f"{model} requires four distortion coefficients"
                )
            return cv2.fisheye.initUndistortRectifyMap(
                matrix,
                coefficients[:4],
                np.eye(3, dtype=np.float64),
                matrix,
                size,
                cv2.CV_32FC1,
            )
        raise ValueError(f"unsupported distortion model: {model}")

    def rectify(
        self,
        image: np.ndarray,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion: tuple[float, ...],
        distortion_model: object,
        nearest: bool,
    ) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(f"cannot rectify image with shape {image.shape}")
        height, width = image.shape[:2]
        model = self._model_name(distortion_model)
        coefficients = tuple(float(value) for value in distortion)
        if model == "none" or not coefficients or all(
            abs(value) <= 1e-12 for value in coefficients
        ):
            return np.ascontiguousarray(image)
        values = (fx, fy, cx, cy, *coefficients)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera calibration contains a non-finite value")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")

        key = (width, height, fx, fy, cx, cy, coefficients, model)
        maps = self._maps.get(key)
        if maps is None:
            maps = self._create_maps(
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                coefficients,
                model,
            )
            self._maps[key] = maps
        interpolation = (
            self._cv2.INTER_NEAREST if nearest else self._cv2.INTER_LINEAR
        )
        return self._cv2.remap(
            image,
            maps[0],
            maps[1],
            interpolation,
            borderMode=self._cv2.BORDER_CONSTANT,
            borderValue=0,
        )


class CameraNode(Node):
    """Shared ROS-facing behavior for one physical camera."""

    def __init__(
        self,
        descriptor: DeviceDescriptor,
        logical_name: str,
        config: CameraConfig,
    ) -> None:
        namespace = config.camera_namespace.strip("/")
        self.logical_name = camera_name_token(logical_name)
        self.base_topic = camera_base_topic(namespace, self.logical_name)
        super().__init__(node_token(descriptor.serial), namespace=f"/{namespace}")
        self.descriptor = descriptor
        self.config = config
        self.started_monotonic = time.monotonic()
        self.last_frame_monotonic: float | None = None
        self._last_log_times: dict[str, float] = {}
        self._last_pointcloud_monotonic = float("-inf")
        self._pointcloud_condition = Condition()
        self._pointcloud_pending = None
        self._pointcloud_worker = None
        self._pointcloud_stopping = False
        rectification_enabled = any(
            (
                config.enable_depth and config.rectify_depth,
                config.enable_color and config.rectify_color,
                config.enable_infra1 and config.rectify_infra1,
                config.enable_infra2 and config.rectify_infra2,
            )
        )
        self._rectifier = ImageRectifier() if rectification_enabled else None

        self._pub_depth = (
            self.create_publisher(
                Image, self.topic("depth/image_rect_raw"), qos_profile_sensor_data
            )
            if config.enable_depth
            else None
        )
        self._pub_depth_info = (
            self.create_publisher(
                CameraInfo,
                self.topic("depth/camera_info"),
                qos_profile_sensor_data,
            )
            if config.enable_depth
            else None
        )
        self._pub_color = (
            self.create_publisher(
                Image, self.topic("color/image_raw"), qos_profile_sensor_data
            )
            if config.enable_color
            else None
        )
        self._pub_color_info = (
            self.create_publisher(
                CameraInfo, self.topic("color/camera_info"), qos_profile_sensor_data
            )
            if config.enable_color
            else None
        )
        self._pub_aligned_depth = (
            self.create_publisher(
                Image,
                self.topic("aligned_depth_to_color/image_raw"),
                qos_profile_sensor_data,
            )
            if config.align_depth
            else None
        )
        self._pub_aligned_depth_info = (
            self.create_publisher(
                CameraInfo,
                self.topic("aligned_depth_to_color/camera_info"),
                qos_profile_sensor_data,
            )
            if config.align_depth
            else None
        )
        self._pub_pointcloud = (
            self.create_publisher(
                PointCloud2,
                self.topic("depth/color/points"),
                qos_profile_sensor_data,
            )
            if config.pointcloud_enabled
            else None
        )
        self._pub_infra1 = (
            self.create_publisher(
                Image, self.topic("infra1/image_rect_raw"), qos_profile_sensor_data
            )
            if config.enable_infra1
            else None
        )
        self._pub_infra1_info = (
            self.create_publisher(
                CameraInfo, self.topic("infra1/camera_info"), qos_profile_sensor_data
            )
            if config.enable_infra1
            else None
        )
        self._pub_infra2 = (
            self.create_publisher(
                Image, self.topic("infra2/image_rect_raw"), qos_profile_sensor_data
            )
            if config.enable_infra2
            else None
        )
        self._pub_infra2_info = (
            self.create_publisher(
                CameraInfo, self.topic("infra2/camera_info"), qos_profile_sensor_data
            )
            if config.enable_infra2
            else None
        )
    @property
    def depth_frame_id(self) -> str:
        return f"{self.logical_name}_depth_optical_frame"

    @property
    def color_frame_id(self) -> str:
        return f"{self.logical_name}_color_optical_frame"

    @property
    def infra1_frame_id(self) -> str:
        return f"{self.logical_name}_infra1_optical_frame"

    @property
    def infra2_frame_id(self) -> str:
        return f"{self.logical_name}_infra2_optical_frame"

    @property
    def gyro_frame_id(self) -> str:
        return f"{self.logical_name}_gyro_optical_frame"

    @property
    def accel_frame_id(self) -> str:
        return f"{self.logical_name}_accel_optical_frame"

    def topic(self, suffix: str = "") -> str:
        return (
            self.base_topic if not suffix else f"{self.base_topic}/{suffix.strip('/')}"
        )

    def publish_calibrated_image(
        self,
        array: np.ndarray,
        encoding: str,
        frame_id: str,
        stamp,
        image_publisher,
        info_publisher,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion: tuple[float, ...],
        distortion_model: object,
        rectify: bool,
        depth: bool = False,
    ) -> None:
        model_name = ImageRectifier._model_name(distortion_model)
        if model_name in {"kannala_brandt4", "equidistant", "ftheta"}:
            ros_model = "equidistant"
        elif len(distortion) >= 8 and any(
            abs(value) > 1e-12 for value in distortion[5:8]
        ):
            ros_model = "rational_polynomial"
        else:
            ros_model = "plumb_bob"
        output = array
        output_distortion = distortion
        if rectify:
            assert self._rectifier is not None
            try:
                output = self._rectifier.rectify(
                    array,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    distortion=distortion,
                    distortion_model=distortion_model,
                    nearest=depth,
                )
                output_distortion = (0.0,) * max(5, len(distortion))
            except (RuntimeError, TypeError, ValueError) as exc:
                self.log_throttled(
                    f"rectification-{frame_id}",
                    f"cannot rectify {frame_id}; publishing the SDK frame: {exc}",
                    error=False,
                )

        image = self.image_message(output, encoding, frame_id, stamp)
        info_publisher.publish(
            self.camera_info(
                image,
                fx,
                fy,
                cx,
                cy,
                distortion=output_distortion,
                distortion_model=ros_model,
            )
        )
        image_publisher.publish(image)

    def mark_frame(self) -> None:
        self.last_frame_monotonic = time.monotonic()

    @staticmethod
    def publishers_requested(*publishers) -> bool:
        return any(
            publisher is not None and publisher.get_subscription_count() > 0
            for publisher in publishers
        )

    def depth_requested(self) -> bool:
        return self.publishers_requested(self._pub_depth, self._pub_depth_info)

    def color_requested(self) -> bool:
        return self.publishers_requested(self._pub_color, self._pub_color_info)

    def aligned_depth_requested(self) -> bool:
        return self.publishers_requested(
            self._pub_aligned_depth,
            self._pub_aligned_depth_info,
        )

    def infra1_requested(self) -> bool:
        return self.publishers_requested(self._pub_infra1, self._pub_infra1_info)

    def infra2_requested(self) -> bool:
        return self.publishers_requested(self._pub_infra2, self._pub_infra2_info)

    def video_consumers_requested(self) -> bool:
        """Return whether a video frame could currently produce ROS output.

        Unlike :meth:`pointcloud_requested`, this deliberately does not apply
        point-cloud rate limiting. SDK callback threads use it only as a cheap
        gate before handing a frame to the ROS executor.
        """
        return self.publishers_requested(
            self._pub_depth,
            self._pub_depth_info,
            self._pub_color,
            self._pub_color_info,
            self._pub_aligned_depth,
            self._pub_aligned_depth_info,
            self._pub_infra1,
            self._pub_infra1_info,
            self._pub_infra2,
            self._pub_infra2_info,
            self._pub_pointcloud,
        )

    def pointcloud_requested(self) -> bool:
        if not self.publishers_requested(self._pub_pointcloud):
            return False
        now = time.monotonic()
        period = 1.0 / self.config.pointcloud_max_fps
        if now - self._last_pointcloud_monotonic < period:
            return False
        self._last_pointcloud_monotonic = now
        return True

    def start_pointcloud_worker(self, callback) -> None:
        if self._pub_pointcloud is None or self._pointcloud_worker is not None:
            return
        self._pointcloud_stopping = False
        self._pointcloud_callback = callback
        self._pointcloud_worker = Thread(
            target=self._run_pointcloud_worker,
            name=f"{self.logical_name}-pointcloud",
            daemon=True,
        )
        self._pointcloud_worker.start()

    def queue_pointcloud(self, *args) -> None:
        if self._pointcloud_worker is None:
            return
        with self._pointcloud_condition:
            self._pointcloud_pending = args
            self._pointcloud_condition.notify()

    def _run_pointcloud_worker(self) -> None:
        while True:
            with self._pointcloud_condition:
                while (
                    self._pointcloud_pending is None
                    and not self._pointcloud_stopping
                ):
                    self._pointcloud_condition.wait()
                if self._pointcloud_stopping:
                    return
                args = self._pointcloud_pending
                self._pointcloud_pending = None
            if not self.publishers_requested(self._pub_pointcloud):
                continue
            try:
                self._pointcloud_callback(*args)
            except Exception as exc:
                self.log_throttled(
                    "pointcloud",
                    f"{self.descriptor.backend} {self.descriptor.serial} "
                    f"point cloud failed: {exc}",
                )

    def stop_pointcloud_worker(self) -> None:
        worker = self._pointcloud_worker
        if worker is None:
            return
        with self._pointcloud_condition:
            self._pointcloud_stopping = True
            self._pointcloud_pending = None
            self._pointcloud_condition.notify_all()
        worker.join(timeout=2.0)
        if worker.is_alive():
            self.log_throttled(
                "pointcloud-stop",
                f"{self.descriptor.backend} {self.descriptor.serial} point cloud "
                "worker did not stop within 2 seconds",
                error=False,
            )
            return
        self._pointcloud_worker = None

    def publish_pointcloud(
        self,
        vertices: np.ndarray,
        *,
        width: int,
        height: int,
        stamp,
        colors: np.ndarray | None = None,
        texture_valid: np.ndarray | None = None,
    ) -> None:
        assert self._pub_pointcloud is not None
        message = self.pointcloud_message(
            vertices,
            width=width,
            height=height,
            frame_id=self.depth_frame_id,
            stamp=stamp,
            ordered=self.config.pointcloud_ordered,
            allow_no_texture_points=(
                self.config.pointcloud_allow_no_texture_points
            ),
            colors=colors,
            texture_valid=texture_valid,
        )
        self._pub_pointcloud.publish(message)

    @staticmethod
    def pointcloud_message(
        vertices: np.ndarray,
        *,
        width: int,
        height: int,
        frame_id: str,
        stamp,
        ordered: bool,
        allow_no_texture_points: bool,
        colors: np.ndarray | None = None,
        texture_valid: np.ndarray | None = None,
    ) -> PointCloud2:
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid point cloud dimensions: {width}x{height}")
        xyz = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        point_count = int(xyz.shape[0])
        if ordered and point_count != width * height:
            raise ValueError(
                "ordered point cloud size does not match its dimensions: "
                f"points={point_count}, dimensions={width}x{height}"
            )

        rgb = None
        if colors is not None:
            rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
            if rgb.shape[0] != point_count:
                raise ValueError("point cloud colors do not match vertices")
        texture_mask = None
        if texture_valid is not None:
            texture_mask = np.asarray(texture_valid, dtype=bool).reshape(-1)
            if texture_mask.size != point_count:
                raise ValueError("point cloud texture mask does not match vertices")

        geometric_valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.0)
        valid = geometric_valid.copy()
        if texture_mask is not None and not allow_no_texture_points:
            valid &= texture_mask

        if ordered:
            output_xyz = xyz.copy()
            output_xyz[~valid] = np.nan
            if rgb is not None:
                output_rgb = rgb.copy()
                output_rgb[~geometric_valid] = 0
                if texture_mask is not None:
                    output_rgb[~texture_mask] = 0
            else:
                output_rgb = None
            output_width = width
            output_height = height
            is_dense = False
        else:
            output_xyz = np.ascontiguousarray(xyz[valid])
            output_rgb = (
                np.ascontiguousarray(rgb[valid]) if rgb is not None else None
            )
            if output_rgb is not None and texture_mask is not None:
                selected_texture = texture_mask[valid]
                output_rgb[~selected_texture] = 0
            output_width = int(output_xyz.shape[0])
            output_height = 1
            is_dense = True

        field_specs = [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
        ]
        if output_rgb is not None:
            field_specs.append(("rgb", "<f4"))
        packed = np.empty(output_xyz.shape[0], dtype=np.dtype(field_specs))
        packed["x"] = output_xyz[:, 0]
        packed["y"] = output_xyz[:, 1]
        packed["z"] = output_xyz[:, 2]
        if output_rgb is not None:
            rgb_bits = (
                output_rgb[:, 0].astype(np.uint32) << 16
                | output_rgb[:, 1].astype(np.uint32) << 8
                | output_rgb[:, 2].astype(np.uint32)
            )
            packed["rgb"] = rgb_bits.view(np.float32)

        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = output_height
        message.width = output_width
        message.fields = []
        for offset, name in enumerate(("x", "y", "z")):
            field = PointField()
            field.name = name
            field.offset = offset * 4
            field.datatype = PointField.FLOAT32
            field.count = 1
            message.fields.append(field)
        if output_rgb is not None:
            field = PointField()
            field.name = "rgb"
            field.offset = 12
            field.datatype = PointField.FLOAT32
            field.count = 1
            message.fields.append(field)
        message.is_bigendian = False
        message.point_step = int(packed.dtype.itemsize)
        message.row_step = message.point_step * message.width
        message.data = byte_array("B", packed.tobytes())
        message.is_dense = is_dense
        return message

    def is_stale(self, now: float) -> bool:
        reference = self.last_frame_monotonic or self.started_monotonic
        return now - reference > self.config.device_timeout_sec

    @staticmethod
    def image_message(array: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
        data = np.ascontiguousarray(array)
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = int(data.shape[0])
        message.width = int(data.shape[1])
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(data.strides[0])
        message.data = byte_array("B", data.tobytes())
        return message

    @staticmethod
    def camera_info(
        image: Image,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        distortion: tuple[float, ...] = (),
        distortion_model: str = "plumb_bob",
    ) -> CameraInfo:
        info = CameraInfo()
        info.header = image.header
        info.width = image.width
        info.height = image.height
        info.distortion_model = distortion_model
        info.d = list(distortion or (0.0,) * 5)
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    @staticmethod
    def fallback_intrinsics(
        width: int, height: int, hfov_deg: float, vfov_deg: float
    ) -> tuple[float, float, float, float]:
        hfov = math.radians(hfov_deg)
        vfov = math.radians(vfov_deg)
        return (
            float(width) / (2.0 * math.tan(hfov / 2.0)),
            float(height) / (2.0 * math.tan(vfov / 2.0)),
            float(width - 1) / 2.0,
            float(height - 1) / 2.0,
        )

    def log_throttled(self, key: str, message: str, *, error: bool = True) -> None:
        now = time.monotonic()
        if now - self._last_log_times.get(key, float("-inf")) < 5.0:
            return
        self._last_log_times[key] = now
        logger = self.get_logger()
        (logger.error if error else logger.warning)(message)

    def destroy_node(self):
        self.stop_pointcloud_worker()
        return super().destroy_node()
