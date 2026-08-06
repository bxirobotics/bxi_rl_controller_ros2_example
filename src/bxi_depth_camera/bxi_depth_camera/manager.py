from __future__ import annotations

from dataclasses import dataclass
import re
from threading import RLock
import time

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from .common import (
    CameraConfig,
    CameraNode,
    DeviceDescriptor,
    camera_name_token,
    parse_profile,
    topic_token,
)
from .orbbec import OrbbecCamera, discover_orbbec
from .realsense import RealSenseCamera, discover_realsense
from .vendor import fallback_modules, load_sdk


CONFIG_PARAMETER_DEFAULTS = {
    "camera_namespace": "hardware",
    "device_timeout_sec": 3.0,
    "enable_depth": True,
    "enable_color": True,
    "enable_infra1": False,
    "enable_infra2": False,
    "enable_gyro": False,
    "enable_accel": False,
    "depth_module.depth_profile": "0,0,0",
    "depth_module.rectification.enable": False,
    "rgb_camera.color_profile": "0,0,0",
    "rgb_camera.rectification.enable": True,
    "infra1.rectification.enable": False,
    "infra2.rectification.enable": False,
    "decimation_filter.enable": False,
    "decimation_filter.filter_magnitude": 1,
    "spatial_filter.enable": True,
    "spatial_filter.filter_smooth_alpha": 0.45,
    "spatial_filter.filter_smooth_delta": 20.0,
    "spatial_filter.holes_fill": 2,
    "temporal_filter.enable": True,
    "temporal_filter.filter_smooth_alpha": 0.45,
    "temporal_filter.filter_smooth_delta": 20.0,
    "temporal_filter.holes_fill": 4,
    "hole_filling_filter.enable": True,
    "hole_filling_filter.holes_fill": 1,
    "second_hole_filling_filter.enable": True,
    "second_hole_filling_filter.holes_fill": 2,
    "orbbec.enable_sdk_filters": True,
    "orbbec.fallback_hfov": 90.0,
    "orbbec.fallback_vfov": 65.0,
}

MANAGER_PARAMETER_DEFAULTS = {
    "serial_no": "",
    "single_camera_name": "head_depth_camera",
    "discovery_interval_sec": 1.0,
    "retry_interval_sec": 2.0,
}

_CAMERA_PARAMETER_PATTERN = re.compile(r"^cameras\.([A-Za-z][A-Za-z0-9_]*)\.(.+)$")
_INHERIT = object()


def camera_parameter_parts(name: str) -> tuple[str, str] | None:
    match = _CAMERA_PARAMETER_PATTERN.fullmatch(name)
    if match is None:
        return None
    logical_name, leaf = match.groups()
    if leaf != "serial_no" and leaf not in CONFIG_PARAMETER_DEFAULTS:
        return None
    if camera_name_token(logical_name) != logical_name:
        return None
    return logical_name, leaf


@dataclass
class _Failure:
    retry_at: float
    message: str
    last_log: float


class CameraManager(Node):
    def __init__(self, *, context=None, parameter_overrides=None) -> None:
        super().__init__(
            "depth_camera_manager",
            context=context,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        for name, default in {
            **MANAGER_PARAMETER_DEFAULTS,
            **CONFIG_PARAMETER_DEFAULTS,
        }.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        self._parameter_lock = RLock()
        self._desired_values: dict[str, object] = {}
        self._pending_restarts: set[str] = set()
        self._single_unmapped_serial: str | None = None
        self.discovery_interval = self._positive_float("discovery_interval_sec")
        self.retry_interval = self._positive_float("retry_interval_sec")
        serial_no = self._value("serial_no")
        if not isinstance(serial_no, str):
            raise ValueError("serial_no must be a string")
        self.serial_no = serial_no.strip().lstrip("_")
        single_camera_name = self._value("single_camera_name")
        if not isinstance(single_camera_name, str):
            raise ValueError("single_camera_name must be a string")
        self.single_camera_name = camera_name_token(single_camera_name)
        self.config = self._read_config()
        self._validate_camera_overrides()
        self.add_on_set_parameters_callback(self._on_parameters)

    def _value(self, name: str):
        return self.get_parameter(name).value

    def _positive_float(self, name: str, value=None) -> float:
        value = self._value(name) if value is None else value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return float(value)

    def _integer(self, name: str, minimum: int, maximum: int, value=None) -> int:
        value = self._value(name) if value is None else value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        return value

    @staticmethod
    def _boolean(name: str, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value

    @staticmethod
    def _number(name: str, value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)

    @staticmethod
    def _profile(name: str, value):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return parse_profile(value, name)

    def _effective_value(
        self,
        name: str,
        token: str | None,
        proposed: dict[str, object] | None,
    ):
        proposed = proposed or self._desired_values
        if token is not None:
            specific = f"cameras.{token}.{name}"
            if specific in proposed:
                value = proposed[specific]
                if value is not _INHERIT:
                    return value
            elif self.has_parameter(specific):
                return self._value(specific)
        if name in proposed and proposed[name] is not _INHERIT:
            return proposed[name]
        return self._value(name)

    def _read_config(
        self,
        token: str | None = None,
        proposed: dict[str, object] | None = None,
    ) -> CameraConfig:
        def value(name: str):
            return self._effective_value(name, token, proposed)

        namespace = value("camera_namespace")
        if not isinstance(namespace, str) or not namespace.strip("/"):
            raise ValueError("camera_namespace must be a non-empty string")
        return CameraConfig(
            camera_namespace=namespace,
            depth_profile=self._profile(
                "depth_module.depth_profile",
                value("depth_module.depth_profile"),
            ),
            color_profile=self._profile(
                "rgb_camera.color_profile",
                value("rgb_camera.color_profile"),
            ),
            enable_depth=self._boolean("enable_depth", value("enable_depth")),
            enable_color=self._boolean("enable_color", value("enable_color")),
            enable_infra1=self._boolean("enable_infra1", value("enable_infra1")),
            enable_infra2=self._boolean("enable_infra2", value("enable_infra2")),
            enable_gyro=self._boolean("enable_gyro", value("enable_gyro")),
            enable_accel=self._boolean("enable_accel", value("enable_accel")),
            rectify_depth=self._boolean(
                "depth_module.rectification.enable",
                value("depth_module.rectification.enable"),
            ),
            rectify_color=self._boolean(
                "rgb_camera.rectification.enable",
                value("rgb_camera.rectification.enable"),
            ),
            rectify_infra1=self._boolean(
                "infra1.rectification.enable",
                value("infra1.rectification.enable"),
            ),
            rectify_infra2=self._boolean(
                "infra2.rectification.enable",
                value("infra2.rectification.enable"),
            ),
            device_timeout_sec=self._positive_float(
                "device_timeout_sec", value("device_timeout_sec")
            ),
            decimation_enabled=self._boolean(
                "decimation_filter.enable", value("decimation_filter.enable")
            ),
            decimation_magnitude=self._integer(
                "decimation_filter.filter_magnitude",
                1,
                8,
                value("decimation_filter.filter_magnitude"),
            ),
            spatial_enabled=self._boolean(
                "spatial_filter.enable", value("spatial_filter.enable")
            ),
            spatial_alpha=self._number(
                "spatial_filter.filter_smooth_alpha",
                value("spatial_filter.filter_smooth_alpha"),
            ),
            spatial_delta=self._number(
                "spatial_filter.filter_smooth_delta",
                value("spatial_filter.filter_smooth_delta"),
            ),
            spatial_holes_fill=self._integer(
                "spatial_filter.holes_fill",
                0,
                5,
                value("spatial_filter.holes_fill"),
            ),
            temporal_enabled=self._boolean(
                "temporal_filter.enable", value("temporal_filter.enable")
            ),
            temporal_alpha=self._number(
                "temporal_filter.filter_smooth_alpha",
                value("temporal_filter.filter_smooth_alpha"),
            ),
            temporal_delta=self._number(
                "temporal_filter.filter_smooth_delta",
                value("temporal_filter.filter_smooth_delta"),
            ),
            temporal_holes_fill=self._integer(
                "temporal_filter.holes_fill",
                0,
                8,
                value("temporal_filter.holes_fill"),
            ),
            hole_filling_enabled=self._boolean(
                "hole_filling_filter.enable", value("hole_filling_filter.enable")
            ),
            hole_filling_mode=self._integer(
                "hole_filling_filter.holes_fill",
                0,
                2,
                value("hole_filling_filter.holes_fill"),
            ),
            second_hole_filling_enabled=self._boolean(
                "second_hole_filling_filter.enable",
                value("second_hole_filling_filter.enable"),
            ),
            second_hole_filling_mode=self._integer(
                "second_hole_filling_filter.holes_fill",
                0,
                2,
                value("second_hole_filling_filter.holes_fill"),
            ),
            orbbec_enable_sdk_filters=self._boolean(
                "orbbec.enable_sdk_filters", value("orbbec.enable_sdk_filters")
            ),
            orbbec_fallback_hfov=self._number(
                "orbbec.fallback_hfov", value("orbbec.fallback_hfov")
            ),
            orbbec_fallback_vfov=self._number(
                "orbbec.fallback_vfov", value("orbbec.fallback_vfov")
            ),
        )

    def _camera_names_from_parameters(self) -> set[str]:
        names = set()
        for suffix in self.get_parameters_by_prefix("cameras"):
            name = f"cameras.{suffix}"
            parts = camera_parameter_parts(name)
            if parts is None:
                raise ValueError(
                    f"invalid per-camera parameter {name!r}; expected "
                    "cameras.<logical_name>.<supported parameter>"
                )
            names.add(parts[0])
        return names

    def _camera_names(self, proposed: dict[str, object] | None = None) -> set[str]:
        names = self._camera_names_from_parameters()
        for parameter_name in proposed or self._desired_values:
            parts = camera_parameter_parts(parameter_name)
            if parts is not None:
                names.add(parts[0])
        return names

    def _camera_serial(
        self,
        logical_name: str,
        proposed: dict[str, object] | None = None,
    ) -> str:
        parameter_name = f"cameras.{logical_name}.serial_no"
        proposed = proposed or self._desired_values
        if parameter_name in proposed:
            value = proposed[parameter_name]
            if value is _INHERIT:
                return ""
        elif self.has_parameter(parameter_name):
            value = self._value(parameter_name)
        else:
            return ""
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{parameter_name} must be a string or integer")
        return str(value).strip()

    def _serial_to_camera_names(
        self, proposed: dict[str, object] | None = None
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for logical_name in self._camera_names(proposed):
            serial = self._camera_serial(logical_name, proposed)
            if not serial:
                continue
            previous = result.get(serial)
            if previous is not None and previous != logical_name:
                raise ValueError(
                    f"camera serial {serial!r} is assigned to both "
                    f"{previous!r} and {logical_name!r}"
                )
            result[serial] = logical_name
        return result

    def camera_name_for(
        self, serial: str, proposed: dict[str, object] | None = None
    ) -> str:
        mapped = self._serial_to_camera_names(proposed).get(serial)
        if mapped is not None:
            return mapped
        if proposed is None and serial == self._single_unmapped_serial:
            return self.single_camera_name
        return topic_token(serial)

    def update_discovered_serials(self, serials: list[str]) -> None:
        """Select the stable fallback name only for one unmapped device."""
        with self._parameter_lock:
            unique = tuple(dict.fromkeys(serial for serial in serials if serial))
            mappings = self._serial_to_camera_names()
            fallback_serial = None
            if len(unique) == 1:
                serial = unique[0]
                fallback_name_reserved = self.single_camera_name in mappings.values()
                if serial not in mappings and not fallback_name_reserved:
                    fallback_serial = serial
            self._single_unmapped_serial = fallback_serial

    def _validate_camera_overrides(self) -> None:
        self._serial_to_camera_names()
        for logical_name in self._camera_names():
            self._read_config(logical_name)

    def config_for(self, serial: str) -> CameraConfig:
        with self._parameter_lock:
            return self._read_config(self.camera_name_for(serial))

    def take_pending_restarts(self) -> set[str]:
        with self._parameter_lock:
            pending = set(self._pending_restarts)
            self._pending_restarts.clear()
            return pending

    def _on_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        with self._parameter_lock:
            proposed = dict(self._desired_values)
            affected: set[str] = set()
            management: dict[str, object] = {}
            try:
                for parameter in parameters:
                    name = parameter.name
                    if name in CONFIG_PARAMETER_DEFAULTS:
                        if parameter.type_ == Parameter.Type.NOT_SET:
                            raise ValueError(
                                f"global parameter {name!r} cannot be unset"
                            )
                        proposed[name] = parameter.value
                        affected.add("*")
                        continue

                    parts = camera_parameter_parts(name)
                    if parts is not None:
                        proposed[name] = (
                            _INHERIT
                            if parameter.type_ == Parameter.Type.NOT_SET
                            else parameter.value
                        )
                        if parts[1] == "serial_no":
                            affected.add("*")
                        else:
                            affected.add(parts[0])
                        continue

                    if name in MANAGER_PARAMETER_DEFAULTS:
                        if parameter.type_ == Parameter.Type.NOT_SET:
                            raise ValueError(
                                f"manager parameter {name!r} cannot be unset"
                            )
                        management[name] = parameter.value
                        continue

                    if name == "use_sim_time":
                        continue
                    if name.startswith("cameras."):
                        raise ValueError(
                            f"invalid per-camera parameter {name!r}; expected "
                            "cameras.<logical_name>.<supported parameter>"
                        )
                    raise ValueError(f"unsupported parameter: {name!r}")

                self._read_config(None, proposed)
                camera_names = self._camera_names(proposed)
                camera_names.update(name for name in affected if name != "*")
                for logical_name in camera_names:
                    self._read_config(logical_name, proposed)
                self._serial_to_camera_names(proposed)

                if "discovery_interval_sec" in management:
                    self._positive_float(
                        "discovery_interval_sec",
                        management["discovery_interval_sec"],
                    )
                if "retry_interval_sec" in management:
                    self._positive_float(
                        "retry_interval_sec", management["retry_interval_sec"]
                    )
                if "serial_no" in management and not isinstance(
                    management["serial_no"], str
                ):
                    raise ValueError("serial_no must be a string")
                if "single_camera_name" in management:
                    value = management["single_camera_name"]
                    if not isinstance(value, str):
                        raise ValueError("single_camera_name must be a string")
                    camera_name_token(value)
            except (TypeError, ValueError) as exc:
                return SetParametersResult(successful=False, reason=str(exc))

            self._desired_values = proposed
            self._pending_restarts.update(affected)
            if "discovery_interval_sec" in management:
                self.discovery_interval = float(management["discovery_interval_sec"])
            if "retry_interval_sec" in management:
                self.retry_interval = float(management["retry_interval_sec"])
            if "serial_no" in management:
                self.serial_no = str(management["serial_no"]).strip().lstrip("_")
                self._pending_restarts.add("*")
            if "single_camera_name" in management:
                self.single_camera_name = camera_name_token(
                    str(management["single_camera_name"])
                )
                self._single_unmapped_serial = None
                self._pending_restarts.add("*")
            self.config = self._read_config(None, proposed)
            return SetParametersResult(successful=True)


class CameraSupervisor:
    def __init__(self, manager: CameraManager, executor: MultiThreadedExecutor) -> None:
        self.manager = manager
        self.executor = executor
        self.workers: dict[tuple[str, str], CameraNode] = {}
        self.failures: dict[tuple[str, str], _Failure] = {}
        self.rs, rs_error = load_sdk("pyrealsense2")
        self.ob, ob_error = load_sdk("pyorbbecsdk")
        self._orbbec_context = None
        providers = []
        bundled = fallback_modules()
        if self.rs is not None:
            providers.append(
                "RealSense:" + ("bundled" if "pyrealsense2" in bundled else "system")
            )
        else:
            manager.get_logger().warning(f"RealSense backend unavailable: {rs_error}")
        if self.ob is not None:
            providers.append(
                "Orbbec:" + ("bundled" if "pyorbbecsdk" in bundled else "system")
            )
            self._orbbec_context = self.ob.Context()
        else:
            manager.get_logger().warning(f"Orbbec backend unavailable: {ob_error}")
        if not providers:
            raise RuntimeError("no supported depth camera SDK is available")
        manager.get_logger().info("camera SDK providers: " + ", ".join(providers))

    def discover(self) -> dict[tuple[str, str], DeviceDescriptor]:
        descriptors: list[DeviceDescriptor] = []
        if self.rs is not None:
            try:
                descriptors.extend(discover_realsense(self.rs))
            except Exception as exc:
                self.manager.get_logger().warning(f"RealSense discovery failed: {exc}")
        if self.ob is not None:
            try:
                descriptors.extend(discover_orbbec(self.ob, self._orbbec_context))
            except Exception as exc:
                self.manager.get_logger().warning(f"Orbbec discovery failed: {exc}")

        selected = [
            descriptor
            for descriptor in descriptors
            if not self.manager.serial_no
            or descriptor.serial == self.manager.serial_no
        ]
        self.manager.update_discovered_serials(
            [descriptor.serial for descriptor in selected]
        )

        discovered: dict[tuple[str, str], DeviceDescriptor] = {}
        topic_owners: dict[str, DeviceDescriptor] = {}
        for descriptor in selected:
            logical_name = self.manager.camera_name_for(descriptor.serial)
            conflict = topic_owners.get(logical_name)
            if conflict is not None and conflict.key != descriptor.key:
                self.manager.get_logger().error(
                    f"camera topic prefix collision: {conflict.key} and "
                    f"{descriptor.key} both map to {logical_name!r}"
                )
                continue
            topic_owners[logical_name] = descriptor
            discovered[descriptor.key] = descriptor
        return discovered

    def reconcile(self) -> None:
        now = time.monotonic()
        pending_restarts = self.manager.take_pending_restarts()
        if pending_restarts:
            for key, worker in tuple(self.workers.items()):
                if "*" in pending_restarts or worker.logical_name in pending_restarts:
                    self._remove(key, "configuration changed")
                    self.failures.pop(key, None)

        discovered = self.discover()
        for key, worker in tuple(self.workers.items()):
            if key not in discovered:
                self._remove(key, "disconnected")
            elif worker.logical_name != self.manager.camera_name_for(key[1]):
                self._remove(key, "logical camera name changed")
                self.failures.pop(key, None)
            elif worker.is_stale(now):
                self._remove(key, "stream timed out")
                self.failures[key] = _Failure(
                    now + self.manager.retry_interval, "", 0.0
                )

        for key, descriptor in discovered.items():
            if key in self.workers:
                continue
            failure = self.failures.get(key)
            if failure is not None and now < failure.retry_at:
                continue
            self._start(descriptor, now)

        for key in tuple(self.failures):
            if key not in discovered and key not in self.workers:
                self.failures.pop(key, None)

    def _start(self, descriptor: DeviceDescriptor, now: float) -> None:
        try:
            logical_name = self.manager.camera_name_for(descriptor.serial)
            config = self.manager.config_for(descriptor.serial)
            if descriptor.backend == "realsense":
                worker = RealSenseCamera(descriptor, logical_name, config, self.rs)
            else:
                worker = OrbbecCamera(descriptor, logical_name, config, self.ob)
            self.executor.add_node(worker)
            self.workers[descriptor.key] = worker
            self.failures.pop(descriptor.key, None)
            identity = f"backend={descriptor.backend}, serial={descriptor.serial}"
            self.manager.get_logger().info(
                f"camera online: {identity}, logical_name={logical_name}, "
                f"topics=/{config.camera_namespace.strip('/')}/{logical_name}"
            )
        except Exception as exc:
            message = str(exc)
            previous = self.failures.get(descriptor.key)
            should_log = (
                previous is None
                or previous.message != message
                or now - previous.last_log >= 30.0
            )
            last_log = now if should_log else previous.last_log
            self.failures[descriptor.key] = _Failure(
                now + self.manager.retry_interval, message, last_log
            )
            if should_log:
                self.manager.get_logger().error(
                    f"camera start failed; retrying: backend={descriptor.backend}, "
                    f"serial={descriptor.serial}: {message}"
                )

    def _remove(self, key: tuple[str, str], reason: str) -> None:
        worker = self.workers.pop(key)
        self.executor.remove_node(worker)
        worker.destroy_node()
        if rclpy.ok():
            self.manager.get_logger().warning(
                f"camera offline: backend={key[0]}, serial={key[1]}, reason={reason}"
            )

    def close(self) -> None:
        for key in tuple(self.workers):
            self._remove(key, "shutdown")
        self._orbbec_context = None


def main(args=None) -> None:
    rclpy.init(args=args)
    manager = CameraManager()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(manager)
    supervisor = CameraSupervisor(manager, executor)
    next_discovery = 0.0
    try:
        while rclpy.ok():
            now = time.monotonic()
            if now >= next_discovery:
                supervisor.reconcile()
                next_discovery = now + manager.discovery_interval
            executor.spin_once(timeout_sec=max(0.0, min(0.1, next_discovery - now)))
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.close()
        executor.remove_node(manager)
        manager.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
