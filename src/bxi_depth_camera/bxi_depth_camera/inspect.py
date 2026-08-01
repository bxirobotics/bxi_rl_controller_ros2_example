from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
import sys
import time
from typing import Any

from .vendor import bootstrap_environment, load_sdk


ORBBEC_VENDOR_ID = 0x2BC5
GEMINI_335_PRODUCT_ID = 0x0800


@dataclass(frozen=True)
class CameraIdentity:
    backend: str
    serial: str
    name: str
    uid: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.backend, self.serial


@dataclass(frozen=True)
class StreamInfo:
    stream_type: str
    width: int
    height: int
    fps: int
    format_name: str = ""


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = ()

    @property
    def hfov_deg(self) -> float:
        if not math.isfinite(self.fx) or self.fx <= 0.0 or self.width <= 0:
            return 0.0
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        if not math.isfinite(self.fy) or self.fy <= 0.0 or self.height <= 0:
            return 0.0
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))


@dataclass
class CameraReport:
    identity: CameraIdentity
    streams: list[StreamInfo] = field(default_factory=list)
    intrinsics: dict[str, Intrinsics] = field(default_factory=dict)
    # Metres represented by one integer step in the original SDK depth frame.
    depth_scale_m: float | None = None
    warnings: list[str] = field(default_factory=list)


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    text = str(value)
    return text.rsplit(".", 1)[-1].lower()


def _realsense_identity(rs, device) -> CameraIdentity:
    return CameraIdentity(
        backend="realsense",
        serial=str(device.get_info(rs.camera_info.serial_number)).strip(),
        name=str(device.get_info(rs.camera_info.name)).strip(),
        uid=str(device.get_info(rs.camera_info.physical_port)).strip(),
    )


def _discover_realsense(rs) -> list[CameraIdentity]:
    result: list[CameraIdentity] = []
    for device in rs.context().query_devices():
        try:
            identity = _realsense_identity(rs, device)
        except Exception:
            continue
        if identity.serial:
            result.append(identity)
    return result


def _realsense_depth_scale(rs, device) -> float | None:
    for sensor in device.query_sensors():
        try:
            if sensor.supports(rs.option.depth_units):
                scale = float(sensor.get_option(rs.option.depth_units))
                if math.isfinite(scale) and scale > 0.0:
                    return scale
        except Exception:
            continue
        try:
            scale = float(sensor.as_depth_sensor().get_depth_scale())
            if math.isfinite(scale) and scale > 0.0:
                return scale
        except Exception:
            continue
    return None


def _start_realsense_pipeline(rs, serial: str):
    errors: list[str] = []
    selections = ((rs.stream.depth, rs.stream.color), (rs.stream.depth,))
    for stream_types in selections:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        for stream_type in stream_types:
            config.enable_stream(stream_type)
        try:
            return pipeline, pipeline.start(config), errors
        except Exception as exc:
            errors.append(
                "cannot start "
                + "+".join(_enum_name(item) for item in stream_types)
                + f" streams: {exc}"
            )
    return None, None, errors


def _inspect_realsense(rs, identity: CameraIdentity) -> CameraReport:
    report = CameraReport(identity)
    context = rs.context()
    device = None
    for candidate in context.query_devices():
        try:
            value = candidate.get_info(rs.camera_info.serial_number)
            serial = str(value).strip()
        except Exception:
            continue
        if serial == identity.serial:
            device = candidate
            break
    if device is None:
        report.warnings.append(
            "device disappeared before its parameters were read"
        )
        return report

    report.depth_scale_m = _realsense_depth_scale(rs, device)
    pipeline, active, start_errors = _start_realsense_pipeline(
        rs, identity.serial
    )
    if active is None:
        report.warnings.extend(start_errors)
        return report
    if len(start_errors) > 0:
        report.warnings.append(start_errors[0] + "; using depth-only fallback")

    try:
        for profile in active.get_streams():
            kind = _enum_name(profile.stream_type())
            if not profile.is_video_stream_profile():
                report.streams.append(
                    StreamInfo(kind, 0, 0, int(profile.fps()))
                )
                continue
            video = profile.as_video_stream_profile()
            report.streams.append(
                StreamInfo(
                    kind,
                    int(video.width()),
                    int(video.height()),
                    int(video.fps()),
                    _enum_name(video.format()),
                )
            )
            try:
                value = video.get_intrinsics()
                report.intrinsics[kind] = Intrinsics(
                    float(value.fx),
                    float(value.fy),
                    float(value.ppx),
                    float(value.ppy),
                    int(value.width),
                    int(value.height),
                    tuple(float(item) for item in value.coeffs),
                )
            except Exception as exc:
                report.warnings.append(f"{kind} intrinsics unavailable: {exc}")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    return report


def _orbbec_identity(info) -> CameraIdentity:
    return CameraIdentity(
        backend="orbbec",
        serial=str(info.get_serial_number()).strip(),
        name=str(info.get_name()).strip(),
        uid=str(info.get_uid()).strip(),
    )


def _discover_orbbec(ob) -> list[CameraIdentity]:
    context = ob.Context()
    devices = context.query_devices()
    result: list[CameraIdentity] = []
    for index in range(devices.get_count()):
        try:
            info = devices.get_device_by_index(index).get_device_info()
            if (
                int(info.get_vid()) != ORBBEC_VENDOR_ID
                or int(info.get_pid()) != GEMINI_335_PRODUCT_ID
            ):
                continue
            identity = _orbbec_identity(info)
        except Exception:
            continue
        if identity.serial:
            result.append(identity)
    return result


def _orbbec_intrinsics(profile) -> Intrinsics:
    video = profile.as_video_stream_profile()
    value = video.get_intrinsic()
    distortion = video.get_distortion()
    return Intrinsics(
        float(value.fx),
        float(value.fy),
        float(value.cx),
        float(value.cy),
        int(value.width),
        int(value.height),
        (
            float(distortion.k1),
            float(distortion.k2),
            float(distortion.p1),
            float(distortion.p2),
            float(distortion.k3),
        ),
    )


def _orbbec_stream(profile, kind: str) -> StreamInfo:
    video = profile.as_video_stream_profile()
    return StreamInfo(
        kind,
        int(video.get_width()),
        int(video.get_height()),
        int(video.get_fps()),
        _enum_name(video.get_format()),
    )


def _start_orbbec_pipeline(ob, device):
    errors: list[str] = []
    for include_color in (True, False):
        pipeline = ob.Pipeline(device)
        config = ob.Config()
        profiles: dict[str, Any] = {}
        try:
            depth = pipeline.get_stream_profile_list(
                ob.OBSensorType.DEPTH_SENSOR
            ).get_default_video_stream_profile()
            profiles["depth"] = depth
            config.enable_stream(depth)
            if include_color:
                color = pipeline.get_stream_profile_list(
                    ob.OBSensorType.COLOR_SENSOR
                ).get_default_video_stream_profile()
                profiles["color"] = color
                config.enable_stream(color)
            pipeline.start(config)
            return pipeline, profiles, errors
        except Exception as exc:
            errors.append(
                f"cannot start {'depth+color' if include_color else 'depth'} "
                f"streams: {exc}"
            )
            try:
                pipeline.stop()
            except Exception:
                pass
    return None, {}, errors


def _inspect_orbbec(ob, identity: CameraIdentity) -> CameraReport:
    report = CameraReport(identity)
    context = ob.Context()
    devices = context.query_devices()
    try:
        device = devices.get_device_by_serial_number(identity.serial)
    except Exception as exc:
        report.warnings.append(f"device disappeared before inspection: {exc}")
        return report

    pipeline, profiles, start_errors = _start_orbbec_pipeline(ob, device)
    if pipeline is None:
        report.warnings.extend(start_errors)
        return report
    if len(start_errors) > 0:
        report.warnings.append(start_errors[0] + "; using depth-only fallback")

    try:
        for kind, profile in profiles.items():
            report.streams.append(_orbbec_stream(profile, kind))
            try:
                report.intrinsics[kind] = _orbbec_intrinsics(profile)
            except Exception as exc:
                report.warnings.append(f"{kind} intrinsics unavailable: {exc}")

        try:
            frames = pipeline.wait_for_frames(3000)
            depth = frames.get_depth_frame() if frames is not None else None
            if depth is not None:
                # Orbbec reports millimetres represented by one raw depth unit.
                scale_mm = float(depth.get_depth_scale())
                if math.isfinite(scale_mm) and scale_mm > 0.0:
                    report.depth_scale_m = scale_mm / 1000.0
            else:
                report.warnings.append(
                    "no depth frame arrived within 3 seconds"
                )
        except Exception as exc:
            report.warnings.append(
                f"depth conversion scale unavailable: {exc}"
            )
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    return report


def _load_backends():
    environment = bootstrap_environment(("pyrealsense2", "pyorbbecsdk"))
    if environment is not None:
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "bxi_depth_camera.inspect", *sys.argv[1:]],
            environment,
        )
    rs, rs_error = load_sdk("pyrealsense2")
    ob, ob_error = load_sdk("pyorbbecsdk")
    return rs, ob, rs_error, ob_error


def _scan(rs, ob) -> tuple[dict[tuple[str, str], CameraIdentity], list[str]]:
    identities: dict[tuple[str, str], CameraIdentity] = {}
    errors: list[str] = []
    for name, backend, discover in (
        ("RealSense", rs, _discover_realsense),
        ("Orbbec", ob, _discover_orbbec),
    ):
        if backend is None:
            continue
        try:
            for identity in discover(backend):
                identities[identity.key] = identity
        except Exception as exc:
            errors.append(f"{name} enumeration failed: {exc}")
    return identities, errors


def _inspect(identity: CameraIdentity, rs, ob) -> CameraReport:
    if identity.backend == "realsense":
        return _inspect_realsense(rs, identity)
    return _inspect_orbbec(ob, identity)


def _print_report(report: CameraReport) -> None:
    identity = report.identity
    print(f"\n[{identity.backend.upper()}] {identity.name}")
    print(f"  序列号 (SN)       : {identity.serial}")
    if identity.uid:
        print(f"  物理端口 / UID    : {identity.uid}")

    if report.streams:
        print("  输出流:")
        for stream in report.streams:
            if stream.width > 0 and stream.height > 0:
                print(
                    f"    {stream.stream_type:<8} "
                    f"{stream.width}x{stream.height} @ {stream.fps} FPS  "
                    f"format={stream.format_name or 'unknown'}"
                )
            else:
                print(f"    {stream.stream_type:<8} @ {stream.fps} FPS")
    else:
        print("  输出流            : 未读取（相机可能正被其他程序占用）")

    if report.depth_scale_m is not None:
        print(
            "  深度转换比例      : "
            f"raw x {report.depth_scale_m:.9g} m "
            f"({report.depth_scale_m * 1000.0:.6g} mm / unit)"
        )
    else:
        print("  深度转换比例      : 未读取")

    for kind in ("depth", "color", "infrared", "infra1", "infra2"):
        value = report.intrinsics.get(kind)
        if value is None:
            continue
        print(
            f"  {kind} FOV         : H {value.hfov_deg:.2f} deg x "
            f"V {value.vfov_deg:.2f} deg"
        )
        print(
            f"    内参             : fx={value.fx:.3f}, fy={value.fy:.3f}, "
            f"cx={value.cx:.3f}, cy={value.cy:.3f}"
        )

    for warning in report.warnings:
        print(f"  警告               : {warning}")


def _matching(
    identities: dict[tuple[str, str], CameraIdentity], serial: str
) -> dict[tuple[str, str], CameraIdentity]:
    if not serial:
        return identities
    return {
        key: identity
        for key, identity in identities.items()
        if identity.serial == serial
    }


def _print_backend_errors(
    rs, ob, rs_error: str | None, ob_error: str | None
) -> None:
    if rs is None:
        print(f"提示: RealSense SDK 不可用: {rs_error}", file=sys.stderr)
    if ob is None:
        print(f"提示: Orbbec SDK 不可用: {ob_error}", file=sys.stderr)


def _run_once(args) -> int:
    rs, ob, rs_error, ob_error = _load_backends()
    _print_backend_errors(rs, ob, rs_error, ob_error)
    if rs is None and ob is None:
        return 2

    identities, errors = _scan(rs, ob)
    for error in errors:
        print(f"警告: {error}", file=sys.stderr)
    selected = _matching(identities, args.serial)
    if not selected:
        suffix = f"（序列号 {args.serial}）" if args.serial else ""
        print(f"未发现受支持的相机{suffix}。")
        return 1

    for identity in sorted(selected.values(), key=lambda item: item.key):
        _print_report(_inspect(identity, rs, ob))
    print(f"\n共发现 {len(selected)} 台相机。")
    return 0


def _run_watch(args) -> int:
    rs, ob, rs_error, ob_error = _load_backends()
    _print_backend_errors(rs, ob, rs_error, ob_error)
    if rs is None and ob is None:
        return 2

    print(
        f"正在监听相机拔插（间隔 {args.interval:g} 秒）"
        + (f"，目标序列号: {args.serial}" if args.serial else "")
        + "；按 Ctrl-C 退出。"
    )
    known: dict[tuple[str, str], CameraIdentity] = {}
    last_errors: tuple[str, ...] = ()
    try:
        while True:
            current, errors = _scan(rs, ob)
            current = _matching(current, args.serial)
            error_tuple = tuple(errors)
            if error_tuple != last_errors:
                for error in errors:
                    print(f"警告: {error}", file=sys.stderr)
                last_errors = error_tuple

            connected = current.keys() - known.keys()
            disconnected = known.keys() - current.keys()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for key in sorted(disconnected):
                identity = known[key]
                print(
                    f"\n[{timestamp}] 已拔出: {identity.backend.upper()} "
                    f"{identity.name}  SN={identity.serial}"
                )
            for key in sorted(connected):
                identity = current[key]
                print(
                    f"\n[{timestamp}] 已接入: {identity.backend.upper()} "
                    f"{identity.name}  SN={identity.serial}"
                )
                _print_report(_inspect(identity, rs, ob))

            known = current
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止监听。")
        return 0


def _positive_interval(value: str) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0.0:
        raise argparse.ArgumentTypeError("interval must be greater than zero")
    return interval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cameras-inspect",
        description="探测深度相机的序列号、输出尺寸、帧率、FOV、内参和深度转换比例。",
    )
    parser.add_argument("--serial", default="", help="只探测指定序列号的相机")
    parser.add_argument(
        "--watch", action="store_true", help="持续监听拔插并显示接入相机的序列号"
    )
    parser.add_argument(
        "--interval",
        type=_positive_interval,
        default=1.0,
        help="--watch 的轮询间隔，单位为秒（默认: 1.0）",
    )
    args = parser.parse_args(argv)
    code = _run_watch(args) if args.watch else _run_once(args)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()


__all__ = [
    "CameraIdentity",
    "CameraReport",
    "Intrinsics",
    "StreamInfo",
    "main",
]
