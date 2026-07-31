"""ZMQ wire-format helpers used by the ELF3 SONIC PICO bridge."""

from __future__ import annotations

import json

import numpy as np

HEADER_SIZE = 1280


def _build_header(fields: list[dict], version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def _dtype_name(value: np.ndarray) -> tuple[str, np.ndarray]:
    if value.dtype == np.float32:
        return "f32", value
    if value.dtype == np.float64:
        return "f64", value
    if value.dtype == np.int32:
        return "i32", value
    if value.dtype == np.int64:
        return "i64", value
    if value.dtype == np.uint8:
        return "u8", value
    if value.dtype == bool:
        return "bool", value
    return "f32", value.astype(np.float32)


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    fields = []
    binary_data = []

    for key, raw_value in pose_data.items():
        if not isinstance(raw_value, np.ndarray):
            continue

        dtype_str, value = _dtype_name(raw_value)
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        binary_data.append(value.tobytes())

    return (
        topic.encode("utf-8")
        + _build_header(fields, version=version)
        + b"".join(binary_data)
    )
