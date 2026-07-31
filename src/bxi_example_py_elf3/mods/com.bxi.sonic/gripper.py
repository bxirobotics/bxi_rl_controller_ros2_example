#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
import communication.msg as bxiMsg


@dataclass
class JointControl:
    """Motor joint command data."""

    p_des: float = 0.0
    v_des: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    t_ff: float = 0.0


class BxiMotor:
    @staticmethod
    def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
        """Convert float value to unsigned integer with given bit width."""
        if bits <= 0:
            raise ValueError("bits must be positive")

        span = x_max - x_min
        if span <= 0:
            raise ValueError("x_max must be greater than x_min")

        # Clamp
        x = max(min(x, x_max), x_min)

        max_val = (1 << bits) - 1

        # Handle boundaries explicitly to avoid floating point errors
        if x == x_min:
            return 0
        if x == x_max:
            return max_val

        # Linear mapping with rounding
        result = (x - x_min) * max_val / span
        return int(round(result))

    @staticmethod
    def fmaxf(x: float, y: float) -> float:
        return max(x, y)

    @staticmethod
    def fminf(x: float, y: float) -> float:
        return min(x, y)

    @staticmethod
    def fmaxf3(x: float, y: float, z: float) -> float:
        return max(x, y, z)

    @staticmethod
    def fminf3(x: float, y: float, z: float) -> float:
        return min(x, y, z)

    @staticmethod
    def limit_norm(x: float, y: float, limit: float) -> tuple[float, float]:
        """Scale vector (x, y) length to be <= limit."""
        norm = math.sqrt(x * x + y * y)
        if norm > limit and norm > 0:
            x = x * limit / norm
            y = y * limit / norm
        return x, y

    @staticmethod
    def zero() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]
        return data

    @staticmethod
    def enter_motor_mode() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
        return data

    @staticmethod
    def exit_motor_mode() -> list[int]:
        data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
        return data

    @staticmethod
    def pack_cmd(
        joint: JointControl,
        p_range: tuple[float, float],
        v_range: tuple[float, float],
        t_range: tuple[float, float],
        kp_range: tuple[float, float],
        kd_range: tuple[float, float],
    ) -> list[int]:
        """Pack motor command into 8 bytes.

        Equivalent to the C function:
        pack_cmd(uint8_t *data, joint_control *joint, ...)
        """
        p_min, p_max = p_range
        v_min, v_max = v_range
        t_min, t_max = t_range
        kp_min, kp_max = kp_range
        kd_min, kd_max = kd_range

        p_des = min(max(p_min, joint.p_des), p_max)
        v_des = min(max(v_min, joint.v_des), v_max)
        kp = min(max(kp_min, joint.kp), kp_max)
        kd = min(max(kd_min, joint.kd), kd_max)
        t_ff = min(max(t_min, joint.t_ff), t_max)

        p_int = BxiMotor.float_to_uint(p_des, p_min, p_max, 16)
        v_int = BxiMotor.float_to_uint(v_des, v_min, v_max, 12)
        kp_int = BxiMotor.float_to_uint(kp, kp_min, kp_max, 12)
        kd_int = BxiMotor.float_to_uint(kd, kd_min, kd_max, 12)
        t_int = BxiMotor.float_to_uint(t_ff, t_min, t_max, 12)

        data = [0] * 8
        data[0] = (p_int >> 8) & 0xFF
        data[1] = p_int & 0xFF
        data[2] = (v_int >> 4) & 0xFF
        data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
        data[4] = kp_int & 0xFF
        data[5] = (kd_int >> 4) & 0xFF
        data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
        data[7] = t_int & 0xFF
        return data

    @staticmethod
    def build_motor_packet(bus, canid, data: list[int]):
        packet_type = getattr(
            bxiMsg, "CANFDPacket", getattr(bxiMsg, "CanfdPacket", None)
        )
        if packet_type is None:
            raise RuntimeError("communication.msg.CANFDPacket is unavailable")
        packet = packet_type()
        packet.bus = int(bus)
        packet.frame.can_id = int(canid)
        packet.frame.flags = int(0x01 | 0x04)
        packet.frame.len = len(data)
        values = list(data)
        try:
            packet.frame.data = values
            return packet
        except Exception:
            pass

        padded = values + [0] * (64 - len(values))
        try:
            packet.frame.data = padded
            return packet
        except Exception:
            pass

        for index, value in enumerate(values):
            packet.frame.data[index] = value
        return packet
