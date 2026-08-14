from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
    StateBuildContext,
)

from .policy import SonicTeleopPolicy
from .state import SonicTeleopState


SONIC_POLICY = ResourceKey[SonicTeleopPolicy]("com.bxi.sonic/policy")


def _load_policy(context: ResourceLoadContext) -> SonicTeleopPolicy:
    return SonicTeleopPolicy(
        str(context.asset("assets/sonic.onnx")),
        str(context.asset("assets/stream_reference.npz")),
    )


def _build_state(
    state: StateBuildContext,
    policy,
) -> SonicTeleopState:
    return SonicTeleopState(
        state.name,
        state.state_id,
        policy,
        require_live_reference=state.bool_param(
            "require_live_reference",
            False,
        ),
        yaw_bias_rad=state.float_param("yaw_bias_rad", 1.57079632679),
        live_reference_timeout_s=state.float_param("live_reference_timeout_s", 0.5),
        idle_frame_start=state.int_param("idle_frame_start", 3509),
        source_blend_seconds=state.float_param("source_blend_seconds", 0.4),
        head_control_enabled=state.bool_param("head_control_enabled", True),
        head_pitch_limit_rad=state.float_param("head_pitch_limit_rad", 0.5),
        head_yaw_limit_rad=state.float_param("head_yaw_limit_rad", 1.0),
        head_pitch_speed_rad_s=state.float_param(
            "head_pitch_speed_rad_s", 1.5
        ),
        head_yaw_speed_rad_s=state.float_param("head_yaw_speed_rad_s", 2.0),
        head_deadband_rad=state.float_param("head_deadband_rad", 0.015),
        hardware_gripper=state.bool_param("hardware_gripper", False),
        gripper_enable_interval_s=state.float_param(
            "gripper_enable_interval_s",
            1.0,
        ),
        gripper_left_bus=state.int_param("gripper_left_bus", 5),
        gripper_right_bus=state.int_param("gripper_right_bus", 6),
        gripper_can_id=state.int_param("gripper_can_id", 1),
        gripper_master_id=state.int_param("gripper_master_id", 0x11),
        gripper_kp=state.float_param("gripper_kp", 20.0),
        gripper_kd=state.float_param("gripper_kd", 1.0),
        gripper_calibration_speed_rad_s=state.float_param(
            "gripper_calibration_speed_rad_s",
            0.2,
        ),
        gripper_calibration_kp=state.float_param("gripper_calibration_kp", 5.0),
        gripper_calibration_kd=state.float_param("gripper_calibration_kd", 0.5),
        gripper_contact_torque=state.float_param("gripper_contact_torque", 2.0),
        gripper_abort_torque=state.float_param("gripper_abort_torque", 8.0),
        gripper_contact_confirm_s=state.float_param(
            "gripper_contact_confirm_s",
            0.25,
        ),
        gripper_stopped_velocity_rad_s=state.float_param(
            "gripper_stopped_velocity_rad_s",
            0.1,
        ),
        gripper_tracking_error_rad=state.float_param(
            "gripper_tracking_error_rad",
            0.08,
        ),
        gripper_limit_margin_rad=state.float_param(
            "gripper_limit_margin_rad",
            0.15,
        ),
        gripper_minimum_span_rad=state.float_param(
            "gripper_minimum_span_rad",
            1.0,
        ),
        gripper_maximum_search_travel_rad=state.float_param(
            "gripper_maximum_search_travel_rad",
            7.0,
        ),
        gripper_response_timeout_s=state.float_param(
            "gripper_response_timeout_s",
            1.0,
        ),
        gripper_feedback_timeout_s=state.float_param(
            "gripper_feedback_timeout_s",
            0.3,
        ),
        gripper_phase_timeout_s=state.float_param(
            "gripper_phase_timeout_s",
            45.0,
        ),
        gripper_maximum_mos_temperature_c=state.int_param(
            "gripper_maximum_mos_temperature_c",
            80,
        ),
        gripper_maximum_motor_temperature_c=state.int_param(
            "gripper_maximum_motor_temperature_c",
            80,
        ),
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(SONIC_POLICY, _load_policy, policy="startup")
    policy = context.resource(SONIC_POLICY)
    return ModDefinition(
        state_factories={
            "sonic_teleop": lambda state: _build_state(state, policy),
        }
    )


__all__ = ["SONIC_POLICY", "create_mod"]
