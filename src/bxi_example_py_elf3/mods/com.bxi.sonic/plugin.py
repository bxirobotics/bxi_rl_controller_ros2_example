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
        hardware_gripper=state.bool_param("hardware_gripper", False),
        gripper_enable_interval_s=state.float_param(
            "gripper_enable_interval_s",
            1.0,
        ),
        gripper_left_bus=state.int_param("gripper_left_bus", 5),
        gripper_right_bus=state.int_param("gripper_right_bus", 6),
        gripper_can_id=state.int_param("gripper_can_id", 1),
        gripper_kp=state.float_param("gripper_kp", 20.0),
        gripper_kd=state.float_param("gripper_kd", 1.0),
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
