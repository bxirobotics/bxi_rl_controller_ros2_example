from bxi_example_py_elf3.policies import DanceMotionPolicyGravityIsaaclab
from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)
from .state import ForwardFlipState


POLICY = ResourceKey[DanceMotionPolicyGravityIsaaclab]("com.bxi.forward_flip/policy")


def _load(context: ResourceLoadContext) -> DanceMotionPolicyGravityIsaaclab:
    return DanceMotionPolicyGravityIsaaclab(
        str(context.asset("assets/forward_flip.npz")),
        str(context.asset("assets/forward_flip.onnx")),
        start_frame=150,
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(POLICY, _load, policy="on_demand")
    policy = context.resource(POLICY)
    return ModDefinition(
        state_factories={
            "forward_flip": lambda state: ForwardFlipState(
                state.name, state.state_id, policy
            )
        }
    )
