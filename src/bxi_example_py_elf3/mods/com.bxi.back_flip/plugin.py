from bxi_example_py_elf3.policies import DanceMotionPolicyGravityIsaaclab
from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)
from .state import BackFlipState


POLICY = ResourceKey[DanceMotionPolicyGravityIsaaclab]("com.bxi.back_flip/policy")


def _load(context: ResourceLoadContext) -> DanceMotionPolicyGravityIsaaclab:
    return DanceMotionPolicyGravityIsaaclab(
        str(context.asset("assets/back_flip.npz")),
        str(context.asset("assets/back_flip.onnx")),
        start_frame=40,
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(POLICY, _load, policy="on_demand")
    policy = context.resource(POLICY)
    return ModDefinition(
        state_factories={
            "back_flip": lambda state: BackFlipState(state.name, state.state_id, policy)
        }
    )
