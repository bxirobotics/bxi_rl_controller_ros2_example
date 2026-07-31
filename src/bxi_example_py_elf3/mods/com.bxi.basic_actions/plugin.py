from bxi_example_py_elf3.policies import (
    HumanoidGaitPolicyLiteIsaaclab,
    DanceMotionPolicyGravityIsaaclabV2,
    DanceMotionPolicyGravityIsaaclabV3,
    DanceMotionPolicyMjlab,
    NormalMotionPolicyMjlab,
)
from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)

from .amp_run_state import AmpRunState
from .applause_state import ApplauseState, PlaybackClip, load_clip
from .dance_state import DanceState
from .hello_state import HelloState
from .initial_pos_state import InitialPosState
from .lie_down_state import LieDownState
from .normal_run_state import NormalRunState
from .normal_state import NormalState
from .pd_brake_state import PdBrakeState
from .recover_state import RecoverState
from .zero_torque_state import ZeroTorqueState


NORMAL_POLICY = ResourceKey[HumanoidGaitPolicyLiteIsaaclab](
    "com.bxi.basic_actions/normal_policy"
)
WITHOUT_ARM_POLICY = ResourceKey[HumanoidGaitPolicyLiteIsaaclab](
    "com.bxi.basic_actions/without_arm_policy"
)
AMP_RUN_POLICY = ResourceKey[HumanoidGaitPolicyLiteIsaaclab](
    "com.bxi.basic_actions/amp_run_policy"
)
NORMAL_RUN_POLICY = ResourceKey[NormalMotionPolicyMjlab](
    "com.bxi.basic_actions/normal_run_policy"
)
DANCE_POLICY = ResourceKey[DanceMotionPolicyGravityIsaaclabV3](
    "com.bxi.basic_actions/dance_policy"
)
LIE_DOWN_POLICY = ResourceKey[DanceMotionPolicyGravityIsaaclabV2](
    "com.bxi.basic_actions/lie_down_policy"
)
RECOVER_POLICY = ResourceKey[DanceMotionPolicyMjlab](
    "com.bxi.basic_actions/recover_policy"
)
APPLAUSE_CLIP = ResourceKey[PlaybackClip]("com.bxi.basic_actions/applause_clip")


def _load_normal_policy(
    context: ResourceLoadContext,
) -> HumanoidGaitPolicyLiteIsaaclab:
    return HumanoidGaitPolicyLiteIsaaclab(
        str(context.asset("assets/amp_terrain.onnx"))
    )


def _load_without_arm_policy(
    context: ResourceLoadContext,
) -> HumanoidGaitPolicyLiteIsaaclab:
    return HumanoidGaitPolicyLiteIsaaclab(
        str(context.asset("assets/withoutarm.onnx"))
    )


def _load_amp_run_policy(
    context: ResourceLoadContext,
) -> HumanoidGaitPolicyLiteIsaaclab:
    return HumanoidGaitPolicyLiteIsaaclab(
        str(context.asset("assets/amp_run.onnx"))
    )


def _load_normal_run_policy(context: ResourceLoadContext) -> NormalMotionPolicyMjlab:
    return NormalMotionPolicyMjlab(str(context.asset("assets/model_normal.onnx")))


def _load_dance_policy(
    context: ResourceLoadContext,
) -> DanceMotionPolicyGravityIsaaclabV3:
    return DanceMotionPolicyGravityIsaaclabV3(
        str(context.asset("assets/shuishou.npz")),
        str(context.asset("assets/shuishou.onnx")),
        start_frame=60,
        fixed_pos=True,
    )


def _load_lie_down_policy(
    context: ResourceLoadContext,
) -> DanceMotionPolicyGravityIsaaclabV2:
    return DanceMotionPolicyGravityIsaaclabV2(
        str(context.asset("assets/lie_down.npz")),
        str(context.asset("assets/lie_down.onnx")),
        start_frame=150,
    )


def _load_recover_policy(context: ResourceLoadContext) -> DanceMotionPolicyMjlab:
    return DanceMotionPolicyMjlab(
        str(context.asset("assets/recover.npz")),
        str(context.asset("assets/recover.onnx")),
        start_frame=600,
    )


def _load_applause_clip(context: ResourceLoadContext) -> PlaybackClip:
    return load_clip(
        context.asset("assets/applause.pkl"),
        start_frame=600,
        tail_trim_frames=600,
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(NORMAL_POLICY, _load_normal_policy)
    context.register_resource(
        WITHOUT_ARM_POLICY, _load_without_arm_policy, policy="on_demand"
    )
    context.register_resource(AMP_RUN_POLICY, _load_amp_run_policy, policy="on_demand")
    context.register_resource(
        NORMAL_RUN_POLICY, _load_normal_run_policy, policy="on_demand"
    )
    context.register_resource(DANCE_POLICY, _load_dance_policy, policy="on_demand")
    context.register_resource(
        LIE_DOWN_POLICY, _load_lie_down_policy, policy="on_demand"
    )
    context.register_resource(RECOVER_POLICY, _load_recover_policy, policy="on_demand")
    context.register_resource(APPLAUSE_CLIP, _load_applause_clip, policy="on_demand")

    normal_policy = context.resource(NORMAL_POLICY)
    without_arm_policy = context.resource(WITHOUT_ARM_POLICY)
    amp_run_policy = context.resource(AMP_RUN_POLICY)
    normal_run_policy = context.resource(NORMAL_RUN_POLICY)
    dance_policy = context.resource(DANCE_POLICY)
    lie_down_policy = context.resource(LIE_DOWN_POLICY)
    recover_policy = context.resource(RECOVER_POLICY)
    applause_clip = context.resource(APPLAUSE_CLIP)

    return ModDefinition(
        state_factories={
            "normal": lambda state: NormalState(
                state.name, state.state_id, normal_policy
            ),
            "zero_torque": lambda state: ZeroTorqueState(
                state.name, state.state_id
            ),
            "pd_brake": lambda state: PdBrakeState(
                state.name, state.state_id, normal_policy
            ),
            "initial_pos": lambda state: InitialPosState(
                state.name, state.state_id
            ),
            "dance": lambda state: DanceState(
                state.name,
                state.state_id,
                dance_policy,
                start_frame=state.int_param("start_frame", 100),
            ),
            "recover": lambda state: RecoverState(
                state.name, state.state_id, recover_policy
            ),
            "amp_run": lambda state: AmpRunState(
                state.name, state.state_id, amp_run_policy
            ),
            "normal_run": lambda state: NormalRunState(
                state.name, state.state_id, normal_run_policy
            ),
            "applause": lambda state: ApplauseState(
                state.name, state.state_id, without_arm_policy, applause_clip
            ),
            "hello": lambda state: HelloState(
                state.name, state.state_id, without_arm_policy
            ),
            "lie_down": lambda state: LieDownState(
                state.name, state.state_id, lie_down_policy
            ),
        }
    )
