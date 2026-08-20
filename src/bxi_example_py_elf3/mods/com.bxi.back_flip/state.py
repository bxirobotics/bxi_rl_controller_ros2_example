from bxi_example_py_elf3.policies import DanceMotionPolicyGravityIsaaclab
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api import MotionReplayState


class BackFlipState(MotionReplayState[DanceMotionPolicyGravityIsaaclab]):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[DanceMotionPolicyGravityIsaaclab],
    ) -> None:
        super().__init__(
            name,
            state_id,
            policy,
            finish_state="com.bxi.basic_actions/normal",
            finish_trigger="back_flip_finished",
            end_frame_trim=30,
            end_transition={
                "profile": "dual_running_blend",
                "duration": 0.45,
                "curve": "linear",
                "sample_from": True,
            },
        )
