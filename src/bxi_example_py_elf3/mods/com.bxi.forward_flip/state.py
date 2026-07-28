from bxi_example_py_elf3.policies import DanceMotionPolicyGravityIsaaclab
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api import MotionReplayState


class ForwardFlipState(MotionReplayState[DanceMotionPolicyGravityIsaaclab]):
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
            finish_trigger="forward_flip_finished",
            end_frame_trim=125,
            end_transition={
                "profile": "dual_running_blend",
                "duration": 1.0,
                "curve": "smootherstep",
                "sample_from": True,
            },
        )
