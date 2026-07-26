from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.beyondmimic import DanceMotionPolicyGravityIsaaclabV2
from bxi_example_py_elf3.mod_api import ResourceHandle
from bxi_example_py_elf3.mod_api import MotionReplayState

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


PD_BRAKE_STATE = "com.bxi.basic_actions/pd_brake"


class LieDownState(MotionReplayState[DanceMotionPolicyGravityIsaaclabV2]):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[DanceMotionPolicyGravityIsaaclabV2],
    ) -> None:
        super().__init__(
            name,
            state_id,
            policy,
            finish_trigger="lie_down_finished",
            end_frame_trim=225,
            end_transition={
                "profile": "dual_running_blend",
                "duration": 1.0,
                "curve": "smootherstep",
                "sample_from": True,
            },
        )

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        policy = self.policy
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        if policy.timestep > policy.end_frame - self.end_frame_trim:
            ctx.request_state(
                PD_BRAKE_STATE,
                trigger=self.finish_trigger,
                transition=self.end_transition,
            )

    # def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
    #     if action_name != "toggle_pause":
    #         return False
    #     self.playing = not self.playing
    #     return True
