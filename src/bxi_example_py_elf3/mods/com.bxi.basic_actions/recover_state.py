from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bxi_example_py_elf3.inference.beyondmimic import DanceMotionPolicyMjlab
from bxi_example_py_elf3.mod_api import ResourceHandle
from bxi_example_py_elf3.mod_api import RobotControlState
from bxi_example_py_elf3.mod_api import NORMAL_STATE, ZERO_TORQUE_STATE
from bxi_example_py_elf3.mod_api import StateBehavior
from bxi_example_py_elf3.mod_api.geometry import quaternion_to_euler_array
from bxi_example_py_elf3.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


class RecoverState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[DanceMotionPolicyMjlab]
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy
        self.playing = True
        self.motion_selected = False
        self.end_frame_trim = 0

    @property
    def policy(self) -> DanceMotionPolicyMjlab:
        return self._policy.get()

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.playing = True
        if self._configure_motion(ctx):
            ctx.preheat_model(self.policy)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True
        if not self.motion_selected:
            ctx.request_state(ZERO_TORQUE_STATE, trigger="recover_pose_rejected")

    def _configure_motion(self, ctx: RobotControlContext) -> bool:
        angles = quaternion_to_euler_array(ctx.quat_xyzw)
        angles[angles > math.pi] -= 2 * math.pi
        if angles[1] < -(math.pi / 4.0):
            self.policy.end_frame = 880
            self.policy.timestep = 600
            self.policy.start_frame = 600
            self.end_frame_trim = 20
        elif angles[1] > (math.pi / 4.0):
            self.policy.end_frame = 1690
            self.policy.timestep = 1350
            self.policy.start_frame = 1350
            self.end_frame_trim = 0
        else:
            self.motion_selected = False
            return False
        self.motion_selected = True
        return True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        if not self.motion_selected:
            return self._motor_frame(ctx.pos_last, ctx.kp_last, ctx.kd_last)
        return self._motor_frame(
            self.policy.target_dof_pos, self.policy.kps, self.policy.kds
        )

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame | None:
        if self.policy.timestep > self.policy.end_frame:
            return None
        qpos = self.policy.inference_step(
            ctx.current_q, ctx.current_dq, ctx.current_quat_wxyz, ctx.current_omega
        )
        if self.playing and advance:
            self.policy.timestep += 50 * dt
        return self._motor_frame(qpos, self.policy.kps, self.policy.kds)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if self.policy.timestep > self.policy.end_frame - self.end_frame_trim:
            ctx.request_state(
                NORMAL_STATE,
                trigger="recover_finished",
                transition={
                    "profile": "dual_running_blend",
                    "duration": 0.5,
                    "sample_from": True,
                },
            )
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
