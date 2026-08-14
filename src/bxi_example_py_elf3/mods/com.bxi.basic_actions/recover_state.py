from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bxi_example_py_elf3.policies import DanceMotionPolicyMjlab
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.framework.mod_api import StateBehavior
from bxi_example_py_elf3.framework.mod_api.geometry import quaternion_to_euler_array
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class RecoverState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[DanceMotionPolicyMjlab]
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
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
            ctx.request_state(
                "com.bxi.basic_actions/zero_torque",
                trigger="recover_pose_rejected",
            )

    def _configure_motion(self, ctx: RobotControlContext) -> bool:
        angles = quaternion_to_euler_array(ctx.current_quat_xyzw)
        angles[angles > math.pi] -= 2 * math.pi
        if angles[1] < -(math.pi / 4.0):
            self.policy.configure_range(start_frame=600, end_frame=880)
            self.end_frame_trim = 20
        elif angles[1] > (math.pi / 4.0):
            self.policy.configure_range(start_frame=1350, end_frame=1690)
            self.end_frame_trim = 0
        else:
            self.motion_selected = False
            return False
        self.motion_selected = True
        return True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        if not self.motion_selected:
            last = ctx.last_motor_frame
            return self._motor_frame(
                ctx,
                last.qpos,
                last.kp,
                last.kd,
            )
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame | None:
        if self.policy.finished():
            return None
        output = self.policy.step(
            ctx.inference_frame,
            dt,
            advance=self.playing and advance,
        )
        if 500 < self.policy._frame <1350:
            output.joints.position[17] = min(output.joints.position[17]+1,2.8)
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if self.policy.finished(self.end_frame_trim):
            ctx.request_state(
                "com.bxi.basic_actions/normal",
                trigger="recover_finished",
                transition={
                    "profile": "dual_running_blend",
                    "duration": 0.5,
                    "sample_from": True,
                },
            )
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
