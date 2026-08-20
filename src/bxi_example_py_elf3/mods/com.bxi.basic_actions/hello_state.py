from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS
from bxi_example_py_elf3.framework.mod_api import (
    EntryFrameProvider,
    JointCommandComposer,
    JointCommandLayer,
    JointLayout,
    JointTargetBuffer,
    MotorFrame,
    ResourceHandle,
    RobotControlState,
    RunningFrameProvider,
    StateBehavior,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


HELLO_WAVE_JOINTS = JointLayout(
    (
        "r_shoulder_y_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
    ),
    label="hello wave command",
)
HELLO_HEAD_JOINTS = JointLayout(
    ("head_y_joint", "head_z_joint"),
    label="hello head command",
)
HELLO_OUTPUT_JOINTS = JointLayout(
    (*ELF3_POLICY_JOINTS.names, *HELLO_HEAD_JOINTS.names),
    label="hello state output",
)


class HelloState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    HELLO_DURATION_FRAMES = 150  # 3 s at the 50 Hz control period
    HEAD_Y_AMPLITUDE = 0.10
    HEAD_Z_AMPLITUDE = 0.20
    HEAD_ANGULAR_SPEED = 1.50
    HEAD_KP = 16.747
    HEAD_KD = 1.066

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self.playing = True
        self.shaketime = 0
        self._head_phase = 0.0
        self._wave_command = JointTargetBuffer(HELLO_WAVE_JOINTS)
        self._head_command = JointTargetBuffer(HELLO_HEAD_JOINTS)
        self._composer: JointCommandComposer | None = None

    @property
    def policy(self) -> HumanoidGaitPolicyLiteIsaaclab:
        return self._policy.get()

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.playing = True
        self.shaketime = 0
        self._head_phase = 0.0
        ctx.preheat_model(self.policy, command=self.get_cmd_vel(ctx))
        self._prepare_command_sources()

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True
        self.shaketime = 0
        self._head_phase = 0.0

    def _prepare_command_sources(self) -> None:
        policy_target = self.policy.output.joints
        for output_index, name in enumerate(HELLO_WAVE_JOINTS.names):
            policy_index = policy_target.layout.index(name)
            self._wave_command.kp[output_index] = policy_target.kp[policy_index]
            self._wave_command.kd[output_index] = policy_target.kd[policy_index]
        self._head_command.kp.fill(self.HEAD_KP)
        self._head_command.kd.fill(self.HEAD_KD)
        self._set_entry_commands()

        composer = self._composer
        if composer is None or composer.layers[0].target is not policy_target:
            self._composer = JointCommandComposer(
                HELLO_OUTPUT_JOINTS,
                (
                    JointCommandLayer("policy", policy_target), #这里的policy_target是policy内部的输出缓冲区，共享同一片内存
                    JointCommandLayer(
                        "wave_controller",
                        self._wave_command.view,
                        override=True,
                    ),
                    JointCommandLayer("head_controller", self._head_command.view),
                ),
            )

    def _set_entry_commands(self) -> None:
        self._wave_command.position[:] = (-0.9, 0.0, -0.3)
        self._head_command.position.fill(0.0)

    def _update_command_sources(self) -> None:
        self._wave_command.position[0] = -0.9
        self._wave_command.position[1] = math.sin(self.shaketime / 10.0) * 0.5
        self._wave_command.position[2] = -0.3

        # This producer can later be replaced by a topic, IK or another
        # controller without changing the composer or the policy.
        self._head_command.position[0] = self.HEAD_Y_AMPLITUDE * math.sin(
            self._head_phase * 0.5
        )
        self._head_command.position[1] = self.HEAD_Z_AMPLITUDE * math.sin(
            self._head_phase
        )

    def _compose(self) -> MotorFrame:
        composer = self._composer
        if composer is None:
            raise RuntimeError("hello command composer has not been prepared")
        return composer.compose()

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        self._set_entry_commands()
        return self._compose()

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        self.get_cmd_vel(ctx)
        self.policy.step(
            ctx.inference_frame,
            dt,
            advance=advance,
        )
        self._update_command_sources()
        frame = self._compose()
        if self.shaketime < 50:
            arm_gain = self.shaketime / 50.0
            for name in HELLO_WAVE_JOINTS.names:
                frame.kp[frame.layout.index(name)] *= arm_gain
        if self.playing and advance:
            self.shaketime += 1
            self._head_phase = math.fmod(
                self._head_phase + self.HEAD_ANGULAR_SPEED * dt,
                math.tau,
            )
        return frame

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
            return
        if self.shaketime >= self.HELLO_DURATION_FRAMES:
            ctx.request_state(
                "com.bxi.basic_actions/normal",
                trigger="hello_finished",
                transition={"profile": "dual_running_blend", "duration": 0.6},
            )
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
