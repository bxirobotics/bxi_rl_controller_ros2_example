from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.inference.amp import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.mod_api import ResourceHandle
from bxi_example_py_elf3.mod_api import RobotControlState
from bxi_example_py_elf3.mod_api import NORMAL_STATE, ZERO_TORQUE_STATE
from bxi_example_py_elf3.mod_api import StateBehavior
from bxi_example_py_elf3.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.mod_api import RobotControlContext


@dataclass(frozen=True)
class PlaybackClip:
    positions: NDArray[np.float32]
    fps: float


def load_clip(path: Path, *, start_frame: int, tail_trim_frames: int) -> PlaybackClip:
    with path.open("rb") as input_file:
        raw = pickle.load(input_file)
    if not isinstance(raw, dict):
        raise ValueError(f"playback clip must contain a map: {path}")
    positions = np.asarray(raw["dof_pos"], dtype=np.float32)[:, -14:]
    start = min(start_frame, positions.shape[0])
    end = max(start, positions.shape[0] - tail_trim_frames)
    positions = positions[start:end]
    if positions.shape[0] == 0:
        raise ValueError(f"playback clip is empty after trimming: {path}")
    return PlaybackClip(positions, float(raw["fps"]))


class ApplauseState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
        clip: ResourceHandle[PlaybackClip],
    ) -> None:
        super().__init__(name, state_id)
        self._policy = policy
        self._clip = clip
        self.frame = 0.0
        self.playing = True

    @property
    def policy(self) -> HumanoidGaitPolicyLiteIsaaclab:
        return self._policy.get()

    @property
    def clip(self) -> PlaybackClip:
        return self._clip.get()

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.frame = 0.0
        self.playing = True
        ctx.preheat_model(self.policy, with_cmd_vel=True, cmd_vel=self.get_cmd_vel(ctx))

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.frame = 0.0
        self.playing = True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        qpos = self.policy.target.copy()
        qpos[-14:] = self.clip.positions[0]
        return self._motor_frame(qpos, self.policy.kp, self.policy.kd)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        qpos, _ = self.policy.step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            self.get_cmd_vel(ctx),
        )
        index = min(int(self.frame), self.clip.positions.shape[0] - 1)
        qpos[-14:] = self.clip.positions[index]
        if self.playing and advance:
            self.frame += self.clip.fps * dt
        return self._motor_frame(qpos, self.policy.kp, self.policy.kd)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(ZERO_TORQUE_STATE, trigger="safety")
            return
        if self.frame >= self.clip.positions.shape[0]:
            ctx.request_state(
                NORMAL_STATE,
                trigger="applause_finished",
                transition={"profile": "dual_running_blend", "duration": 1.0},
            )
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
