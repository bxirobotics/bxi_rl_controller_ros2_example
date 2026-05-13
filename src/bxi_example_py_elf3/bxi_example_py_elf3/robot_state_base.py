from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from bxi_example_py_elf3.state_machine import StateBehavior, TransitionProfile

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
else:
    BxiExample = Any

MotorFrame = Tuple[np.ndarray, np.ndarray, np.ndarray]


class RobotControlState(StateBehavior[BxiExample]):
    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self._prepared_first_frame: Optional[MotorFrame] = None

    def on_exit(self, ctx: BxiExample) -> None:
        ctx.pos_last_state = ctx.qpos.copy()
        ctx.kp_last_state = ctx.kp_last.copy()
        ctx.kd_last_state = ctx.kd_last.copy()
        self._prepared_first_frame = None

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        self._prepared_first_frame = None

    def on_enter_transition(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        progress: float,
        transition: TransitionProfile,
    ) -> None:
        if transition.enter_behavior == "hold_last_motor":
            ctx.hold_last_motor_target()
        elif transition.enter_behavior == "first_frame_ramp_kp":
            self._enter_first_frame_ramp_kp(ctx, progress, transition)

    def on_exit_transition(
        self,
        ctx: BxiExample,
        to_state: StateBehavior[BxiExample],
        progress: float,
        transition: TransitionProfile,
    ) -> None:
        if transition.exit_behavior == "hold_last_motor":
            ctx.hold_last_motor_target()

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return None

    def reset_loop(self, ctx: BxiExample) -> None:
        ctx.loop_count = 0

    def _enter_first_frame_ramp_kp(
        self,
        ctx: BxiExample,
        progress: float,
        transition: TransitionProfile,
    ) -> None:
        if self._prepared_first_frame is None:
            first_frame = self.get_first_frame(ctx)
            if first_frame is None:
                ctx.hold_last_motor_target()
                return
            self._prepared_first_frame = self._motor_frame(*first_frame)

        qpos, kp_target, kd_target = self._prepared_first_frame
        alpha = min(max(float(progress), 0.0), 1.0)
        kp_start_mode = str(transition.data.get("kp_start", "current"))
        kd_start_mode = str(transition.data.get("kd_start", "target"))
        kp_start = self._gain_start(kp_start_mode, kp_target, ctx.kp_last)
        kd_start = self._gain_start(kd_start_mode, kd_target, ctx.kd_last)
        kp = kp_start + (kp_target - kp_start) * alpha
        kd = kd_start + (kd_target - kd_start) * alpha
        ctx.set_motor_target(qpos, kp.astype(np.float32), kd.astype(np.float32))

    def _gain_start(
        self,
        mode: str,
        target: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        if mode == "target":
            return target.copy()
        if mode == "zero":
            return np.zeros_like(target)
        if mode != "current":
            raise ValueError(f"unsupported transition gain start mode: {mode}")

        current_array = np.asarray(current, dtype=np.float32)
        if current_array.shape != target.shape:
            raise ValueError(
                f"current gain shape {current_array.shape} does not match target shape {target.shape}"
            )
        return current_array.copy()

    def _motor_frame(self, qpos, kp, kd) -> MotorFrame:
        return (
            np.asarray(qpos, dtype=np.float32).copy(),
            np.asarray(kp, dtype=np.float32).copy(),
            np.asarray(kd, dtype=np.float32).copy(),
        )
