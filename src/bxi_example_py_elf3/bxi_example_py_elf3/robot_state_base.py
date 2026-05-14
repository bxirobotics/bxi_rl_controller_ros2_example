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
        self._entered_during_transition = False

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
        self._entered_during_transition = False

    def on_transition_commit(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        if self._entered_during_transition:
            self._entered_during_transition = False
            self._prepared_first_frame = None
            return
        self.on_enter(ctx)

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
        elif transition.enter_behavior == "dual_running_blend":
            self._enter_dual_running_blend(ctx, from_state, progress, transition)

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

    def get_motor_frame(self, ctx: BxiExample, dt: float) -> Optional[MotorFrame]:
        return None

    def get_transition_frame(
        self,
        ctx: BxiExample,
        role: str,
        transition: TransitionProfile,
    ) -> Optional[MotorFrame]:
        dt = float(getattr(ctx, "dt", 0.0))
        return self.get_motor_frame(ctx, dt)

    def on_transition_runtime_enter(
        self,
        ctx: BxiExample,
        transition: TransitionProfile,
    ) -> None:
        self.on_enter(ctx)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        frame = self.get_motor_frame(ctx, dt)
        if frame is None:
            return
        ctx.set_motor_target(*frame)

    def reset_loop(self, ctx: BxiExample) -> None:
        ctx.loop_count = 0

    def _enter_for_transition_running(
        self,
        ctx: BxiExample,
        transition: TransitionProfile,
    ) -> None:
        if self._entered_during_transition:
            return

        self.on_transition_runtime_enter(ctx, transition)
        self._entered_during_transition = True

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

    def _enter_dual_running_blend(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        progress: float,
        transition: TransitionProfile,
    ) -> None:
        from_frame = self._resolve_transition_frame(ctx, from_state, "from", transition)
        to_frame = self._resolve_transition_frame(ctx, self, "to", transition)

        if from_frame is None and to_frame is None:
            ctx.hold_last_motor_target()
            return
        if from_frame is None:
            ctx.set_motor_target(*to_frame)
            return
        if to_frame is None:
            ctx.set_motor_target(*from_frame)
            return

        alpha = self._transition_alpha(progress, transition)
        qpos = from_frame[0] + (to_frame[0] - from_frame[0]) * alpha
        kp = from_frame[1] + (to_frame[1] - from_frame[1]) * alpha
        kd = from_frame[2] + (to_frame[2] - from_frame[2]) * alpha
        ctx.set_motor_target(
            qpos.astype(np.float32),
            kp.astype(np.float32),
            kd.astype(np.float32),
        )

    def _resolve_transition_frame(
        self,
        ctx: BxiExample,
        state: StateBehavior[BxiExample],
        role: str,
        transition: TransitionProfile,
    ) -> Optional[MotorFrame]:
        if self._transition_bool(transition, f"run_{role}", True):
            if role == "to":
                enter_for_transition = getattr(
                    state, "_enter_for_transition_running", None
                )
                if callable(enter_for_transition):
                    enter_for_transition(ctx, transition)

            sampler = getattr(state, "get_transition_frame", None)
            if callable(sampler):
                frame = sampler(ctx, role, transition)
                if frame is not None:
                    return self._motor_frame(*frame)

        return self._fallback_transition_frame(ctx, state, role, transition)

    def _fallback_transition_frame(
        self,
        ctx: BxiExample,
        state: StateBehavior[BxiExample],
        role: str,
        transition: TransitionProfile,
    ) -> Optional[MotorFrame]:
        default_mode = "last_motor" if role == "from" else "first_frame"
        mode = str(transition.data.get(f"{role}_fallback", default_mode))

        if mode in ("none", "disabled", "disable"):
            return None
        if mode in ("last_motor", "hold_last", "hold_last_motor"):
            return self._motor_frame(ctx.pos_last, ctx.kp_last, ctx.kd_last)
        if mode == "first_frame":
            first_frame = None
            getter = getattr(state, "get_first_frame", None)
            if callable(getter):
                first_frame = getter(ctx)
            if first_frame is None:
                return None
            return self._motor_frame(*first_frame)

        raise ValueError(f"unsupported transition frame fallback mode: {mode}")

    def _transition_alpha(
        self,
        progress: float,
        transition: TransitionProfile,
    ) -> float:
        alpha = min(max(float(progress), 0.0), 1.0)
        curve = str(transition.data.get("curve", "linear"))
        if curve == "linear":
            return alpha
        if curve == "smoothstep":
            return alpha * alpha * (3.0 - 2.0 * alpha)
        if curve == "smootherstep":
            return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
        raise ValueError(f"unsupported transition blend curve: {curve}")

    def _transition_bool(
        self,
        transition: TransitionProfile,
        key: str,
        default: bool,
    ) -> bool:
        value = transition.data.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "yes", "on", "1"):
                return True
            if normalized in ("false", "no", "off", "0"):
                return False
        raise ValueError(f"transition data '{key}' must be a bool: {value}")

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
