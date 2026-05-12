import math
import os
import pickle

import numpy as np

from ament_index_python.packages import get_package_share_path
from bxi_example_py_elf3.state_machine import StateBehavior
from bxi_example_py_elf3.utils.tfs import quaternion_to_euler_array


class RobotControlState(StateBehavior):
    def on_exit(self, ctx) -> None:
        ctx.pos_last_state = ctx.qpos.copy()
        ctx.kp_last_state = ctx.kp_last.copy()
        ctx.kd_last_state = ctx.kd_last.copy()

    def on_enter_transition(self, ctx, from_state, progress, transition) -> None:
        if transition.enter_behavior == "hold_last_motor":
            ctx.hold_last_motor_target()

    def on_exit_transition(self, ctx, to_state, progress, transition) -> None:
        if transition.exit_behavior == "hold_last_motor":
            ctx.hold_last_motor_target()

    def reset_loop(self, ctx) -> None:
        ctx.loop_count = 0


class NormalState(RobotControlState):
    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.normal, with_cmd_vel=True)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)

    def on_update(self, ctx, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        qpos, vel = ctx.normal.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            ctx.current_cmd_vel,
        )
        ctx.set_motor_target(qpos, ctx.normal.kps, ctx.normal.kds)


class ZeroTorqueState(RobotControlState):
    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)

    def on_update(self, ctx, dt: float) -> None:
        ctx.set_motor_target(
            ctx.joint_nominal_pos,
            np.zeros(ctx.dof_num, dtype=np.float32),
            np.zeros(ctx.dof_num, dtype=np.float32),
        )


class PdBrakeState(RobotControlState):
    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)

    def on_update(self, ctx, dt: float) -> None:
        soft_start = min(ctx.loop_count / (2.0 / ctx.dt), 1.0)
        qpos = ctx.pos_last_state + (ctx.pd_pos - ctx.pos_last_state) * soft_start
        kp = ctx.kp_last + (ctx.normal.kps - ctx.kp_last) * soft_start
        kd = ctx.kd_last + (ctx.normal.kds - ctx.kd_last) * soft_start
        ctx.set_motor_target(qpos, kp, kd)


class InitialPosState(RobotControlState):
    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)

    def on_update(self, ctx, dt: float) -> None:
        soft_start = min(ctx.loop_count / (2.0 / ctx.dt), 1.0)
        qpos = ctx.pos_last_state + (ctx.initial_pos - ctx.pos_last_state) * soft_start
        kp = ctx.kp_last + (ctx.joint_kp - ctx.kp_last) * soft_start
        kd = ctx.kd_last + (ctx.joint_kd - ctx.kd_last) * soft_start
        ctx.set_motor_target(qpos, kp, kd)


class DanceState(RobotControlState):
    def __init__(self, name: str, state_id: int, start_frame: int = 100):
        super().__init__(name, state_id)
        self.start_frame = start_frame
        self.playing = True

    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.dance)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        self.playing = True
        ctx.dance.timestep = self.start_frame

    def on_update(self, ctx, dt: float) -> None:
        if ctx.dance.timestep >= ctx.dance.motionpos.shape[0]:
            print("Motion replay finished, resetting simulation.")
            ctx.dance.timestep = self.start_frame
            ctx.request_state("normal", trigger="motion_finished")
            return

        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        qpos = ctx.dance.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )
        ctx.set_motor_target(qpos, ctx.dance.stiffness_array, ctx.dance.damping_array)

        if self.playing:
            ctx.dance.timestep += 1

    def on_action(self, ctx, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class FlipState(RobotControlState):
    policy_attr = ""
    finish_trigger = "flip_finished"
    end_frame_trim = 0

    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.playing = True
        self.policy_configured = False

    def _policy(self, ctx):
        return getattr(ctx, self.policy_attr)

    def _configure_policy(self, policy) -> None:
        if self.policy_configured:
            return
        if self.end_frame_trim > 0:
            policy.end_frame = max(policy.start_frame, policy.end_frame - self.end_frame_trim)
        self.policy_configured = True

    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        policy = self._policy(ctx)
        self._configure_policy(policy)
        policy.timestep = policy.start_frame
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0
        ctx.preheat_model(policy)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        self.playing = True
        policy = self._policy(ctx)
        policy.timestep = policy.start_frame
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0

    def on_update(self, ctx, dt: float) -> None:
        policy = self._policy(ctx)

        if policy.timestep <= policy.end_frame:
            qpos = policy.inference_step(
                ctx.current_q,
                ctx.current_dq,
                ctx.current_quat_wxyz,
                ctx.current_omega,
            )
            ctx.set_motor_target(qpos, policy.kps, policy.kds)

            if self.playing:
                policy.timestep += 1

        if policy.timestep > policy.end_frame:
            print("Motion replay finished, resetting simulation.")
            policy.timestep = policy.start_frame
            ctx.request_state("normal", trigger=self.finish_trigger, transition="soft_switch")


class BackFlipState(FlipState):
    policy_attr = "back_flip"
    finish_trigger = "back_flip_finished"
    end_frame_trim = 20


class ForwardFlipState(FlipState):
    policy_attr = "forward_flip"
    finish_trigger = "forward_flip_finished"


class ApplauseState(RobotControlState):
    def __init__(self, name, state_id):
        super().__init__(name, state_id)
        self.start_frame = 600
        self.tail_trim_frames = 600
        self.return_time = 0.5
        self.frame = 0.0
        self.return_elapsed = 0.0
        self.returning = False
        self.return_start_pos = np.zeros(14, dtype=np.float32)
        self.applause_data, self.fps = self._load_applause_data()

    def _load_applause_data(self):
        try:
            data_path = os.path.join(
                get_package_share_path("bxi_example_py_elf3"),
                "data",
                "applause.pkl",
            )
        except Exception:
            data_path = ""

        if not data_path or not os.path.exists(data_path):
            package_root = os.path.dirname(os.path.dirname(__file__))
            data_path = os.path.join(package_root, "data", "applause.pkl")

        with open(data_path, "rb") as data_file:
            data = pickle.load(data_file)

        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)[:, -14:]
        start = min(self.start_frame, dof_pos.shape[0])
        end = max(start, dof_pos.shape[0] - self.tail_trim_frames)
        applause_data = dof_pos[start:end]
        if applause_data.shape[0] == 0:
            raise ValueError(f"applause data is empty after frame trim: {data_path}")

        return applause_data, float(data["fps"])

    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.noarm, with_cmd_vel=True)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        self.frame = 0.0
        self.return_elapsed = 0.0
        self.returning = False
        self.return_start_pos = self.applause_data[0].copy()
        self.playing = True

    def on_update(self, ctx, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        qpos, vel = ctx.noarm.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            ctx.current_cmd_vel,
        )

        if self.returning:
            alpha = min(1.0, self.return_elapsed / self.return_time)
            qpos[-14:] = self.return_start_pos + (qpos[-14:] - self.return_start_pos) * alpha
            self.return_elapsed += dt
            if self.return_elapsed >= self.return_time:
                ctx.set_motor_target(qpos, ctx.noarm.kps, ctx.noarm.kds)
                ctx.request_state("normal", trigger="applause_finished", transition="soft_switch")
                return
        else:
            frame_index = int(self.frame)
            if frame_index >= self.applause_data.shape[0]:
                self.returning = True
                self.return_elapsed = 0.0
                self.return_start_pos = self.applause_data[-1].copy()
                qpos[-14:] = self.return_start_pos
            else:
                qpos[-14:] = self.applause_data[frame_index]
                if self.playing:
                    self.frame += self.fps * dt

        ctx.set_motor_target(qpos, ctx.noarm.kps, ctx.noarm.kds)
    def on_action(self, ctx, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True

class HelloState(RobotControlState):
    def __init__(self, name, state_id):
        super().__init__(name, state_id)
    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.noarm, with_cmd_vel=True)
    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        self.playing = True
        self.shaketime = 0
    def on_update(self, ctx, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("zero_torque", trigger="safety")
            return
        if self.shaketime < 50:
            self.kp = self.shaketime/50 * ctx.noarm.kps
        qpos, vel = ctx.noarm.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            ctx.current_cmd_vel,
        )
        qpos[22] = -0.9
        qpos[24] = math.sin(self.shaketime/10) * 0.5
        qpos[25] = -0.3
        ctx.set_motor_target(qpos, self.kp, ctx.noarm.kds)
        if self.playing:
            self.shaketime += 1
    def on_action(self, ctx, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True

class RecoverState(RobotControlState):
    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.playing = True

    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.recover)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        self.playing = True
        eu_ang = quaternion_to_euler_array(ctx.quat_xyzw)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        if eu_ang[1] < -(math.pi / 4.0):
            ctx.recover.end_frame = 880
            ctx.recover.timestep = 600
            ctx.recover.start_frame = 600
        elif eu_ang[1] > (math.pi / 4.0):
            ctx.recover.end_frame = 1690
            ctx.recover.timestep = 1350
            ctx.recover.start_frame = 1350
        else:
            ctx.request_state("zero_torque", trigger="recover_pose_rejected")

    def on_update(self, ctx, dt: float) -> None:
        if ctx.recover.timestep > ctx.recover.end_frame:
            ctx.recover.timestep = ctx.recover.start_frame
            ctx.request_state("normal", trigger="recover_finished")
            return

        qpos = ctx.recover.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )
        ctx.set_motor_target(qpos, ctx.recover.kps, ctx.recover.kds)

        if self.playing:
            ctx.recover.timestep += 1


class AmpRunState(RobotControlState):
    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])

    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.amp_run, with_cmd_vel=True)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])

    def on_update(self, ctx, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        self.cmd_vel_run[:2] = (
            0.98 * self.pre_cmd_vel_run[:2] + 0.02 * ctx.current_cmd_vel[:2]
        )
        self.cmd_vel_run[2] = ctx.current_cmd_vel[2]
        qpos, vel = ctx.amp_run.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            self.cmd_vel_run,
        )

        if vel[0] > self.max_vel:
            self.max_vel = vel[0]
        if ctx.loop_count >= 100 + int(0.3 / ctx.dt):
            print(self.max_vel)
            ctx.loop_count = int(0.3 / ctx.dt)
            self.max_vel = 0.0

        self.pre_cmd_vel_run = self.cmd_vel_run.copy()
        ctx.set_motor_target(qpos, ctx.amp_run.kps, ctx.amp_run.kds)


class NormalRunState(RobotControlState):
    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.normal_run, with_cmd_vel=True)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        if hasattr(ctx.normal_run, "action"):
            ctx.normal_run.action = np.zeros_like(ctx.normal_run.action)

    def on_update(self, ctx, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        qpos = ctx.normal_run.infer_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_xyzw,
            ctx.current_omega,
            ctx.current_cmd_vel,
        )
        ctx.set_motor_target(
            qpos,
            ctx.normal_run.joint_stiffness,
            ctx.normal_run.joint_damping,
        )


def _state_behavior_classes():
    def walk_subclasses(base_class):
        for subclass in base_class.__subclasses__():
            yield subclass
            yield from walk_subclasses(subclass)

    return {cls.__name__: cls for cls in walk_subclasses(RobotControlState)}


def _allocate_state_id(state_name, state_config, used_ids, next_id):
    configured_id = (state_config or {}).get("id")
    if configured_id is not None:
        state_id = int(configured_id)
        if state_id in used_ids:
            raise ValueError(f"duplicate state id {state_id} for state: {state_name}")
        used_ids.add(state_id)
        next_id = max(next_id, state_id + 1)
        return state_id, next_id

    while next_id in used_ids:
        next_id += 1
    state_id = next_id
    used_ids.add(state_id)
    return state_id, next_id + 1


def build_robot_states(config):
    states_config = config.get("states", {})
    if not states_config:
        raise ValueError("state machine config must define states")

    behavior_classes = _state_behavior_classes()
    states = {}
    used_ids = set()
    next_id = 0

    for state_name, state_config in states_config.items():
        state_config = state_config or {}
        behavior_name = state_config.get("behavior")
        if not behavior_name:
            raise ValueError(f"state '{state_name}' must define behavior")

        behavior_class = behavior_classes.get(behavior_name)
        if behavior_class is None:
            raise ValueError(f"unknown state behavior '{behavior_name}' for state '{state_name}'")

        state_id, next_id = _allocate_state_id(state_name, state_config, used_ids, next_id)
        params = state_config.get("params", {}) or {}
        states[state_name] = behavior_class(state_name, state_id, **params)

    return states
