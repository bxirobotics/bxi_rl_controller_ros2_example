import math
import os
import pickle
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import communication.msg as bxiMsg
import std_msgs.msg
from rclpy.qos import QoSProfile

from ament_index_python.packages import get_package_share_path
from bxi_example_py_elf3.utils.bxi_motor import BxiMotor, JointControl as BxiJointControl
from bxi_example_py_elf3.utils.robot_state_base import MotorFrame, RobotControlState
from bxi_example_py_elf3.utils.state_machine import StateBehavior, TransitionProfile
from bxi_example_py_elf3.utils.tfs import quaternion_to_euler_array

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
else:
    BxiExample = Any


class NormalState(RobotControlState):
    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.normal,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.normal.target_dof_pos, ctx.normal.kps, ctx.normal.kds
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        cmd_vel = self.get_cmd_vel(ctx)
        qpos, vel = ctx.normal.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )
        return self._motor_frame(qpos, ctx.normal.kps, ctx.normal.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)


class SonicTeleopState(RobotControlState):
    def __init__(
        self,
        name: str,
        state_id: int,
        hardware_gripper: bool = False,
        gripper_input_timeout_s: float = 0.2,
        gripper_release_threshold: float = 0.05,
    ) -> None:
        super().__init__(name, state_id)
        if isinstance(hardware_gripper, str):
            hardware_gripper = hardware_gripper.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self.hardware_gripper_requested = bool(hardware_gripper)
        self.gripper_input_timeout_s = float(gripper_input_timeout_s)
        if (
            not math.isfinite(self.gripper_input_timeout_s)
            or self.gripper_input_timeout_s <= 0.0
        ):
            raise ValueError("gripper_input_timeout_s must be finite and positive")
        self.gripper_release_threshold = float(gripper_release_threshold)
        if not math.isfinite(self.gripper_release_threshold):
            raise ValueError("gripper_release_threshold must be finite")
        self.gripper_release_threshold = float(
            np.clip(self.gripper_release_threshold, 0.0, 1.0)
        )
        self.gripper_enabled = False
        self.gripper_armed = False
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.left_trigger_received_at: Optional[float] = None
        self.right_trigger_received_at: Optional[float] = None
        self._gripper_session_active = False
        self._gripper_session_started_at: Optional[float] = None
        self._gripper_arm_wait_reason: Optional[str] = None
        self._stale_trigger_sides: set[str] = set()

    def on_bind(self, ctx: BxiExample) -> None:
        super().on_bind(ctx)
        # The state-machine mode is the only runtime opt-in.  Even the explicit
        # gripper state is forbidden from publishing CAN outside hardware/.
        self.gripper_enabled = (
            self.hardware_gripper_requested
            and getattr(ctx, "topic_prefix", "") == "hardware/"
        )
        if not self.gripper_enabled:
            return

        try:
            self.gripper_left_bus = int(
                os.environ.get("BXI_SONIC_GRIPPER_LEFT_BUS", "5")
            )
            self.gripper_right_bus = int(
                os.environ.get("BXI_SONIC_GRIPPER_RIGHT_BUS", "6")
            )
            self.gripper_can_id = int(
                os.environ.get("BXI_SONIC_GRIPPER_CAN_ID", "1")
            )
            self.gripper_kp = float(
                os.environ.get("BXI_SONIC_GRIPPER_KP", "20")
            )
            self.gripper_kd = float(
                os.environ.get("BXI_SONIC_GRIPPER_KD", "1")
            )
            if min(
                self.gripper_left_bus,
                self.gripper_right_bus,
                self.gripper_can_id,
            ) < 0 or not all(
                math.isfinite(value)
                for value in (self.gripper_kp, self.gripper_kd)
            ):
                raise ValueError("bus/CAN ID must be non-negative and gains finite")
        except ValueError as exc:
            self.gripper_enabled = False
            ctx.get_logger().error(f"SONIC gripper disabled: invalid config: {exc}")
            return
        self.gripper_msg_type = getattr(
            bxiMsg, "CANFDPacket", getattr(bxiMsg, "CanfdPacket", None)
        )
        if self.gripper_msg_type is None:
            self.gripper_enabled = False
            ctx.get_logger().error(
                "SONIC gripper disabled: communication.msg.CANFDPacket is unavailable"
            )
            return

        qos = QoSProfile(depth=1)
        self.left_trigger_sub = ctx.create_subscription(
            std_msgs.msg.Float32,
            "pico/left_trigger",
            self.left_trigger_callback,
            qos,
        )
        self.right_trigger_sub = ctx.create_subscription(
            std_msgs.msg.Float32,
            "pico/right_trigger",
            self.right_trigger_callback,
            qos,
        )
        self.gripper_control_pub = ctx.create_publisher(
            self.gripper_msg_type,
            "canfd_packet/tx",
            QoSProfile(depth=100),
        )

    def left_trigger_callback(self, msg: std_msgs.msg.Float32) -> None:
        value = self._valid_trigger(msg.data)
        if value is None or not self._gripper_session_active:
            return
        self.left_trigger = value
        self.left_trigger_received_at = time.monotonic()

    def right_trigger_callback(self, msg: std_msgs.msg.Float32) -> None:
        value = self._valid_trigger(msg.data)
        if value is None or not self._gripper_session_active:
            return
        self.right_trigger = value
        self.right_trigger_received_at = time.monotonic()

    @staticmethod
    def _valid_trigger(value: float) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _finite_trigger(value: float, previous: float) -> float:
        trigger = SonicTeleopState._valid_trigger(value)
        return float(previous) if trigger is None else trigger

    def _start_gripper_session(self) -> None:
        if not self.gripper_enabled:
            return
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.left_trigger_received_at = None
        self.right_trigger_received_at = None
        self.gripper_armed = False
        self._gripper_session_active = True
        self._gripper_session_started_at = time.monotonic()
        self._gripper_arm_wait_reason = None
        self._stale_trigger_sides.clear()

    def _publish_gripper_enter(self) -> None:
        if not self.gripper_enabled or not hasattr(self, "gripper_control_pub"):
            return
        for bus in (self.gripper_left_bus, self.gripper_right_bus):
            self.gripper_control_pub.publish(
                BxiMotor.build_motor_packet(
                    bus,
                    self.gripper_can_id,
                    BxiMotor.enter_motor_mode(),
                )
            )

    def _trigger_is_fresh(self, received_at: Optional[float], now: float) -> bool:
        return bool(
            received_at is not None
            and 0.0 <= now - received_at <= self.gripper_input_timeout_s
        )

    def _log_arm_wait(self, ctx: BxiExample, reason: str, message: str) -> None:
        if self._gripper_arm_wait_reason == reason:
            return
        self._gripper_arm_wait_reason = reason
        ctx.get_logger().warning(message)

    def _try_arm_gripper(self, ctx: BxiExample, now: float) -> bool:
        if self.gripper_armed:
            return True

        inputs_fresh = self._trigger_is_fresh(
            self.left_trigger_received_at, now
        ) and self._trigger_is_fresh(self.right_trigger_received_at, now)
        if not inputs_fresh:
            started_at = self._gripper_session_started_at
            if (
                started_at is not None
                and now - started_at >= self.gripper_input_timeout_s
            ):
                self._log_arm_wait(
                    ctx,
                    "input",
                    "SONIC夹爪等待PICO trigger新数据；夹爪尚未解锁",
                )
            return False

        if (
            self.left_trigger > self.gripper_release_threshold
            or self.right_trigger > self.gripper_release_threshold
        ):
            self._log_arm_wait(
                ctx,
                "release",
                "SONIC夹爪等待左右PICO trigger松开；夹爪尚未解锁",
            )
            return False

        self._publish_gripper_enter()
        self.gripper_armed = True
        self._gripper_arm_wait_reason = None
        self._stale_trigger_sides.clear()
        ctx.get_logger().info("SONIC夹爪已解锁：左右电机进入motor mode")
        return True

    def _monitor_gripper_input(self, ctx: BxiExample, now: float) -> None:
        received_at = {
            "left": self.left_trigger_received_at,
            "right": self.right_trigger_received_at,
        }
        stale_now = {
            side
            for side, timestamp in received_at.items()
            if not self._trigger_is_fresh(timestamp, now)
        }
        newly_stale = stale_now - self._stale_trigger_sides
        recovered = self._stale_trigger_sides - stale_now
        if newly_stale:
            sides = ",".join(sorted(newly_stale))
            ctx.get_logger().warning(
                f"SONIC夹爪PICO trigger断流：{sides}；保持最后位置"
            )
        if recovered:
            sides = ",".join(sorted(recovered))
            ctx.get_logger().info(f"SONIC夹爪PICO trigger已恢复：{sides}")
        self._stale_trigger_sides = stale_now

    def _publish_gripper_command(self, bus: int, trigger: float) -> None:
        if (
            not self.gripper_enabled
            or not self.gripper_armed
            or not hasattr(self, "gripper_control_pub")
        ):
            return
        trigger = float(np.clip(trigger, 0.0, 1.0))
        self.gripper_control_pub.publish(
            BxiMotor.build_motor_packet(
                bus,
                self.gripper_can_id,
                BxiMotor.pack_cmd(
                    joint=BxiJointControl(
                        p_des=float((1.0 - trigger) * 0.5 - 0.1),
                        v_des=0.0,
                        kp=self.gripper_kp,
                        kd=self.gripper_kd,
                        t_ff=0.0,
                    ),
                    p_range=(-12.5, 12.5),
                    v_range=(-45.0, 45.0),
                    t_range=(-40.0, 40.0),
                    kp_range=(0.0, 500.0),
                    kd_range=(0.0, 5.0),
                ),
            )
        )

    def _update_gripper(self, ctx: BxiExample) -> None:
        if not self.gripper_enabled or not self._gripper_session_active:
            return
        now = time.monotonic()
        if not self._try_arm_gripper(ctx, now):
            return
        self._monitor_gripper_input(ctx, now)
        self._publish_gripper_command(self.gripper_left_bus, self.left_trigger)
        self._publish_gripper_command(self.gripper_right_bus, self.right_trigger)

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        self._start_gripper_session()
        ctx.sonic_teleop.reset()
        ctx.preheat_model(ctx.sonic_teleop)

    def on_enter(self, ctx: BxiExample) -> None:
        if self.gripper_enabled and not self._gripper_session_active:
            self._start_gripper_session()
        message = getattr(
            ctx,
            "sonic_connection_message",
            "机器人IP：未检测到，请检查机器人网络",
        )
        mode = (
            "SONIC遥操（夹爪）"
            if self.hardware_gripper_requested
            else "SONIC遥操"
        )
        ctx.get_logger().info(
            f"{mode}已启动：{message}；PICO按ABXY完成校准，再按A+X切入实时POSE"
        )

    def on_exit(self, ctx: BxiExample) -> None:
        self._gripper_session_active = False
        self.gripper_armed = False
        self._gripper_arm_wait_reason = None
        self._stale_trigger_sides.clear()
        super().on_exit(ctx)

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.sonic_teleop.target_dof_pos,
            ctx.sonic_teleop.kps,
            ctx.sonic_teleop.kds,
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        qpos = ctx.sonic_teleop.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )
        return self._motor_frame(qpos, ctx.sonic_teleop.kps, ctx.sonic_teleop.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        eu_ang = quaternion_to_euler_array(ctx.current_quat_xyzw)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        orientation_limit = math.radians(180.0)

        if (np.abs(eu_ang[0]) > orientation_limit or np.abs(eu_ang[1]) > orientation_limit):
            print("sonic teleop orientation unsafe, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)
        self._update_gripper(ctx)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "reset_sonic_alignment":
            return False
        ctx.sonic_teleop.reset_yaw_alignment()
        return True


class ZeroTorqueState(RobotControlState):
    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.joint_nominal_pos,
            np.zeros(ctx.dof_num, dtype=np.float32),
            np.zeros(ctx.dof_num, dtype=np.float32),
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.joint_nominal_pos,
            np.zeros(ctx.dof_num, dtype=np.float32),
            np.zeros(ctx.dof_num, dtype=np.float32),
        )


class PdBrakeState(RobotControlState):
    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(ctx.pd_pos, ctx.normal.kps, ctx.normal.kds)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        return self._motor_frame(ctx.pd_pos, ctx.normal.kps, ctx.normal.kds)


class InitialPosState(RobotControlState):
    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(ctx.initial_pos, ctx.joint_kp, ctx.joint_kd)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        return self._motor_frame(ctx.initial_pos, ctx.joint_kp, ctx.joint_kd)


class DanceState(RobotControlState):
    def __init__(self, name: str, state_id: int, start_frame: int = 100):
        super().__init__(name, state_id)
        self.start_frame = start_frame
        self.playing = True

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.dance.timestep = self.start_frame
        if hasattr(ctx.dance, "timeinit"):
            ctx.dance.timeinit = 0.0
        ctx.preheat_model(ctx.dance)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        ctx.dance.timestep = self.start_frame

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.dance.target_dof_pos,
            ctx.dance.kps,
            ctx.dance.kds,
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        if ctx.dance.timestep >= ctx.dance.motionpos.shape[0]:
            return None

        qpos = ctx.dance.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )

        if self.playing:
            ctx.dance.timestep += 50 * dt  # 模型动画是50hz播放的，dt是推理间隔

        return self._motor_frame(
            qpos,
            ctx.dance.kps,
            ctx.dance.kds,
        )

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.dance.timestep >= ctx.dance.motionpos.shape[0]:
            print("Motion replay finished, resetting simulation.")
            ctx.dance.timestep = self.start_frame
            ctx.request_state(
                "normal",
                trigger="motion_finished",
                transition={
                    "base": "dual_running_blend",
                    "duration": 0.5,
                    "data": {"run_from": False},
                },
            )
            return

        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class MotionState(RobotControlState):
    policy_attr = ""
    finish_trigger = "flip_finished"
    end_frame_trim = 0
    end_transition = {}

    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.playing = True

    def _policy(self, ctx: BxiExample) -> Any:
        return getattr(ctx, self.policy_attr)

    def on_enter_transition(self, ctx, from_state, progress, transition):
        policy = self._policy(ctx)
        policy.timestep = policy.start_frame
        return super().on_enter_transition(ctx, from_state, progress, transition)

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        policy = self._policy(ctx)
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0
        ctx.preheat_model(policy)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        policy = self._policy(ctx)
        policy.timestep = policy.start_frame
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        policy = self._policy(ctx)
        qpos = getattr(policy, "target_dof_pos", None)
        if qpos is None:
            qpos = getattr(policy, "default_dof_pos", None)
        if qpos is None:
            return None
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        policy = self._policy(ctx)

        qpos = policy.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )

        if self.playing and not on_translation:
            policy.timestep += 50 * dt  # 模型动画是50hz播放的，dt是推理间隔

        return self._motor_frame(qpos, policy.kps, policy.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        policy = self._policy(ctx)

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

        if policy.timestep > policy.end_frame - self.end_frame_trim:
            print("Motion replay finished, resetting simulation.")
            ctx.request_state(
                "normal", trigger=self.finish_trigger, transition=self.end_transition
            )


class ForwardFlipState(MotionState):
    policy_attr = "forward_flip"
    finish_trigger = "forward_flip_finished"
    end_frame_trim = 125
    end_transition = {
        "base": "dual_running_blend",
        "duration": 1.0,
        "data": {
            "curve": "smootherstep",
            "run_from": True,
        },  # 过渡的时候模型继续推理，同时推理下一个模型
    }


class HandPlayBackState(RobotControlState):
    start_frame = 0
    tail_trim_frames = 0
    return_time = 0.5
    file_name = "applause.pkl"

    def __init__(self, name, state_id):
        super().__init__(name, state_id)
        self.frame = 0.0
        self.applause_data, self.fps = self._load_applause_data()

    def _load_applause_data(self) -> tuple[np.ndarray, float]:
        data_path = os.path.join(
            get_package_share_path("bxi_example_py_elf3"),
            "data",
            self.file_name,
        )
        with open(data_path, "rb") as data_file:
            data = pickle.load(data_file)

        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)[:, -14:]
        start = min(self.start_frame, dof_pos.shape[0])
        end = max(start, dof_pos.shape[0] - self.tail_trim_frames)
        applause_data = dof_pos[start:end]
        if applause_data.shape[0] == 0:
            raise ValueError(
                f"HandPlayBack data is empty after frame trim: {data_path}"
            )

        return applause_data, float(data["fps"])

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.withoutarm,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.frame = 0.0
        self.playing = True

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        qpos = ctx.withoutarm.target_dof_pos.copy()
        qpos[-14:] = self.applause_data[0]
        return self._motor_frame(qpos, ctx.withoutarm.kps, ctx.withoutarm.kds)

    def get_motor_frame(self, ctx, dt, on_translation):
        cmd_vel = self.get_cmd_vel(ctx)
        qpos, vel = ctx.withoutarm.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )
        if self.frame < self.applause_data.shape[0]:
            qpos[-14:] = self.applause_data[int(self.frame)]
        else:
            qpos[-14:] = self.applause_data[-1]
        if self.playing and not on_translation:
            self.frame += self.fps * dt
        return self._motor_frame(qpos, ctx.withoutarm.kps, ctx.withoutarm.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("zero_torque", trigger="safety")
            return
        if self.frame >= self.applause_data.shape[0]:
            ctx.request_state(
                "normal",
                trigger="applause_finished",
                transition={
                    "base": "dual_running_blend",
                    "duration": 1.0,
                },
            )
            return
        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class ApplauseState(HandPlayBackState):
    start_frame = 600
    tail_trim_frames = 600
    file_name = "isaaclab_model/applause.pkl"


class HelloState(RobotControlState):
    def __init__(self, name, state_id):
        super().__init__(name, state_id)

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.withoutarm,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        self.shaketime = 0

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        qpos = ctx.withoutarm.target_dof_pos.copy()
        qpos[22] = -0.9
        qpos[24] = 0.0
        qpos[25] = -0.3
        return self._motor_frame(qpos, ctx.withoutarm.kps, ctx.withoutarm.kds)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        if self.shaketime < 50:
            self.kp = self.shaketime / 50 * ctx.withoutarm.kps
        cmd_vel = self.get_cmd_vel(ctx)
        qpos, vel = ctx.withoutarm.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )
        qpos[22] = -0.9
        qpos[24] = math.sin(self.shaketime / 10) * 0.5
        qpos[25] = -0.3
        if self.playing:
            self.shaketime += 1
        return self._motor_frame(qpos, self.kp, ctx.withoutarm.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("zero_torque", trigger="safety")
            return
        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class RecoverState(RobotControlState):
    end_frame_trim = 0

    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.playing = True
        self.motion_selected = False

    def on_enter_transition(self, ctx, from_state, progress, transition):
        ctx.recover.timestep = ctx.recover.start_frame
        return super().on_enter_transition(ctx, from_state, progress, transition)

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        if self._configure_recover_motion(ctx):
            ctx.preheat_model(ctx.recover)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        if not self._configure_recover_motion(ctx):
            ctx.request_state("zero_torque", trigger="recover_pose_rejected")

    def _configure_recover_motion(self, ctx: BxiExample) -> bool:
        eu_ang = quaternion_to_euler_array(ctx.quat_xyzw)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        if eu_ang[1] < -(math.pi / 4.0):
            # 躺地上
            ctx.recover.end_frame = 880
            ctx.recover.timestep = 600
            ctx.recover.start_frame = 600
            self.end_frame_trim = 20
            self.motion_selected = True
            return True
        elif eu_ang[1] > (math.pi / 4.0):
            # 趴地上
            ctx.recover.end_frame = 1690
            ctx.recover.timestep = 1350
            ctx.recover.start_frame = 1350
            self.end_frame_trim = 0
            self.motion_selected = True
            return True

        self.motion_selected = False
        return False

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        if not self.motion_selected:
            return None
        return self._motor_frame(
            ctx.recover.target_dof_pos, ctx.recover.kps, ctx.recover.kds
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        if ctx.recover.timestep > ctx.recover.end_frame:
            return None

        qpos = ctx.recover.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )

        if self.playing:
            ctx.recover.timestep += 50 * dt  # 模型动画是50hz播放的，dt是推理间隔
        return self._motor_frame(qpos, ctx.recover.kps, ctx.recover.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.recover.timestep > ctx.recover.end_frame - self.end_frame_trim:
            ctx.request_state(
                "normal",
                trigger="recover_finished",
                transition={
                    "base": "dual_running_blend",
                    "duration": 0.5,
                    "data": {"run_from": True},  #
                },
            )
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)


class AmpRunState(RobotControlState):
    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.amp_run,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.amp_run.target_dof_pos, ctx.amp_run.kps, ctx.amp_run.kds
        )

    def process_cmd_vel(
        self,
        ctx: BxiExample,
        cmd_vel: np.ndarray,
    ) -> Optional[np.ndarray]:
        self.cmd_vel_run[:2] = 0.98 * self.pre_cmd_vel_run[:2] + 0.02 * cmd_vel[:2]
        self.cmd_vel_run[2] = cmd_vel[2]
        self.pre_cmd_vel_run = self.cmd_vel_run.copy()
        return self.cmd_vel_run

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        cmd_vel = self.get_cmd_vel(ctx)
        qpos, vel = ctx.amp_run.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )

        if vel[0] > self.max_vel:
            self.max_vel = vel[0]
        if ctx.loop_count >= 100 + int(0.3 / ctx.dt):
            print(self.max_vel)
            ctx.loop_count = int(0.3 / ctx.dt)
            self.max_vel = 0.0

        return self._motor_frame(qpos, ctx.amp_run.kps, ctx.amp_run.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)


class NormalRunState(RobotControlState):
    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.normal_run,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        if hasattr(ctx.normal_run, "action"):
            ctx.normal_run.action = np.zeros_like(ctx.normal_run.action)

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        qpos = ctx.normal_run.default_joint_pos.copy()
        if hasattr(ctx.normal_run, "target_q"):
            qpos += ctx.normal_run.target_q
        return self._motor_frame(
            qpos,
            ctx.normal_run.joint_stiffness,
            ctx.normal_run.joint_damping,
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        cmd_vel = self.get_cmd_vel(ctx)
        qpos = ctx.normal_run.infer_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_xyzw,
            ctx.current_omega,
            cmd_vel,
        )
        return self._motor_frame(
            qpos,
            ctx.normal_run.joint_stiffness,
            ctx.normal_run.joint_damping,
        )

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)
