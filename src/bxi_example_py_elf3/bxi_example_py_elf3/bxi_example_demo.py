import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg
import sensor_msgs.msg
from threading import Lock
import numpy as np
# import torch
import time
import sys
import os
import math
import json
from collections import deque
from std_msgs.msg import Header, String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory

from bxi_example_py_elf3.inference.beyondmimic import DanceMotionPolicy, DanceMotionPolicyGravityIsaaclab
from bxi_example_py_elf3.inference.normal import NormalMotionPolicy
from bxi_example_py_elf3.inference.amp import HumanoidGaitPolicyLite
from bxi_example_py_elf3.state_machine import (
    RobotStateMachine,
    RemoteEventAdapter,
    load_state_machine_config,
)
from bxi_example_py_elf3.robot_states import build_robot_states
from bxi_example_py_elf3.utils.tfs import quaternion_to_euler_array

robot_name = "elf3"

dof_num = 29

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",

    "l_hip_y_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_z_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_y_joint",   # 右腿_髋关节_z轴
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_z_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shoulder_y_joint",   # 左臂_肩关节_y轴
    "l_shoulder_x_joint",   # 左臂_肩关节_x轴
    "l_shoulder_z_joint",   # 左臂_肩关节_z轴
    "l_elbow_y_joint",   # 左臂_肘关节_y轴
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",

    "r_shoulder_y_joint",   # 右臂_肩关节_y轴
    "r_shoulder_x_joint",   # 右臂_肩关节_x轴
    "r_shoulder_z_joint",   # 右臂_肩关节_z轴
    "r_elbow_y_joint",    # 右臂_肘关节_y轴
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
    )

joint_nominal_pos = np.array([   # 指定的固定关节角度
    0.0, 0.0, 0.0,
    -0.4,0.0,0.0,0.8,-0.4,0.0,
    -0.4,0.0,0.0,0.8,-0.4,0.0,
    0.5, 0.3,-0.1,-0.2, 0.0,0.0,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    0.5,-0.3, 0.1,-0.2, 0.0,0.0,0.0],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    dtype=np.float32)

joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
    500,500,300,
    300,100,100,300,50,50,
    300,100,100,300,50,50,
    100,80,80,100, 20,20,20,
    100,80,80,100, 20,20,20],
    dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    3,3,3,
    2.5,2,2,2.5,2,2,
    2.5,2,2,2.5,2,2,
    2.5,2,2,2.5, 1,1,1,
    2.5,2,2,2.5, 1,1,1],
    dtype=np.float32)

kp_recover = np.array([     # 跌到起身腰部手部pd加大(add pd for hands and waist)
    500,500,300,
    150, 150, 150, 200, 50, 50,
    150, 150, 150, 200, 50, 50,
    80, 80, 80, 60, 20, 50, 50,
    80, 80, 80, 60, 20, 50, 50,],
    dtype=np.float32)

kd_recover = np.array([  # 跌到起身腰部手部pd加大(add pd for hands and waist)
    5,3,3,
    2,2,2,2,1,1,
    2,2,2,2,1,1,
    2,2,2,2, 1,2,2,
    2,2,2,2, 1,2,2],
    dtype=np.float32)

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # w, x, y, z = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

class BxiExample(Node):
    def __init__(self):
        super().__init__('bxi_example_py')

        # 从launch文件中获取模型路径
        self.load_files()

        # 加载模型
        self.normal = HumanoidGaitPolicyLite(self.onnx_file_dict["normal"])
        self.recover = DanceMotionPolicy(self.npz_file_dict["recover"], self.onnx_file_dict["recover"], start_frame=600)
        self.dance = DanceMotionPolicy(self.npz_file_dict["dance"], self.onnx_file_dict["dance"])
        self.amp_run = HumanoidGaitPolicyLite(self.onnx_file_dict["amp_run"])
        self.normal_run = NormalMotionPolicy(self.onnx_file_dict["normal_run"])
        self.back_flip = DanceMotionPolicyGravityIsaaclab(
            self.npz_file_dict["back_flip"],
            self.onnx_file_dict["back_flip"],
            start_frame=40,
        )
        self.forward_flip = DanceMotionPolicyGravityIsaaclab(
            self.npz_file_dict["forward_flip"],
            self.onnx_file_dict["forward_flip"],
            start_frame=150,
        )
        self.noarm = HumanoidGaitPolicyLite(self.onnx_file_dict["noarm"])

        self.initial_pos = np.zeros(dof_num, dtype=np.double)
        self.pd_pos = self.normal.default_dof_pos

        # 订阅发布ros主题
        self.init_pub_sub()

        # 机器人状态变量(robot states)
        self.qpos = np.zeros(dof_num, dtype=np.double)
        self.qvel = np.zeros(dof_num, dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat_xyzw = np.zeros(4,dtype=np.double)
        self.quat_wxyz = np.zeros(4,dtype=np.double)

        self.pos_last = np.zeros(dof_num, dtype=np.float32)
        self.kp_last = np.zeros(dof_num, dtype=np.float32)
        self.kd_last = np.zeros(dof_num, dtype=np.float32)
        self.pos_last_state = np.zeros(dof_num, dtype=np.float32)
        self.kp_last_state = np.zeros(dof_num, dtype=np.float32)
        self.kd_last_state = np.zeros(dof_num, dtype=np.float32)

        # 状态切换参数
        self.dof_num = dof_num
        self.joint_nominal_pos = joint_nominal_pos
        self.joint_kp = joint_kp
        self.joint_kd = joint_kd
        self.loop_count = 0
        self.motor_target = None
        self.speed_profiles = self.state_machine_config.get("speed_profiles", {})
        self.pending_remote_events = deque()
        self.current_q = np.zeros(dof_num, dtype=np.double)
        self.current_dq = np.zeros(dof_num, dtype=np.double)
        self.current_omega = np.zeros(3, dtype=np.double)
        self.current_quat_xyzw = np.zeros(4, dtype=np.double)
        self.current_quat_wxyz = np.zeros(4, dtype=np.double)
        self.raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.current_raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.current_cmd_vel = np.zeros(3, dtype=np.float32)

        robot_states = build_robot_states(self.state_machine_config)
        self.robot_states = robot_states
        self.state_id_by_name = {name: state.state_id for name, state in robot_states.items()}
        self.state_name_by_id = {value: key for key, value in self.state_id_by_name.items()}
        self.bind_robot_states(robot_states)
        self.state_machine = RobotStateMachine(
            self,
            self.state_machine_config,
            robot_states,
        )
        self.remote_event_adapter = RemoteEventAdapter(self.state_machine_config.get("remote_events", {}))

        self.state = self.state_machine.current_state_id

        # 定时器初始化
        self.step = 0
        self.dt = 0.02  # loop @100Hz
        self.state_machine_info_elapsed = 0.0
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

    def load_files(self):
        # 加载模型
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value

        self.declare_parameter('/npz_file_dict', json.dumps({}))
        npz_file_json = self.get_parameter('/npz_file_dict').value
        self.npz_file_dict = json.loads(npz_file_json)

        self.declare_parameter('/onnx_file_dict', json.dumps({}))
        onnx_file_json = self.get_parameter('/onnx_file_dict').value
        self.onnx_file_dict = json.loads(onnx_file_json)

        default_state_machine_config = self.default_state_machine_config_path()
        self.declare_parameter('/state_machine_config', default_state_machine_config)
        self.state_machine_config_path = self.get_parameter('/state_machine_config').value
        self.state_machine_config = load_state_machine_config(self.state_machine_config_path)

        self.declare_parameter('/state_machine_info_topic', '')
        self.state_machine_info_topic = self.get_parameter('/state_machine_info_topic').get_parameter_value().string_value

        self.declare_parameter('/state_machine_info_hz', 10.0)
        self.state_machine_info_hz = float(self.get_parameter('/state_machine_info_hz').value)

    def default_state_machine_config_path(self):
        try:
            package_share = get_package_share_directory("bxi_example_py_elf3")
            config_path = os.path.join(package_share, "config", "elf3_state_machine.yaml")
            if os.path.exists(config_path):
                return config_path
        except Exception:
            pass

        package_root = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(package_root, "config", "elf3_state_machine.yaml")

    def bind_robot_states(self, robot_states):
        for state in robot_states.values():
            state.on_bind(self)

    def init_pub_sub(self):
        # 订阅和发布主题
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)

        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)  # CHANGE
        state_machine_info_topic = self.state_machine_info_topic or (self.topic_prefix + 'state_machine_info')
        self.state_machine_info_pub = self.create_publisher(String, state_machine_info_topic, 10)

        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', self.odom_callback, qos)
        # self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.actuator_sub = self.create_subscription(bxiMsg.ActuatorStates, self.topic_prefix+'actuator_states', self.actuator_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)

        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')

        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group_2 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()

    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        events = []
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (1./self.dt): # 延迟2s
            self.robot_reset(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return

        if self.step == 2:
            with self.lock_in:
                self.current_q = self.qpos.copy()
                self.current_dq = self.qvel.copy()
                self.current_quat_xyzw = self.quat_xyzw.copy()
                self.current_quat_wxyz = self.quat_wxyz.copy()
                self.current_omega = self.omega.copy()
                self.current_raw_cmd_vel[:] = self.raw_cmd_vel
                self.current_cmd_vel.fill(0.0)
                events = list(self.pending_remote_events)
                self.pending_remote_events.clear()

            self.motor_target = None
            transition_active = self.state_machine.update(self.dt, events)
            self.state = self.state_machine.current_state_id

            if not transition_active:
                self.state_machine.update_current_state(self.dt)
                self.state = self.state_machine.current_state_id

            if self.motor_target is not None:
                qpos, kp, kd = self.motor_target
                self.pos_last = qpos
                self.kp_last = kp
                self.kd_last = kd
                self.send_to_motor(qpos, kp, kd)

        self.loop_count += 1
        self.publish_state_machine_info_if_due(events)

    def send_to_motor(self, dof_pos_target, joint_kp, joint_kd):
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = dof_pos_target.tolist()
        msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.kp = joint_kp.tolist()
        msg.kd = joint_kd.tolist()
        self.act_pub.publish(msg)

    def publish_state_machine_info_if_due(self, events):
        if self.state_machine_info_hz <= 0.0:
            return

        self.state_machine_info_elapsed += self.dt
        period = 1.0 / self.state_machine_info_hz
        if self.state_machine_info_elapsed + 1e-9 < period:
            return

        self.state_machine_info_elapsed = 0.0
        self.publish_state_machine_info(events)

    def publish_state_machine_info(self, events):
        now = self.get_clock().now()
        current_cmd_vel = self.current_cmd_vel.tolist()
        info = self.state_machine.snapshot(include_graph=True)
        info.update(
            {
                "stamp": {
                    "sec": int(now.nanoseconds // 1000000000),
                    "nanosec": int(now.nanoseconds % 1000000000),
                },
                "step": int(self.step),
                "loop_count": int(self.loop_count),
                "events": list(events),
                "cmd_vel": {
                    "x": float(current_cmd_vel[0]),
                    "y": float(current_cmd_vel[1]),
                    "yaw": float(current_cmd_vel[2]),
                },
            }
        )

        msg = String()
        msg.data = json.dumps(info, ensure_ascii=False, sort_keys=True)
        self.state_machine_info_pub.publish(msg)

    def robot_reset(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name

        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

        self.rest_srv.call_async(req)

    def sim_robot_reset(self):
        req = bxiSrv.SimulationReset.Request()
        req.header.frame_id = robot_name

        base_pose = Pose()
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 1.0
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = 0.0
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 1.0

        joint_state = JointState()
        joint_state.name = joint_name
        joint_state.position = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.velocity = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.effort = np.zeros(dof_num, dtype=np.float32).tolist()

        req.base_pose = base_pose
        req.joint_state = joint_state

        while not self.sim_rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

        self.sim_rest_srv.call_async(req)

    def joint_callback(self, msg):
        joint_pos = msg.position
        joint_vel = msg.velocity
        joint_tor = msg.effort

        with self.lock_in:
            self.qpos[:] = np.array(joint_pos[:])
            self.qvel[:] = np.array(joint_vel[:])

    def actuator_callback(self, msg):
        joint_pos = msg.position
        joint_vel = msg.velocity
        joint_tor = msg.effort
        drv_temperature = msg.driver_temperature
        motor_temperature = msg.motor_temperature

        with self.lock_in:
            self.qpos[:] = np.array(joint_pos[:])
            self.qvel[:] = np.array(joint_vel[:])

    def set_motor_target(self, qpos, kp, kd):
        frame = (
            np.asarray(qpos, dtype=np.float32).copy(),
            np.asarray(kp, dtype=np.float32).copy(),
            np.asarray(kd, dtype=np.float32).copy(),
        )
        self.motor_target = frame

    def hold_last_motor_target(self):
        self.set_motor_target(self.pos_last, self.kp_last, self.kd_last)

    def request_state(self, state_name, trigger="code", transition="instant", delay=0.0):
        self.state_machine.request_transition(state_name, trigger=trigger, transition=transition, delay=delay)

    def is_orientation_unsafe(self, quat_xyzw):
        eu_ang = quaternion_to_euler_array(quat_xyzw)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        return (np.abs(eu_ang[0]) > (math.pi / 3.0)) or (np.abs(eu_ang[1]) > (math.pi / 3.0))

    def joy_callback(self, msg):
        with self.lock_in:
            self.raw_cmd_vel[:] = (
                msg.vel_des.x,
                msg.vel_des.y,
                msg.yawdot_des,
            )
            events = self.remote_event_adapter.extract_events(msg, sync_only=self.step < 2)
            self.pending_remote_events.extend(events)

        if self.step < 2:
            return

    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)
        quat_tmp2 = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.double)

        with self.lock_in:
            self.quat_xyzw = quat_tmp1
            self.quat_wxyz = quat_tmp2
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, msg):
        foot_force = msg.value

    def odom_callback(self, msg): # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

    def fill_model_obs_history(self, model):
        if not hasattr(model, 'obs_history') or len(model.obs_history) == 0:
            return

        latest_obs = np.array(model.obs_history[-1], dtype=np.float32).copy()
        history_len = getattr(model, 'obs_history_len', model.obs_history.maxlen)

        model.obs_history.clear()
        for _ in range(history_len):
            model.obs_history.append(latest_obs.copy())

        if hasattr(model, 'obs') and hasattr(model, 'single_obs_dim'):
            for i, hist_obs in enumerate(model.obs_history):
                start_idx = i * model.single_obs_dim
                end_idx = start_idx + model.single_obs_dim
                model.obs[start_idx:end_idx] = hist_obs

            model.obs_tensor = np.expand_dims(model.obs, axis=0)

    # --- 模型切换过渡逻辑 ---
    def preheat_model(self, model, with_cmd_vel=False, cmd_vel=None):
        # 用当前观测预推理一次，不输出到电机；有历史观测的模型随后用当前观测填满历史。
        q = self.qpos.copy()
        dq = self.qvel.copy()
        omega = self.omega.copy()
        quat_xyzw = self.quat_xyzw.copy()
        quat_wxyz = self.quat_wxyz.copy()
        if cmd_vel is None:
            cmd_vel = self.current_cmd_vel.copy()
        else:
            cmd_vel = np.asarray(cmd_vel, dtype=np.float32)

        if model is self.normal_run:
            model.infer_step(q, dq, quat_xyzw, omega, cmd_vel)
        else:
            if with_cmd_vel:
                model.inference_step(q, dq, quat_wxyz, omega, cmd_vel)
            else:
                model.inference_step(q, dq, quat_wxyz, omega)

        self.fill_model_obs_history(model)

def main(args=None):
       
    time.sleep(5)
    
    rclpy.init(args=args)
    node = BxiExample()
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
