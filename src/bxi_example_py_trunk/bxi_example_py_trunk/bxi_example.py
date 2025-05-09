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
import math
from collections import deque
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from .arm_motion_controller import ArmMotionController # 导入新的控制器类
from .right_arm_handshake_controller import RightArmHandshakeController # 导入右手握手控制器
from .running_arm_controller import RunningArmController # 导入奔跑手臂控制器

import onnxruntime as ort

robot_name = "elf25"

dof_num = 25

dof_use = 12

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    
    "l_hip_z_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_y_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_z_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_y_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shld_y_joint",   # 左臂_肩关节_y轴
    "l_shld_x_joint",   # 左臂_肩关节_x轴
    "l_shld_z_joint",   # 左臂_肩关节_z轴
    "l_elb_y_joint",   # 左臂_肘关节_y轴
    "l_elb_z_joint",   # 左臂_肘关节_y轴
    
    "r_shld_y_joint",   # 右臂_肩关节_y轴   
    "r_shld_x_joint",   # 右臂_肩关节_x轴
    "r_shld_z_joint",   # 右臂_肩关节_z轴
    "r_elb_y_joint",    # 右臂_肘关节_y轴
    "r_elb_z_joint",    # 右臂_肘关节_y轴
    )   

joint_nominal_pos = np.array([   # 指定的固定关节角度
    0.0, 0.0, 0.0,
    0,0.0,-0.3,0.6,-0.3,0.0,
    0,0.0,-0.3,0.6,-0.3,0.0,
    0.7,0.2,-0.1,-1.5,0.0,
    0.7,-0.2,0.1,-1.5,0.0], dtype=np.float32)

joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
    1000,1000,300,
    100,100,100,100,20,10,
    100,100,100,100,20,10,
    30,30,30,30,30,
    30,30,30,30,30], dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    10,10,3,
    3,3,3,3,1,1,
    3,3,3,3,1,1,
    1,1,0.8,1,0.8,
    1,1,0.8,1,0.8], dtype=np.float32)

class env_cfg():
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env():
        frame_stack = 15  # 历史观测帧数
        num_single_obs = (47+0)  # 单帧观测数
        num_observations = int(frame_stack * num_single_obs)  # 观测数
        num_actions = (12+0)  # 动作数
        num_commands = 5 # sin[2] vx vy vz

    class init_state():

        default_joint_angles = {
            # 'waist_z_joint':0.0,
            # 'waist_x_joint':0.0,
            # 'waist_y_joint':0.0,
            
            'l_hip_z_joint': 0.0,
            'l_hip_x_joint': 0.0,
            'l_hip_y_joint': -0.3,
            'l_knee_y_joint': 0.6,
            'l_ankle_y_joint': -0.3,
            'l_ankle_x_joint': 0.0,
            
            'r_hip_z_joint': 0.0,
            'r_hip_x_joint': 0.0,
            'r_hip_y_joint': -0.3,
            'r_knee_y_joint': 0.6,
            'r_ankle_y_joint': -0.3,
            'r_ankle_x_joint': 0.0,
        }

    class control():
        action_scale = 0.5
        
    class commands():
        stand_com_threshold = -1.0 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not

    class rewards:
        cycle_time = 0.7

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.1
            quat = 1.
        clip_observations = 100.
        clip_actions = 10.

class cfg():

    class robot_config:
        default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))   

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
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

def  _get_sin(phase):
    
    phase %= 1.
    
    f = 0
    phase_1 = 0.6
    
    width_1 = phase_1
    width_2 = 1 - phase_1
    
    width_sin_1 = (2*math.pi)/2.
    
    if phase < phase_1:
        f = math.sin(width_sin_1 * (phase / width_1))
    else: 
        f = -math.sin(width_sin_1 * ((phase - phase_1) / width_2))
    
    return f

from scipy.spatial.transform import Rotation as R
def compute_gravity_projection(quat):
    """
    通过姿态四元数将全局重力方向投影到本地坐标系。
    :param attitude_quat: 姿态四元数 (w, x, y, z)
    :param global_gravity: 全局重力方向 (默认向量 [0, 0, -1])
    :return: 重力在本地坐标系的投影 (x, y, z)
    """
    gravity=np.array([0.0, 0.0, -1.0]).astype(np.float32)
    
    # 将姿态四元数转换为旋转矩阵
    rotation = R.from_quat(quat)  # (x, y, z, w)
    # rot_matrix = rotation.as_matrix()
    
    # 将全局重力方向转换到本地坐标系
    # local_gravity = np.dot(rot_matrix.T, gravity)
    
    local_gravity = rotation.apply(gravity, inverse=True).astype(np.float32)
    
    # 返回归一化的本地重力方向
    return local_gravity # / np.linalg.norm(local_gravity)

class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)
        
        self.declare_parameter('/onnx_file', 'default_value')
        self.onnx_file = self.get_parameter('/onnx_file').get_parameter_value().string_value        
        print("onnx_file:", self.onnx_file)

        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)
        
        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)  # CHANGE
        
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', self.odom_callback, qos)
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)

        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')
        
        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()
        self.qpos = np.zeros(env_cfg.env.num_actions,dtype=np.double)
        self.qvel = np.zeros(env_cfg.env.num_actions,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)
        
        self.hist_obs = deque()
        for _ in range(env_cfg.env.frame_stack):
            self.hist_obs.append(np.zeros([1, env_cfg.env.num_single_obs], dtype=np.double))
        self.target_q = np.zeros((env_cfg.env.num_actions), dtype=np.double)
        self.action = np.zeros((env_cfg.env.num_actions), dtype=np.double)

        self.last_action = np.zeros((env_cfg.env.num_actions), dtype=np.double)
        
        policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
        print("policy test")
        
        self.initialize_onnx(self.onnx_file)
        self.action[:] = self.inference_step(policy_input)

        self.vx = 0.1
        self.vy = 0
        self.dyaw = 0

        self.step = 0
        self.loop_count = 0
        self.dt = 0.01  # loop @100Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        # 实例化手臂运动控制器
        self.arm_motion_controller = ArmMotionController(
            logger=self.get_logger(),
            arm_freq=0.5,
            arm_amp=0.6,
            arm_base_height_y=-1.6, # 根据需要调整，或者参考旧版的值
            arm_float_amp=0.3,
            arm_startup_duration=3.0,
            joint_nominal_pos_ref=joint_nominal_pos 
        )
        self.enable_arm_waving_flag = False 

        # 实例化右手握手控制器
        self.right_arm_handshake_controller = RightArmHandshakeController(
            logger=self.get_logger(),
            handshake_startup_duration=1.5, 
            joint_nominal_pos_ref=joint_nominal_pos
        )
        self.enable_right_arm_handshake_flag = False

        # 实例化奔跑手臂控制器
        self.running_arm_controller = RunningArmController(
            logger=self.get_logger(),
            joint_nominal_pos_ref=joint_nominal_pos,
            arm_amplitude_y=0.2, # 降低Y轴摆幅从0.4到0.2
            arm_amplitude_z=0.0, # Z轴摆幅保持0
            elbow_coeff=0.3     # 降低肘部弯曲系数从0.5到0.3
        )
        self.enable_running_arm_motion_flag = False

    # 初始化部分（完整版）
    def initialize_onnx(self, model_path):
        # 配置执行提供者（根据硬件选择最优后端）
        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=options
        )
        
        # 预存输入输出信息
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        # 预分配输入内存（可选，适合固定输入尺寸）
        self.input_buffer = np.zeros(
            self.input_info.shape,
            dtype=np.float32
        )

    # 循环推理部分（极速版）
    def inference_step(self, obs_data):
        # 使用预分配内存（如果适用）
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        
        # 极简推理（比原版快5-15%）
        return self.session.run(
            [self.output_info.name], 
            {self.input_info.name: self.input_buffer}
        )[0][0]  # 直接获取第一个输出的第一个样本
 
    def timer_callback(self):
        
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_rest(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (10./self.dt): # 延迟10s
            self.robot_rest(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(1./self.dt) # 1秒关节缓启动
            if soft_start > 1:
                soft_start = 1
                
            soft_joint_kp = joint_kp * soft_start
                
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = joint_nominal_pos.tolist()
            # msg.pos = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = soft_joint_kp.tolist()
            msg.kd = joint_kd.tolist()
            self.act_pub.publish(msg)
            
        elif self.step == 2:
            with self.lock_in:
                q = self.qpos
                dq = self.qvel
                quat = self.quat
                omega = self.omega
                
                x_vel_cmd = self.vx
                y_vel_cmd = self.vy
                yaw_vel_cmd = self.dyaw
            
            count_lowlevel = self.loop_count
            
            if hasattr(env_cfg.commands,"sw_switch"):
                vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
                if env_cfg.commands.sw_switch and vel_norm <= env_cfg.commands.stand_com_threshold:
                    count_lowlevel = 0
                    
            obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
            
            projected_gravity = compute_gravity_projection(quat)

            phase = count_lowlevel * self.dt  / env_cfg.rewards.cycle_time
            
            obs[0, 0:3] = projected_gravity
            obs[0, 3:6] = omega
            obs[0, 6:6+12] = (q - cfg.robot_config.default_dof_pos) * env_cfg.normalization.obs_scales.dof_pos
            obs[0, 6+12:6+12+12] = dq * env_cfg.normalization.obs_scales.dof_vel
            obs[0, 6+12+12:6+12+12+12] = self.action

            obs[0, 42] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
            obs[0, 43] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
            obs[0, 44] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            obs[0, 45] = _get_sin(phase)
            obs[0, 46] = _get_sin(phase + 0.5)
            
            obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)

            self.hist_obs.append(obs)
            self.hist_obs.popleft()

            policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
            for i in range(env_cfg.env.frame_stack):
                policy_input[0, i * env_cfg.env.num_single_obs : (i + 1) * env_cfg.env.num_single_obs] = self.hist_obs[i][0, :]
            
            self.action[:] = self.inference_step(policy_input)
            
            self.action = np.clip(self.action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)

            self.target_q = self.action * env_cfg.control.action_scale
            
            qpos = joint_nominal_pos.copy()
            
            qpos[3:15] += self.target_q

            # 新增：手臂控制逻辑
            current_sim_time = self.loop_count * self.dt

            # 左手挥舞控制
            if self.enable_arm_waving_flag:
                if not self.arm_motion_controller.is_waving and not self.arm_motion_controller.is_shutting_down:
                    self.arm_motion_controller.start_waving(current_sim_time)
            else:
                if self.arm_motion_controller.is_waving and not self.arm_motion_controller.is_shutting_down:
                    self.arm_motion_controller.stop_waving(current_sim_time)

            # 右手握手控制
            if self.enable_right_arm_handshake_flag:
                if not self.right_arm_handshake_controller.is_handshaking and not self.right_arm_handshake_controller.is_shutting_down:
                    self.right_arm_handshake_controller.start_handshake(current_sim_time)
            else:
                if self.right_arm_handshake_controller.is_handshaking and not self.right_arm_handshake_controller.is_shutting_down:
                    self.right_arm_handshake_controller.stop_handshake(current_sim_time)

            # 如果左手控制器处于挥舞或关闭状态，则计算左手臂动作
            if self.arm_motion_controller.is_waving or self.arm_motion_controller.is_shutting_down:
                qpos = self.arm_motion_controller.calculate_arm_waving(qpos, current_sim_time, self.loop_count)

            # 如果右手控制器处于握手或关闭状态，则计算右手臂动作
            if self.right_arm_handshake_controller.is_handshaking or self.right_arm_handshake_controller.is_shutting_down:
                qpos = self.right_arm_handshake_controller.calculate_handshake_motion(qpos, current_sim_time, self.loop_count)

            # 新增：奔跑手臂运动控制
            leg_phase_left_signal = obs[0, 45]  # _get_sin(phase)
            leg_phase_right_signal = obs[0, 46] # _get_sin(phase + 0.5)

            if self.enable_running_arm_motion_flag:
                if not self.running_arm_controller.is_active and not self.running_arm_controller.is_shutting_down:
                    self.running_arm_controller.start_running_motion(current_sim_time)
            else:
                if self.running_arm_controller.is_active and not self.running_arm_controller.is_shutting_down:
                    self.running_arm_controller.stop_running_motion(current_sim_time)
            
            if self.running_arm_controller.is_active_or_shutting_down:
                qpos = self.running_arm_controller.calculate_running_arm_motion(
                    qpos,
                    current_sim_time,
                    leg_phase_left_signal,
                    leg_phase_right_signal,
                    self.loop_count
                )
            
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = qpos.tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = joint_kp.tolist()
            msg.kd = joint_kd.tolist()
            self.act_pub.publish(msg)

        self.loop_count += 1
    
    def robot_rest(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name
    
        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.rest_srv.call_async(req)
        
    def sim_robot_rest(self):        
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
            self.qpos = np.array(joint_pos[3:15])
            self.qvel = np.array(joint_vel[3:15])
            
    def joy_callback(self, msg):
        with self.lock_in:
            self.vx = msg.vel_des.x
            self.vy = msg.vel_des.y
            self.dyaw = msg.yawdot_des
            # 根据接收到的 mode 控制手臂挥舞或握手
            if msg.mode == 1: 
                self.enable_arm_waving_flag = True
                self.enable_right_arm_handshake_flag = False #确保不冲突
                self.enable_running_arm_motion_flag = False #确保不冲突
            elif msg.mode == 3:
                self.enable_right_arm_handshake_flag = True
                self.enable_arm_waving_flag = False #确保不冲突
                self.enable_running_arm_motion_flag = False #确保不冲突
            elif msg.mode == 5: # Changed: Mode for running arm motion now 5
                self.get_logger().info("Mode 5 activated: Enabling running arm motion.")
                self.enable_running_arm_motion_flag = True
                self.enable_arm_waving_flag = False #确保不冲突
                self.enable_right_arm_handshake_flag = False #确保不冲突
            else: # Default: disable all special arm motions
                self.enable_arm_waving_flag = False
                self.enable_right_arm_handshake_flag = False
                self.enable_running_arm_motion_flag = False
                # Log if mode is unexpected, but still disable flags
                # Mode 6 is no longer a special mode for running arms.
                if msg.mode not in [0, 1, 2, 3, 4, 5]: # Adjusted known modes
                     self.get_logger().warn(f"Unexpected mode {msg.mode} received, disabling all special arm motions.")
        
    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)

        with self.lock_in:
            self.quat = quat_tmp1
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, msg):
        foot_force = msg.value
        
    def odom_callback(self, msg): # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

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
