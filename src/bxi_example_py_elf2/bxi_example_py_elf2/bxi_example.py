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
from collections import deque
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from .arm_motion_controller import ArmMotionController # 导入左手挥舞控制器
from .right_arm_handshake_controller import RightArmHandshakeController # 导入右手握手控制器
from .running_arm_controller import RunningArmController # 导入奔跑手臂控制器

import onnxruntime as ort

robot_name = "elf25"

dof_num = 25

dof_use = 12

ankle_y_offset = 0.0

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

    # "waist_z_joint",
    # "waist_x_joint",
    # "waist_y_joint",

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
    0.1,0.0,0.0,-0.3,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    0.1,0.0,0.0,-0.3,0.0],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    dtype=np.float32)

joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
    500,500,300,
    100,100,100,150,30,10,
    100,100,100,150,30,10,
    20,20,10,20,10,
    20,20,10,20,10], dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    5,5,3,
    2,2,2,2.5,1,1,
    2,2,2,2.5,1,1,
    1,1,0.8,1,0.8,
    1,1,0.8,1,0.8], dtype=np.float32)

class env_cfg():
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env():
        frame_stack = 66  # 历史观测帧数
        num_single_obs = (47+0)  # 单帧观测数
        num_observations = int(frame_stack * num_single_obs)  # 总观测空间 (66×47)
        num_actions = (12+0)  # 动作数
        num_commands = 5 # sin[2] vx vy vz

    class init_state():

        default_joint_angles = {
            # 'waist_z_joint':0.0,
            # # 'waist_x_joint':0.0,
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
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not

    class rewards:
        cycle_time = 0.7

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
        clip_observations = 100.
        clip_actions = 100.

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
    
    # return math.sin(2 * math.pi * phase)

def  _get_cos(phase):
    
    # phase %= 1.
    
    # f = 0
    # phase_1 = 0.6
    
    # width_1 = phase_1
    # width_2 = 1 - phase_1
    
    # width_sin_1 = (2*math.pi)/2.
    
    # if phase < phase_1:
    #     f = math.sin(width_sin_1 * (phase / width_1))
    # else: 
    #     f = -math.sin(width_sin_1 * ((phase - phase_1) / width_2))
    
    # return f
    
    return math.cos(2 * math.pi * phase)


class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)

        # # 策略文件在policy目录下
        # self.declare_parameter('/policy_file', 'default_value')
        # self.policy_file = self.get_parameter('/policy_file').get_parameter_value().string_value
        # print('policy_file:', self.policy_file)
        
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
        
        # # 加载策略文件，策略文件在policy目录下
        # self.policy = torch.jit.load(self.policy_file)
        # print("Load model from:", self.policy_file)

        policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
        print("policy test")
        # # 执行推理，输出角度(双腿是12个)
        # self.action[:] = self.policy(torch.tensor(policy_input))[0].detach().numpy()
        
        self.initialize_onnx(self.onnx_file)
        self.action[:] = self.inference_step(policy_input)

        self.vx = 1.0
        self.vy = 0
        self.dyaw = 0

        self.step = 0
        self.loop_count = 0
        self.dt = 0.01  # loop @100Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        # 实例化手臂运动控制器
        self.arm_motion_controller = ArmMotionController(
            logger=self.get_logger(),
            arm_freq=0.5,           # 挥舞频率，适中
            arm_amp=0.7,            # 大幅度的左右摆动
            arm_base_height_y=-2.2, # 手臂抬起高度，适中
            arm_float_amp=0.0,      # 去除上下浮动
            arm_startup_duration=2.0, # 快速启动
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

        # 添加按钮状态跟踪，用于检测状态变化
        self.prev_left_arm_btn = 0
        self.prev_right_arm_btn = 0

        # 实例化奔跑手臂控制器
        self.running_arm_controller = RunningArmController(
            logger=self.get_logger(),
            joint_nominal_pos_ref=joint_nominal_pos,
            arm_startup_duration=3.0,  # 更长的启动时间确保非常平滑的过渡
            arm_shutdown_duration=3.0, # 更长的关闭时间确保非常平滑的过渡
            arm_amplitude_y=0.15,      # 进一步减小Y轴摆幅使动作更柔和
            arm_amplitude_z=0.08,      # 极轻微Z轴摆动增加自然感
            elbow_coeff=0.1,          # 进一步降低肘部弯曲系数使动作柔和
            smoothing_factor=0.8       # 添加平滑因子，确保动作流畅
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
            self.robot_reset(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (10./self.dt): # 延迟10s
            self.robot_reset(2, True) # first reset
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
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            
            #check safe
            if (np.abs(eu_ang[0]) > (math.pi/4.0)) or (np.abs(eu_ang[1]) > (math.pi/4.0)):
                print("check safe error, exit!")
                os._exit()

            phase = count_lowlevel * self.dt  / env_cfg.rewards.cycle_time
            obs[0, 0] = np.sin(2. * np.pi * phase) #_get_sin(phase)
            obs[0, 1] = np.cos(2. * np.pi * phase) #_get_sin(phase + 0.5)
            # obs[0, 1] = _get_cos(phase)
            
            obs[0, 2] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            
            obs[0, env_cfg.env.num_commands:env_cfg.env.num_commands+env_cfg.env.num_actions] = (q - cfg.robot_config.default_dof_pos) * env_cfg.normalization.obs_scales.dof_pos
            obs[0, env_cfg.env.num_commands+env_cfg.env.num_actions:env_cfg.env.num_commands+2*env_cfg.env.num_actions] = dq * env_cfg.normalization.obs_scales.dof_vel
            obs[0, env_cfg.env.num_commands+2*env_cfg.env.num_actions:env_cfg.env.num_commands+3*env_cfg.env.num_actions] = self.action
            obs[0, env_cfg.env.num_commands+3*env_cfg.env.num_actions:env_cfg.env.num_commands+3*env_cfg.env.num_actions+3] = omega
            obs[0, env_cfg.env.num_commands+3*env_cfg.env.num_actions+3:env_cfg.env.num_commands+3*env_cfg.env.num_actions+6] = eu_ang
            
            obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)

            self.hist_obs.append(obs)
            self.hist_obs.popleft()

            policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
            for i in range(env_cfg.env.frame_stack):
                policy_input[0, i * env_cfg.env.num_single_obs : (i + 1) * env_cfg.env.num_single_obs] = self.hist_obs[i][0, :]
            
            # # 执行推理，输出角度(双腿是12个)
            # self.action[:] = self.policy(torch.tensor(policy_input))[0].detach().numpy()
            
            self.action[:] = self.inference_step(policy_input)
            # self.action = self.action*0.8 + self.last_action*0.2
            self.action = np.clip(self.action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)

            self.target_q = self.action * env_cfg.control.action_scale
            
            qpos = joint_nominal_pos.copy()
            
            # qpos[0] += self.target_q[0]
            # qpos[2] += self.target_q[1]
            # qpos[3:15] += self.target_q[2:14]
            
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

            # 新增：奔跑手臂运动控制（移至前面以便后续优先级处理）
            leg_phase_left_signal = obs[0, 0]  # np.sin(2. * np.pi * phase)
            leg_phase_right_signal = obs[0, 1] # np.cos(2. * np.pi * phase)

            if self.enable_running_arm_motion_flag:
                if not self.running_arm_controller.is_active and not self.running_arm_controller.is_shutting_down:
                    self.running_arm_controller.start_running_motion(current_sim_time)
            else:
                if self.running_arm_controller.is_active and not self.running_arm_controller.is_shutting_down:
                    self.running_arm_controller.stop_running_motion(current_sim_time)

            # 并行控制输出：允许多个控制器同时计算，通过关节索引分配避免冲突
            # 关节分配：左臂(15-19) 右臂(20-24) 奔跑控制器(15-24作为基础)
            
            # 先应用奔跑控制器作为基础（如果活跃）
            if self.running_arm_controller.is_active or self.running_arm_controller.is_shutting_down:
                qpos = self.running_arm_controller.calculate_running_arm_motion(
                    qpos,
                    current_sim_time,
                    leg_phase_left_signal,
                    leg_phase_right_signal,
                    self.loop_count
                )
            
            # 并行应用左手控制器（覆盖左臂关节15-19）
            if self.arm_motion_controller.is_waving or self.arm_motion_controller.is_shutting_down:
                temp_qpos = qpos.copy()
                temp_qpos = self.arm_motion_controller.calculate_arm_waving(temp_qpos, current_sim_time, self.loop_count)
                # 只更新左臂关节
                qpos[15:20] = temp_qpos[15:20]
            
            # 并行应用右手控制器（覆盖右臂关节20-24）
            if self.right_arm_handshake_controller.is_handshaking or self.right_arm_handshake_controller.is_shutting_down:
                temp_qpos = qpos.copy()
                temp_qpos = self.right_arm_handshake_controller.calculate_handshake_motion(temp_qpos, current_sim_time, self.loop_count)
                # 只更新右臂关节
                qpos[20:25] = temp_qpos[20:25]
            
            qpos[4+3] += ankle_y_offset
            qpos[10+3] += ankle_y_offset
            
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
            self.last_action=self.action.copy()

        self.loop_count += 1
    
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
            self.qpos = np.array(joint_pos[3:15])
            self.qvel = np.array(joint_vel[3:15])
            
            self.qpos[4] -= ankle_y_offset
            self.qpos[10] -= ankle_y_offset
            # self.qpos[0] = np.array(joint_pos[0])
            # self.qpos[1] = np.array(joint_pos[2])
            # self.qvel[0] = np.array(joint_vel[0])
            # self.qvel[1] = np.array(joint_vel[2])
            
            # self.qpos[0] = 0
            # self.qpos[1] = 0
            # self.qvel[0] = 0
            # self.qvel[1] = 0
            
            
            # self.qpos[2:14] = np.array(joint_pos[3:15])
            # self.qvel[2:14] = np.array(joint_vel[3:15])


    def joy_callback(self, msg):
        with self.lock_in:
            self.vx = msg.vel_des.x
            self.vy = msg.vel_des.y
            self.dyaw = msg.yawdot_des
            
            # 智能按键控制逻辑
            left_arm_btn = msg.btn_6   # BT6控制左手挥舞
            right_arm_btn = msg.btn_7  # BT7控制右手握手
            
            # 检测按钮状态变化（双边沿触发）
            left_btn_changed = (left_arm_btn != self.prev_left_arm_btn)
            right_btn_changed = (right_arm_btn != self.prev_right_arm_btn)
            
            # 智能状态处理
            if left_btn_changed:  # BT6状态发生变化
                if self.enable_arm_waving_flag:
                    # 当前左手在挥舞，任何按键变化都停止左手
                    self.enable_arm_waving_flag = False
                else:
                    # 当前左手未挥舞，任何按键变化都启动左手并停止右手
                    self.enable_arm_waving_flag = True
                    self.enable_right_arm_handshake_flag = False  # 互斥：停止右手
            
            if right_btn_changed:  # BT7状态发生变化
                if self.enable_right_arm_handshake_flag:
                    # 当前右手在握手，任何按键变化都停止右手
                    self.enable_right_arm_handshake_flag = False
                else:
                    # 当前右手未握手，任何按键变化都启动右手并停止左手
                    self.enable_right_arm_handshake_flag = True
                    self.enable_arm_waving_flag = False  # 互斥：停止左手
            
            # 更新按钮状态历史
            self.prev_left_arm_btn = left_arm_btn
            self.prev_right_arm_btn = right_arm_btn
            
            # 检测机器人速度，解决静止时手臂不对称问题
            vel_norm = np.sqrt(self.vx**2 + self.vy**2 + self.dyaw**2)
            
            # 默认奔跑手臂动作：当没有特殊手臂动作且非静止状态时启用
            if not self.enable_arm_waving_flag and not self.enable_right_arm_handshake_flag:
                if vel_norm > env_cfg.commands.stand_com_threshold:
                    # 运动状态：启用奔跑手臂动作
                    self.enable_running_arm_motion_flag = True
                else:
                    # 静止状态：禁用奔跑手臂动作，确保手臂左右对称
                    self.enable_running_arm_motion_flag = False
            else:
                self.enable_running_arm_motion_flag = False    
        
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
    
