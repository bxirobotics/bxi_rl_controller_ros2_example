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
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from bxi_example_py_elf3.inference.beyondmimic import DanceMotionPolicy
from bxi_example_py_elf3.inference.host import TumbleRecoverPolicy
from bxi_example_py_elf3.inference.normal import NormalMotionPolicy

robot_name = "elf3"

dof_num = 29

real = 1                        # 1 for real; 0 for sim

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
    150,100,100,200,50,20,
    150,100,100,200,50,20,
    80,80,80,60, 20,20,20,
    80,80,80,60, 20,20,20], 
    dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    3,3,3,
    2,2,2,2.5,1,1,
    2,2,2,2.5,1,1,
    2,2,2,2, 1,1,1,
    2,2,2,2, 1,1,1], 
    dtype=np.float32)

kp_host = np.array([     # 跌到起身腰部手部pd加大(add pd for hands and waist)
    500,500,300, 
    150, 150, 150, 200, 50, 50, 
    150, 150, 150, 200, 50, 50,
    80, 80, 80, 60, 20, 50, 50,
    80, 80, 80, 60, 20, 50, 50,], 
    dtype=np.float32)

kd_host = np.array([  # 跌到起身腰部手部pd加大(add pd for hands and waist)
    5,3,3,
    2,2,2,2,1,1,
    2,2,2,2,1,1,
    2,2,2,2, 1,2,2,
    2,2,2,2, 1,2,2], 
    dtype=np.float32)

class robotState:
    normal      = 0     # 站、走、跑(stand walk run)
    zero_torque = 1     # 零力模式(zero torque mode)
    pd_brake    = 2     # pd模式(pd mode)
    initial_pos = 3     # 初始模式(zero position mode)

    dance       = 4
    host        = 5

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
        self.normal = NormalMotionPolicy(self.onnx_file_dict["normal"]) 
        self.host = TumbleRecoverPolicy(self.onnx_file_dict["host"])
        self.dance = DanceMotionPolicy(self.npz_file_dict["dance"], self.onnx_file_dict["dance"])   

        self.initial_pos = np.zeros(dof_num, dtype=np.double)
        self.pd_pos = self.normal.default_joint_pos

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
        self.state = robotState.zero_torque
        self.next_state = self.state 
        self.last_state = self.state
        self.change_state = 1

        if(self.topic_prefix == "hardware/"):
            real = 1
            print("real == 1")
        else:
            real = 0
            print("real == 0")

        if(not real):
            # sim 
            self.change_time = 0.1
        else:
            # real
            self.change_time = 0.3

        # 遥控器参数
        self.normal_mode_prev = False
        self.zero_torque_prev = False
        self.pd_brake_prev = False
        self.initial_pos_prev = False

        self.dance_mode_prev = False
        self.host_mode_prev = False

        self.dance_flag_prev = False

        self.dance_mode_changed = True  # False: 暂停跳舞(stop dancing)  True： 继续跳舞(continue dancing)

        # 运动命令变量
        self.vx = 0.0
        self.vy = 0
        self.dyaw = 0

        # 定时器初始化
        self.step = 0
        self.loop_count = 0
        self.dt = 0.02  # loop @100Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        # 特殊动作初始帧设置
        self.start_frame_dance = 100
        self.dance.timestep = self.start_frame_dance
        self.start_frame_pos = self.dance.motioninputpos[self.start_frame_dance,:] # 跳过前150帧准备动作

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

    def init_pub_sub(self):
        # 订阅和发布主题
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
        self.timer_callback_group_2 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()

    def enter_state(self):
        match self.next_state:
            case robotState.normal:
                self.loop_count = 0
                self.normal.action = np.zeros(dof_num, dtype=np.float32)
                return
            
            case robotState.pd_brake:
                self.loop_count = 0
                return
            
            case robotState.initial_pos:
                self.loop_count = 0
                return
            
            case robotState.dance:
                self.loop_count = 0
                self.dance_mode_changed = True
                self.dance.timestep = self.start_frame_dance
                return
            
            case robotState.host:
                self.loop_count = 0
                print("loop:", self.loop_count)
                return
        return
    
    def exit_state(self):
        self.pos_last_state = self.qpos
        self.kp_last_state = self.kp_last
        self.kd_last_state = self.kd_last
        return
    
    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
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
                q = self.qpos
                dq = self.qvel
                quat = self.quat_xyzw
                quat_wxyz = self.quat_wxyz
                omega = self.omega
                
                x_vel_cmd = self.vx
                y_vel_cmd = self.vy
                yaw_vel_cmd = self.dyaw
        
            if(self.next_state != self.state):
                self.exit_state()
                if(self.change_state == 1):
                    self.enter_state()
                    self.last_state = self.state
                    self.state = self.next_state

            if self.state == robotState.normal:
                if(self.loop_count * self.dt < self.change_time):
                    soft_start = self.loop_count/(self.change_time/self.dt)
                    if soft_start > 1:
                        soft_start = 1
                        
                    qpos = self.pos_last_state + (self.normal.default_joint_pos - self.pos_last_state) * soft_start

                    if(not real):
                        # sim
                        kp = self.kp_last_state + (self.normal.joint_stiffness * 0.9 - self.kp_last_state) * soft_start
                        kd = self.kd_last_state + (self.normal.joint_damping * 0.2 - self.kd_last_state) * soft_start
                    else:
                        # real
                        kp = self.kp_last_state + (self.normal.joint_stiffness - self.kp_last_state) * soft_start
                        kd = self.kd_last_state + (self.normal.joint_damping - self.kd_last_state) * soft_start
                else:
                    eu_ang = quaternion_to_euler_array(quat)
                    eu_ang[eu_ang > math.pi] -= 2 * math.pi

                    #check safe
                    if (np.abs(eu_ang[0]) > (math.pi/6.0)) or (np.abs(eu_ang[1]) > (math.pi/6.0)):
                        print("check safe error, zero_torque!")

                        self.next_state = robotState.zero_torque
                        return
                    
                    cmd = [x_vel_cmd, y_vel_cmd, yaw_vel_cmd]
                    qpos = self.normal.infer_step(q, dq, quat, omega, cmd)

                    if(not real):
                        # sim
                        kp = self.normal.joint_stiffness * 0.9
                        kd = self.normal.joint_damping * 0.2
                    else:
                        # real
                        kp = self.normal.joint_stiffness
                        kd = self.normal.joint_damping

            elif self.state == robotState.zero_torque:      # kp,kd 均给0
                qpos = joint_nominal_pos
                kp = np.zeros(dof_num, dtype=np.float32)
                kd = np.zeros(dof_num, dtype=np.float32)

            elif self.state == robotState.pd_brake:
                soft_start = self.loop_count/(2./self.dt) # 1秒关节缓启动
                if soft_start > 1:
                    soft_start = 1
                    
                qpos = self.pos_last_state + (self.pd_pos - self.pos_last_state) * soft_start

                if(not real):
                    # sim
                    kp = self.kp_last_state + (self.normal.joint_stiffness * 0.9  - self.kp_last_state) * soft_start
                    kd = self.kd_last_state + (self.normal.joint_damping * 0.2 - self.kd_last_state) * soft_start
                else:
                    # real
                    kp = self.kp_last + (self.normal.joint_stiffness  - self.kp_last) * soft_start
                    kd = self.kd_last + (self.normal.joint_damping - self.kd_last) * soft_start

            elif self.state == robotState.initial_pos:
                soft_start = self.loop_count/(2./self.dt)
                if soft_start > 1:
                    soft_start = 1
                    
                qpos = self.pos_last_state + (self.initial_pos- self.pos_last_state) * soft_start

                if(not real):
                    # sim
                    kp = self.kp_last_state + (self.normal.joint_stiffness * 0.9  - self.kp_last_state) * soft_start
                    kd = self.kd_last_state + (self.normal.joint_damping * 0.2 - self.kd_last_state) * soft_start
                else:
                    # real
                    kp = self.kp_last + (self.normal.joint_stiffness  - self.kp_last) * soft_start
                    kd = self.kd_last + (self.normal.joint_damping - self.kd_last) * soft_start
            
            elif self.state == robotState.dance:
                if(self.loop_count * self.dt < self.change_time):
                    soft_start = self.loop_count/(self.change_time/self.dt)
                    if soft_start > 1:
                        soft_start = 1
                        
                    qpos = self.pos_last_state + (self.normal.default_joint_pos - self.pos_last_state) * soft_start

                    if(not real):
                        # sim
                        kp = self.kp_last_state + (self.normal.joint_stiffness * 0.9  - self.kp_last_state) * soft_start
                        kd = self.kd_last_state + (self.normal.joint_damping * 0.2 - self.kd_last_state) * soft_start
                    else:
                        # real
                        kp = self.kp_last + (self.normal.joint_stiffness  - self.kp_last) * soft_start
                        kd = self.kd_last + (self.normal.joint_damping - self.kd_last) * soft_start
                else:
                    if self.dance.timestep < self.dance.motionpos.shape[0]:
                        eu_ang = quaternion_to_euler_array(quat)
                        eu_ang[eu_ang > math.pi] -= 2 * math.pi

                        #check safe
                        if (np.abs(eu_ang[0]) > (math.pi/3.0)) or (np.abs(eu_ang[1]) > (math.pi/3.0)):
                            print("check safe error, zero_torque!")

                            self.next_state = robotState.zero_torque
                            return
                        
                        qpos = self.dance.inference_step(q, dq, quat_wxyz, omega)

                        if(not real):
                            # sim
                            kp = self.dance.stiffness_array
                            kd = self.dance.damping_array * 0.2
                        else:
                            # real
                            kp = self.dance.stiffness_array
                            kd = self.dance.damping_array

                    if self.dance_mode_changed == True:
                        self.dance.timestep += 1
                    
                    # 动作结束检测    
                    if self.dance.timestep >= self.dance.motionpos.shape[0]:
                        print("Motion replay finished, resetting simulation.")
                        self.dance.timestep = self.start_frame_dance
                        self.next_state = robotState.normal

            elif self.state == robotState.host:
                if((self.loop_count * self.dt > 5.0)):
                    eu_ang = quaternion_to_euler_array(quat)
                    eu_ang[eu_ang > math.pi] -= 2 * math.pi
                    
                    if((np.abs(eu_ang[0]) > (math.pi/3.0)) or (np.abs(eu_ang[1]) > (math.pi/3.0))):
                        print("check safe error, zero_torque!")

                        self.next_state = robotState.zero_torque
                        return
                    
                qpos = self.host.inference_step(q, dq, quat_wxyz, omega)
                kp = kp_host
                kd = kd_host

            self.pos_last = qpos
            self.kp_last = kp
            self.kd_last = kd
            self.send_to_motor(qpos, kp, kd)

        self.loop_count += 1

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

    def joy_callback(self, msg):
        with self.lock_in:
            self.vx = msg.vel_des.x * 2                 # * 3
            self.vx = np.clip(self.vx, -1.5, 2.0)       # -2.0  3.0
            self.vy = msg.vel_des.y * 2
            self.dyaw = msg.yawdot_des * 2

            normal_mode = msg.btn_1             # RB+X 切换为普通模式
            zero_torque_mode = msg.btn_2        # RB+A 切换为零力模式
            pd_brake_mode = msg.btn_3           # RB+B 切换为pd模式
            initial_pos_mode = msg.btn_4        # RB+Y 切换为初始状态

            dance_mode = msg.btn_5              # LB+X 切换为跳舞模式
            host_mode = msg.btn_6               # RB+A 切换为host模式
            # = msg.btn_7
            # = msg.btn_8

            dance_flag = msg.btn_9              # X 暂停或继续跳舞
            # = msg.btn_10
            # = msg.btn_11
            # = msg.btn_12

        if self.step < 2:
            self.normal_mode_prev = normal_mode
            self.zero_torque_prev = zero_torque_mode
            self.pd_brake_prev = pd_brake_mode
            self.initial_pos_prev = initial_pos_mode

            self.dance_mode_prev = dance_mode
            self.host_mode_prev = host_mode
            
            self.dance_flag_prev = dance_flag

        #按键状态变化检测  
        match self.state:
            case robotState.normal:
                if zero_torque_mode != self.zero_torque_prev:
                    self.next_state = robotState.zero_torque
                    print("switch to zero_torque")

                elif pd_brake_mode != self.pd_brake_prev:
                    self.next_state = robotState.pd_brake
                    print("switch to pd_mode")

                elif initial_pos_mode != self.initial_pos_prev:
                    self.next_state = robotState.initial_pos
                    print("switch to initial")

                elif dance_mode != self.dance_mode_prev:
                    self.next_state = robotState.dance
                    print("switch to dance")

                elif host_mode != self.host_mode_prev:
                    self.next_state = robotState.host
                    print("switch to host")
            
            case robotState.zero_torque:
                if normal_mode != self.normal_mode_prev:
                    self.next_state = robotState.normal
                    print("switch to normal")

                elif pd_brake_mode != self.pd_brake_prev:
                    self.next_state = robotState.pd_brake
                    print("switch to pd_mode")

                elif initial_pos_mode != self.initial_pos_prev:
                    self.next_state = robotState.initial_pos
                    print("switch to initial")

                elif host_mode != self.host_mode_prev:
                    self.next_state = robotState.host
                    print("switch to host")
            
            case robotState.pd_brake:
                if normal_mode != self.normal_mode_prev:
                    self.next_state = robotState.normal
                    print("switch to normal")

                elif zero_torque_mode != self.zero_torque_prev:
                    self.next_state = robotState.zero_torque
                    print("switch to zero_torque")

                elif initial_pos_mode != self.initial_pos_prev:
                    self.next_state = robotState.initial_pos
                    print("switch to initial")
                
                elif host_mode != self.host_mode_prev:
                    self.next_state = robotState.host
                    print("switch to host")

            case robotState.initial_pos:
                if normal_mode != self.normal_mode_prev:
                    self.next_state = robotState.normal
                    print("switch to normal")

                elif pd_brake_mode != self.pd_brake_prev:
                    self.next_state = robotState.pd_brake
                    print("switch to pd_mode")

                elif zero_torque_mode != self.zero_torque_prev:
                    self.next_state = robotState.zero_torque
                    print("switch to zero_torque")
                
                elif host_mode != self.host_mode_prev:
                    self.next_state = robotState.host
                    print("switch to host")

            case robotState.dance:
                if normal_mode != self.normal_mode_prev:
                    self.next_state = robotState.normal
                    print("switch to normal")

                if dance_flag != self.dance_flag_prev:
                    self.dance_mode_changed = not self.dance_mode_changed
            
            case robotState.host:
                if normal_mode != self.normal_mode_prev:
                    self.next_state = robotState.normal
                    print("switch to normal")

                elif zero_torque_mode != self.zero_torque_prev:
                    self.next_state = robotState.zero_torque
                    print("switch to zero_torque")

                elif pd_brake_mode != self.pd_brake_prev:
                    self.next_state = robotState.pd_brake
                    print("switch to pd_mode")

                elif initial_pos_mode != self.initial_pos_prev:
                    self.next_state = robotState.initial_pos
                    print("switch to initial")
            
        self.normal_mode_prev = normal_mode
        self.zero_torque_prev = zero_torque_mode
        self.pd_brake_prev = pd_brake_mode
        self.initial_pos_prev = initial_pos_mode

        self.dance_mode_prev = dance_mode
        self.host_mode_prev = host_mode
        
        self.dance_flag_prev = dance_flag

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
    
