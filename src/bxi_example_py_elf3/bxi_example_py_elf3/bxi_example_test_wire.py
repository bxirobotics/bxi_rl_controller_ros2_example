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
import json

import onnxruntime as ort

robot_name = "elf3"

dof_num = 29

dof_use = 23

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

joint_kp = np.array([
    108.448, 162.672, 176.421,
    176.421, 176.421,  54.224, 176.421,  33.493,  21.771,
    176.421, 176.421,  54.224, 176.421,  33.493,  21.771,
    54.224,  54.224,  16.747, 54.224,  16.747,  16.747,  16.747,
    54.224,  54.224,  16.747, 54.224,  16.747,  16.747,  16.747],
    dtype=np.float32)

joint_kd = np.array([
    6.904, 10.356, 11.231,
    11.231, 11.231,  3.452, 11.231,  2.132,  1.386,
    11.231, 11.231,  3.452, 11.231,  2.132,  1.386,
    3.452,  3.452,  1.066,  3.452,  1.066,  1.066,  1.066,
    3.452,  3.452,  1.066,  3.452,  1.066,  1.066,  1.066],
    dtype=np.float32)

joint_nominal_pos = np.array([   # 默认关节角度，来自 bxi_example_py_elf3_demo-2
    0.0, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
    -0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
    0.2,  0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    dtype=np.float32)

class env_cfg():
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env():
        frame_stack = 15  # 历史观测帧数
        num_single_obs = (47+(3*11))  # 单帧观测数
        num_observations = int(frame_stack * num_single_obs)  # 总观测空间 (66×47)
        num_actions = (12+11)  # 动作数
        num_commands = 5 # sin[2] vx vy vz

    class init_state():

        default_joint_angles = {
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_z_joint": 0.0,
            
            'l_hip_z_joint': -0.4,
            'l_hip_x_joint': 0.0,
            'l_hip_y_joint': 0.0,
            'l_knee_y_joint': 0.8,
            'l_ankle_y_joint': -0.4,
            'l_ankle_x_joint': 0.0,
            
            'r_hip_z_joint': -0.4,
            'r_hip_x_joint': 0.0,
            'r_hip_y_joint': 0.0,
            'r_knee_y_joint': 0.8,
            'r_ankle_y_joint': -0.4,
            'r_ankle_x_joint': 0.0,
            
            'l_shoulder_y_joint': 0.5,
            'l_shoulder_x_joint': 0.3,
            'l_shoulder_z_joint': -0.2,
            'l_elbow_y_joint': -1.5,
            
            'r_shoulder_y_joint': 0.5,
            'r_shoulder_x_joint': -0.0,
            'r_shoulder_z_joint': 0.2,
            'r_elbow_y_joint': -1.5,
        }

    class control():
        action_scale = 0.5
        
    class commands():
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not

    class rewards:
        cycle_time = 0.6

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

class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')

        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)
        
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
        
        self.vx = 0.0
        self.vy = 0
        self.dyaw = 0
    
        self.step = 0
        self.loop_count = 0
        self.dt = 0.02  # loop @100Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        self.data_txt_path = './src/bxi_example_py_elf3/data/data.txt'
        self.pos_data_lines = self.load_pos_file()
        self.pos_data_index = 0

        self.dance_flag_prev = False
        self.dance_mode = False

    def load_pos_file(self):
        try:
            with open(self.data_txt_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) == 0:
                self.get_logger().warning(f'pos file found but empty: {self.data_txt_path}')
            else:
                self.get_logger().info(f'loaded {len(lines)} pose lines from {self.data_txt_path}')
            return lines
        except FileNotFoundError:
            self.get_logger().warning(f'pos file not found: {self.data_txt_path}')
            return []
        except Exception as e:
            self.get_logger().error(f'failed to load pos file: {e}')
            return []

    def get_next_pos_from_file(self):
        if not self.pos_data_lines:
            return None

        if self.pos_data_index >= len(self.pos_data_lines):
            # return None
            self.pos_data_index = 0
            # 如果希望循环播放，则从头开始读取；如果不希望循环，则改成返回 None

        line = self.pos_data_lines[self.pos_data_index]
        self.pos_data_index += 1
        clean_line = line.strip()
        if clean_line.startswith('[') and clean_line.endswith(']'):
            clean_line = clean_line[1:-1].strip()

        if ',' in clean_line:
            tokens = [tok.strip() for tok in clean_line.split(',') if tok.strip()]
        else:
            tokens = [tok.strip() for tok in clean_line.split() if tok.strip()]

        try:
            values = [float(x) for x in tokens]
        except ValueError as e:
            self.get_logger().error(f'failed to parse pos line {self.pos_data_index}: {e} -> {repr(line)}')
            self.pos_data_lines = []
            return None

        if len(values) != dof_num:
            self.get_logger().error(f'expected {dof_num} values, got {len(values)} on line {self.pos_data_index}')
            self.pos_data_lines = []
            return None

        return np.array(values, dtype=np.float32)

    def timer_callback(self):
        
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (2./self.dt): # 延迟10s
            self.robot_reset(2, False) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(1./self.dt) # 1秒关节缓启动
            if soft_start > 1:
                soft_start = 1
                
            soft_joint_kp = joint_kp * soft_start
            soft_joint_kd = joint_kd 
                
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = joint_nominal_pos.tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = soft_joint_kp.tolist()
            msg.kd = soft_joint_kd.tolist()
            self.act_pub.publish(msg)
            
        elif self.step == 2:
            count_lowlevel = self.loop_count
            
            qpos = joint_nominal_pos.copy()

            if self.dance_mode == True:
                if self.pos_data_lines:
                    qpos_file = self.get_next_pos_from_file()
                    if qpos_file is not None:
                        qpos = qpos_file
            
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

    def joy_callback(self, msg):
        dance_flag = msg.btn_9              # X 暂停或继续跳舞

        if dance_flag != self.dance_flag_prev:
            self.dance_mode = not self.dance_mode

        if self.step < 2:
            self.dance_flag_prev = dance_flag

        self.dance_flag_prev = dance_flag
        return
        
    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)

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
    
