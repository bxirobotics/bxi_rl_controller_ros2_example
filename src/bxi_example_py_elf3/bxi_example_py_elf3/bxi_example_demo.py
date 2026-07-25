import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg
import sensor_msgs.msg
from threading import Lock
import numpy as np

import time
import os
import json
from collections import deque
from pathlib import Path
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory

from bxi_example_py_elf3._runtime.controller import (
    RobotControlFramework,
    RobotObservation,
)
from bxi_example_py_elf3._runtime.state_machine import load_state_machine_config
from bxi_example_py_elf3.mod_api import MotorFrame

robot_name = "elf3"

dof_num = 29

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    "l_hip_y_joint",  # 左腿_髋关节_z轴
    "l_hip_x_joint",  # 左腿_髋关节_x轴
    "l_hip_z_joint",  # 左腿_髋关节_y轴
    "l_knee_y_joint",  # 左腿_膝关节_y轴
    "l_ankle_y_joint",  # 左腿_踝关节_y轴
    "l_ankle_x_joint",  # 左腿_踝关节_x轴
    "r_hip_y_joint",  # 右腿_髋关节_z轴
    "r_hip_x_joint",  # 右腿_髋关节_x轴
    "r_hip_z_joint",  # 右腿_髋关节_y轴
    "r_knee_y_joint",  # 右腿_膝关节_y轴
    "r_ankle_y_joint",  # 右腿_踝关节_y轴
    "r_ankle_x_joint",  # 右腿_踝关节_x轴
    "l_shoulder_y_joint",  # 左臂_肩关节_y轴
    "l_shoulder_x_joint",  # 左臂_肩关节_x轴
    "l_shoulder_z_joint",  # 左臂_肩关节_z轴
    "l_elbow_y_joint",  # 左臂_肘关节_y轴
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",  # 右臂_肩关节_y轴
    "r_shoulder_x_joint",  # 右臂_肩关节_x轴
    "r_shoulder_z_joint",  # 右臂_肩关节_z轴
    "r_elbow_y_joint",  # 右臂_肘关节_y轴
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
)


class BxiExample(Node):
    def __init__(self):
        super().__init__("bxi_example_py")

        # 加载运行参数
        self.load_files()

        # 订阅发布ros主题
        self.init_pub_sub()

        # 机器人状态变量(robot states)
        self.qpos = np.zeros(dof_num, dtype=np.double)
        self.qvel = np.zeros(dof_num, dtype=np.double)
        self.omega = np.zeros(3, dtype=np.double)
        self.quat_xyzw = np.zeros(4, dtype=np.double)
        self.quat_wxyz = np.zeros(4, dtype=np.double)
        self.raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.pending_remote_events = deque()

        # 定时器初始化
        self.step = 0
        self.startup_loop_count = 0
        self.dt = 0.02  # loop @50Hz
        self.state_machine_info_elapsed = 0.0

        self.framework = RobotControlFramework(
            self.state_machine_config,
            built_in_mod_root=self.package_share / "mods",
            dof_num=dof_num,
            ros_node=self,
            inference_period=self.dt,
        )
        for message in self.framework.startup_messages():
            self.get_logger().info(message)

        self.timer = self.create_timer(
            self.dt, self.timer_callback, callback_group=self.timer_callback_group_1
        )

    def load_files(self):
        self.declare_parameter("/topic_prefix", "default_value")
        self.topic_prefix = (
            self.get_parameter("/topic_prefix").get_parameter_value().string_value
        )

        self.package_share = Path(
            get_package_share_directory("bxi_example_py_elf3")
        )
        self.declare_parameter(
            "/state_machine_config",
            os.path.join(
                self.package_share,
                "config",
                "elf3_state_machine.yaml",
            ),
        )
        state_machine_config_path = self.get_parameter(
            "/state_machine_config"
        ).value
        self.state_machine_config = load_state_machine_config(
            state_machine_config_path
        )

        self.declare_parameter("/state_machine_info_topic", "")
        self.state_machine_info_topic = (
            self.get_parameter("/state_machine_info_topic")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("/state_machine_info_hz", 10.0)
        self.state_machine_info_hz = float(
            self.get_parameter("/state_machine_info_hz").value
        )

    def destroy_node(self):
        framework = getattr(self, "framework", None)
        if framework is not None:
            try:
                framework.close()
            except Exception as exc:
                self.get_logger().warning(f"control framework cleanup failed: {exc}")
        return super().destroy_node()

    # ---------------------------------------------------------------------------- #
    #                                    ROS话题部分                                   #
    # ---------------------------------------------------------------------------- #
    def init_pub_sub(self):
        # 订阅和发布主题
        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )

        self.act_pub = self.create_publisher(
            bxiMsg.ActuatorCmds, self.topic_prefix + "actuators_cmds", qos
        )  # CHANGE
        state_machine_info_topic = self.state_machine_info_topic or (
            self.topic_prefix + "state_machine_info"
        )
        self.state_machine_info_pub = self.create_publisher(
            String, state_machine_info_topic, 10
        )

        self.odom_sub = self.create_subscription(
            nav_msgs.msg.Odometry, self.topic_prefix + "odom", self.odom_callback, qos
        )
        # self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.actuator_sub = self.create_subscription(
            bxiMsg.ActuatorStates,
            self.topic_prefix + "actuator_states",
            self.actuator_callback,
            qos,
        )
        self.imu_sub = self.create_subscription(
            sensor_msgs.msg.Imu, self.topic_prefix + "imu_data", self.imu_callback, qos
        )
        self.touch_sub = self.create_subscription(
            bxiMsg.TouchSensor,
            self.topic_prefix + "touch_sensor",
            self.touch_callback,
            qos,
        )
        self.joy_sub = self.create_subscription(
            bxiMsg.MotionCommands, "motion_commands", self.joy_callback, qos
        )

        self.rest_srv = self.create_client(
            bxiSrv.RobotReset, self.topic_prefix + "robot_reset"
        )
        self.sim_rest_srv = self.create_client(
            bxiSrv.SimulationReset, self.topic_prefix + "sim_reset"
        )

        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group_2 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in  # Lock()

    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        events = []
        if self.step == 0:
            self.robot_reset(1, False)  # first reset
            print("robot reset 1!")
            self.step = 1
            return
        elif self.step == 1 and self.startup_loop_count >= (1.0 / self.dt):
            self.robot_reset(2, True)  # first reset
            print("robot reset 2!")
            self.step = 2
            self.framework.reset_inference_timeout_monitor()
            return

        if self.step == 2:
            with self.lock_in:
                observation = RobotObservation(
                    q=self.qpos.copy(),
                    dq=self.qvel.copy(),
                    quat_xyzw=self.quat_xyzw.copy(),
                    quat_wxyz=self.quat_wxyz.copy(),
                    omega=self.omega.copy(),
                    raw_cmd_vel=self.raw_cmd_vel.copy(),
                )
                events = list(self.pending_remote_events)
                self.pending_remote_events.clear()

            frame = self.framework.update(observation, events, self.dt)
            if frame is not None:
                self.send_to_motor(frame)
        else:
            self.startup_loop_count += 1

        self.publish_state_machine_info_if_due(events)

    def send_to_motor(self, frame: MotorFrame):
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = frame.qpos.tolist()
        msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.kp = frame.kp.tolist()
        msg.kd = frame.kd.tolist()
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
        info = self.framework.snapshot(include_graph=True)
        info.update(
            {
                "stamp": {
                    "sec": int(now.nanoseconds // 1000000000),
                    "nanosec": int(now.nanoseconds % 1000000000),
                },
                "step": int(self.step),
                "events": list(events),
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
            print("service not available, waiting again...")

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
            print("service not available, waiting again...")

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

    def joy_callback(self, msg):
        with self.lock_in:
            self.raw_cmd_vel[:] = (
                msg.vel_des.x,
                msg.vel_des.y,
                msg.yawdot_des,
            )
            events = self.framework.extract_remote_events(
                msg, sync_only=self.step < 2
            )
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

    def odom_callback(self, msg):  # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

    # ---------------------------------- ROS话题部分 --------------------------------- #


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


if __name__ == "__main__":
    main()
