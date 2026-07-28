import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg
import sensor_msgs.msg
from threading import Event, Lock
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

from bxi_example_py_elf3._runtime.control_runtime import RobotControlRuntime
from bxi_example_py_elf3._runtime.controller import RobotObservation
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

        self._shutting_down = Event()

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

        # 控制循环初始化
        self.step = 0
        self._second_reset_at = 0.0
        self._zero_actuator_values = [0.0] * dof_num

        self.runtime = RobotControlRuntime(
            self.state_machine_config,
            built_in_mod_root=self.package_share / "mods",
            dof_num=dof_num,
            ros_node=self,
            platform=self,
            logger=self.get_logger(),
            fatal_callback=self._on_control_fatal,
        )
        self.state_machine_info_timer = None
        if self.state_machine_info_hz > 0.0:
            self.state_machine_info_timer = self.create_timer(
                1.0 / self.state_machine_info_hz,
                self.publish_state_machine_info,
                callback_group=self.status_callback_group,
            )

    def load_files(self):
        self.declare_parameter("/topic_prefix", "default_value")
        self.topic_prefix = (
            self.get_parameter("/topic_prefix").get_parameter_value().string_value
        )

        self.package_share = Path(get_package_share_directory("bxi_example_py_elf3"))
        self.declare_parameter(
            "/state_machine_config",
            os.path.join(
                self.package_share,
                "config",
                "elf3_state_machine.yaml",
            ),
        )
        state_machine_config_path = self.get_parameter("/state_machine_config").value
        self.state_machine_config = load_state_machine_config(state_machine_config_path)

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

    def _on_control_fatal(self, _message: str):
        self._shutting_down.set()
        rclpy.try_shutdown()

    def destroy_node(self):
        self._shutting_down.set()
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                self.get_logger().warning(f"control runtime cleanup failed: {exc}")
        return super().destroy_node()

    # ---------------------------------------------------------------------------- #
    #                                    ROS话题部分                                   #
    # ---------------------------------------------------------------------------- #
    def init_pub_sub(self):
        # 订阅和发布主题
        self.io_callback_group = MutuallyExclusiveCallbackGroup()
        self.status_callback_group = MutuallyExclusiveCallbackGroup()
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
            nav_msgs.msg.Odometry,
            self.topic_prefix + "odom",
            self.odom_callback,
            qos,
            callback_group=self.io_callback_group,
        )
        self.actuator_sub = self.create_subscription(
            bxiMsg.ActuatorStates,
            self.topic_prefix + "actuator_states",
            self.actuator_callback,
            qos,
            callback_group=self.io_callback_group,
        )
        self.imu_sub = self.create_subscription(
            sensor_msgs.msg.Imu,
            self.topic_prefix + "imu_data",
            self.imu_callback,
            qos,
            callback_group=self.io_callback_group,
        )
        self.touch_sub = self.create_subscription(
            bxiMsg.TouchSensor,
            self.topic_prefix + "touch_sensor",
            self.touch_callback,
            qos,
            callback_group=self.io_callback_group,
        )
        self.joy_sub = self.create_subscription(
            bxiMsg.MotionCommands,
            "motion_commands",
            self.joy_callback,
            qos,
            callback_group=self.io_callback_group,
        )

        self.rest_srv = self.create_client(
            bxiSrv.RobotReset, self.topic_prefix + "robot_reset"
        )
        self.sim_rest_srv = self.create_client(
            bxiSrv.SimulationReset, self.topic_prefix + "sim_reset"
        )

        self.lock_in = Lock()
        self.lock_ou = self.lock_in  # Lock()

    # --------------------------- Runtime 平台适配接口 --------------------------- #

    def startup_step(self, now: float) -> bool:
        """Perform the ELF3-specific two-stage reset before control starts."""
        if self.step == 0:
            if self.robot_reset(1, False):
                self.get_logger().info("robot reset step 1 requested")
                self._second_reset_at = now + 1.0
                self.step = 1
            return False
        if self.step == 1:
            if now < self._second_reset_at:
                return False
            if not self.robot_reset(2, True):
                return False
            self.get_logger().info("robot reset step 2 requested")
            self.step = 2
            if self.topic_prefix.find("simulation") != -1:
                self.runtime.request_state(
                    "com.bxi.basic_actions/pd_brake", trigger="AutoPdbreak"
                )
                self.runtime.request_state(
                    "com.bxi.basic_actions/normal", trigger="AutoRelease"
                )
            return False
        return True

    def snapshot_control_inputs(self):
        """Copy the latest ROS inputs into one coherent framework observation."""
        with self.lock_in:
            observation = RobotObservation(
                q=self.qpos.copy(),
                dq=self.qvel.copy(),
                quat_xyzw=self.quat_xyzw.copy(),
                quat_wxyz=self.quat_wxyz.copy(),
                omega=self.omega.copy(),
                raw_cmd_vel=self.raw_cmd_vel.copy(),
            )
            events = tuple(self.pending_remote_events)
            self.pending_remote_events.clear()
        return observation, events

    def publish_motor_frame(self, frame: MotorFrame):
        """Convert a framework motor frame into the ELF3 ROS command message."""
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = frame.qpos.tolist()
        msg.vel = self._zero_actuator_values
        msg.torque = self._zero_actuator_values
        msg.kp = frame.kp.tolist()
        msg.kd = frame.kd.tolist()
        self.act_pub.publish(msg)

    def publish_state_machine_info(self):
        info = self.runtime.snapshot(include_graph=True)
        if info is None:
            return
        now = self.get_clock().now()
        info.update(
            {
                "stamp": {
                    "sec": int(now.nanoseconds // 1000000000),
                    "nanosec": int(now.nanoseconds % 1000000000),
                },
                "step": int(self.step),
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

        while not self.rest_srv.wait_for_service(timeout_sec=0.2):
            if self._shutting_down.is_set() or not rclpy.ok():
                return False
            self.get_logger().info("robot reset service not available; waiting")

        self.rest_srv.call_async(req)
        return True

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

        with self.lock_in:
            self.qpos[:] = np.array(joint_pos[:])
            self.qvel[:] = np.array(joint_vel[:])

    def actuator_callback(self, msg):
        joint_pos = msg.position
        joint_vel = msg.velocity

        with self.lock_in:
            self.qpos[:] = np.array(joint_pos[:])
            self.qvel[:] = np.array(joint_vel[:])

    def joy_callback(self, msg):
        events = self.runtime.extract_remote_events(msg, sync_only=self.step < 2)
        with self.lock_in:
            self.raw_cmd_vel[:] = (
                msg.vel_des.x,
                msg.vel_des.y,
                msg.yawdot_des,
            )
            self.pending_remote_events.extend(events)

        if self.step < 2:
            return

    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)
        quat_tmp2 = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.double)

        with self.lock_in:
            self.quat_xyzw = quat_tmp1
            self.quat_wxyz = quat_tmp2
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, _msg):
        pass

    def odom_callback(self, _msg):  # 全局里程计（上帝视角，仅限仿真使用）
        pass


def main(args=None):

    time.sleep(5)

    rclpy.init(args=args)
    node = BxiExample()

    executor = MultiThreadedExecutor(num_threads=3)
    try:
        executor.add_node(node)
        node.runtime.attach_executor(executor)
        node.runtime.start()
        executor.spin()
    finally:
        try:
            node.destroy_node()
        finally:
            executor.shutdown()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
