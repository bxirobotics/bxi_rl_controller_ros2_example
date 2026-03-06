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
# from geometry_msgs.msg import Pose
# from sensor_msgs.msg import JointState
# import signal

class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')
    
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)
        self.joy_pub = self.create_publisher(bxiMsg.MotionCommands, 'motion_commands', qos)

        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()
        
        self.dir = -1
        self.vel_forward= 0.2
        self.vel_back= -0.2

        self.loop_count = 0
        self.dt = 0.01  # loop @100Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)
        
        # signal.signal(signal.SIGINT, self.usr_exit)

    def timer_callback(self):
        
        msg = bxiMsg.MotionCommands()
        
        msg.vel_des.y = 0.
        msg.yawdot_des = 0.
        
        if self.loop_count % 200 == 0:
            if self.dir == 1:
                msg.vel_des.x = self.vel_forward
            else:
                msg.vel_des.x = self.vel_back
                
            self.dir *= -1
            
            self.joy_pub.publish(msg)
            
            print("vel: ", msg.vel_des)
        
        
        self.loop_count += 1
    
    def usr_exit(self):
        msg = bxiMsg.MotionCommands()
        
        msg.vel_des.x = 0.
        msg.vel_des.y = 0.
        msg.yawdot_des = 0.
        
        self.joy_pub.publish(msg)
        
        print("usr exit")
        
            
def main(args=None):
   
    time.sleep(5)
    
    rclpy.init(args=args)
    node = BxiExample()
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        print("finally exit------------------")
        node.usr_exit()
        executor.shutdown()
        node.destroy_node()
        
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
    
