import math
import numpy as np

class RightArmHandshakeController:
    def __init__(self, logger, handshake_startup_duration=1.5, joint_nominal_pos_ref=None):
        self.logger = logger
        self.handshake_startup_duration = handshake_startup_duration
        self.handshake_shutdown_duration = handshake_startup_duration # Can be set independently

        self.handshake_start_time = None
        self.handshake_stop_time = None
        self.is_handshaking = False # Internal state, True if handshake is active (startup complete to shutdown start)
        self.is_starting_up = False # True if in startup transition
        self.is_shutting_down = False # True if in shutdown transition

        # Joint indices from bxi_example.py for the right arm
        self.R_SHLD_Y_IDX = 20  # r_shld_y_joint
        self.R_SHLD_X_IDX = 21  # r_shld_x_joint
        self.R_SHLD_Z_IDX = 22  # r_shld_z_joint
        self.R_ELB_Y_IDX = 23   # r_elb_y_joint
        # self.R_ELB_Z_IDX = 24 # r_elb_z_joint - not actively controlled for simple handshake

        # 握手姿势的目标关节角度设置
        self.target_handshake_pose = {
            "r_shld_y": -1.0,  # 抬起肩膀（Y轴旋转）
            "r_shld_x": -0.2,  # 向前伸展肩膀（X轴旋转，比之前的-0.8更前）
            "r_shld_z": 0.2,   # 调整肩膀内旋，使肘部向内（之前为0.1）
            "r_elb_y": -1.0    # 弯曲肘部
        }
        
        # 保存当前实际关节角度，用于平滑过渡
        self.current_joint_positions = np.zeros(4) # [r_shld_y, r_shld_x, r_shld_z, r_elb_y]
        self.shutdown_start_positions = np.zeros(4) # 开始关闭时的关节位置，用于平滑过渡

        if joint_nominal_pos_ref is None:
            # Store nominal positions for relevant joints for smooth return if needed
            self.joint_nominal_r_shld_y = 0.7 # Default from bxi_example.py
            self.joint_nominal_r_shld_x = -0.2
            self.joint_nominal_r_shld_z = 0.1
            self.joint_nominal_r_elb_y = -1.5
        else:
            self.joint_nominal_r_shld_y = joint_nominal_pos_ref[self.R_SHLD_Y_IDX]
            self.joint_nominal_r_shld_x = joint_nominal_pos_ref[self.R_SHLD_X_IDX]
            self.joint_nominal_r_shld_z = joint_nominal_pos_ref[self.R_SHLD_Z_IDX]
            self.joint_nominal_r_elb_y = joint_nominal_pos_ref[self.R_ELB_Y_IDX]

        # 初始化当前关节位置为标称位置
        self.current_joint_positions[0] = self.joint_nominal_r_shld_y
        self.current_joint_positions[1] = self.joint_nominal_r_shld_x
        self.current_joint_positions[2] = self.joint_nominal_r_shld_z
        self.current_joint_positions[3] = self.joint_nominal_r_elb_y

    def start_handshake(self, current_sim_time):
        if not self.is_handshaking or self.is_shutting_down:
            # 如果之前正在关闭，需要从当前位置开始新的过渡
            self.is_handshaking = True
            self.is_starting_up = True
            self.is_shutting_down = False
            self.handshake_start_time = current_sim_time
            self.handshake_stop_time = None

    def stop_handshake(self, current_sim_time):
        if self.is_handshaking and not self.is_shutting_down:
            self.is_shutting_down = True
            self.is_starting_up = False
            self.handshake_stop_time = current_sim_time
            # 保存当前关节位置作为关闭过渡的起点
            self.shutdown_start_positions = self.current_joint_positions.copy()

    def calculate_handshake_motion(self, base_pos, time_in_seconds, loop_count_for_log=0):
        pos = base_pos.copy()

        if not self.is_handshaking and not self.is_shutting_down:
            return pos

        current_transition_factor = 0.0

        # 获取目标位置和标称位置
        target_r_shld_y = self.target_handshake_pose["r_shld_y"]
        target_r_shld_x = self.target_handshake_pose["r_shld_x"]
        target_r_shld_z = self.target_handshake_pose["r_shld_z"]
        target_r_elb_y = self.target_handshake_pose["r_elb_y"]

        nominal_r_shld_y = self.joint_nominal_r_shld_y
        nominal_r_shld_x = self.joint_nominal_r_shld_x
        nominal_r_shld_z = self.joint_nominal_r_shld_z
        nominal_r_elb_y = self.joint_nominal_r_elb_y
        
        # 关闭握手动作
        if self.is_shutting_down:
            if self.handshake_stop_time is None:
                self.is_handshaking = False
                self.is_shutting_down = False
                return pos
            
            shutdown_elapsed_time = time_in_seconds - self.handshake_stop_time
            
            # 确保平滑关闭过渡
            if shutdown_elapsed_time >= self.handshake_shutdown_duration:
                # 完全回到标称位置
                current_transition_factor = 0.0
                self.is_handshaking = False
                self.is_shutting_down = False
                self.handshake_start_time = None
                self.handshake_stop_time = None
                
                # 直接设置为标称位置
                pos[self.R_SHLD_Y_IDX] = nominal_r_shld_y
                pos[self.R_SHLD_X_IDX] = nominal_r_shld_x
                pos[self.R_SHLD_Z_IDX] = nominal_r_shld_z
                pos[self.R_ELB_Y_IDX] = nominal_r_elb_y
                
                # 更新当前关节位置
                self.current_joint_positions[0] = nominal_r_shld_y
                self.current_joint_positions[1] = nominal_r_shld_x
                self.current_joint_positions[2] = nominal_r_shld_z
                self.current_joint_positions[3] = nominal_r_elb_y
                
                return pos
            else:
                # 从停止时的位置平滑过渡到标称位置，使用二次缓动函数
                # 生成从1到0的平滑过渡因子
                t = shutdown_elapsed_time / self.handshake_shutdown_duration
                current_transition_factor = 1.0 - self._smooth_easing(t)
            
            # 从保存的停止位置插值到标称位置
            final_target_r_shld_y = self.shutdown_start_positions[0] * current_transition_factor + nominal_r_shld_y * (1.0 - current_transition_factor)
            final_target_r_shld_x = self.shutdown_start_positions[1] * current_transition_factor + nominal_r_shld_x * (1.0 - current_transition_factor)
            final_target_r_shld_z = self.shutdown_start_positions[2] * current_transition_factor + nominal_r_shld_z * (1.0 - current_transition_factor)
            final_target_r_elb_y = self.shutdown_start_positions[3] * current_transition_factor + nominal_r_elb_y * (1.0 - current_transition_factor)

        # 启动或维持握手动作
        elif self.is_handshaking:
            if self.handshake_start_time is None:
                return pos 

            startup_elapsed_time = time_in_seconds - self.handshake_start_time
            
            # 握手启动阶段
            if self.is_starting_up:
                if startup_elapsed_time >= self.handshake_startup_duration:
                    # 完全达到握手姿势
                    current_transition_factor = 1.0
                    self.is_starting_up = False
                else:
                    # 使用平滑缓动函数实现从标称姿势到握手姿势的平滑过渡
                    t = startup_elapsed_time / self.handshake_startup_duration
                    current_transition_factor = self._smooth_easing(t)
            else: 
                # 正常握手阶段（启动已完成）
                current_transition_factor = 1.0
            
            current_transition_factor = max(0.0, min(1.0, current_transition_factor))

            # 从标称姿势插值到目标握手姿势
            final_target_r_shld_y = nominal_r_shld_y * (1.0 - current_transition_factor) + target_r_shld_y * current_transition_factor
            final_target_r_shld_x = nominal_r_shld_x * (1.0 - current_transition_factor) + target_r_shld_x * current_transition_factor
            final_target_r_shld_z = nominal_r_shld_z * (1.0 - current_transition_factor) + target_r_shld_z * current_transition_factor
            final_target_r_elb_y = nominal_r_elb_y * (1.0 - current_transition_factor) + target_r_elb_y * current_transition_factor
        
        else: # 不应该发生，但作为后备
            return pos

        # 更新当前关节位置记录
        self.current_joint_positions[0] = final_target_r_shld_y
        self.current_joint_positions[1] = final_target_r_shld_x
        self.current_joint_positions[2] = final_target_r_shld_z
        self.current_joint_positions[3] = final_target_r_elb_y

        # 应用计算的关节角度
        pos[self.R_SHLD_Y_IDX] = final_target_r_shld_y
        pos[self.R_SHLD_X_IDX] = final_target_r_shld_x
        pos[self.R_SHLD_Z_IDX] = final_target_r_shld_z
        pos[self.R_ELB_Y_IDX] = final_target_r_elb_y

        return pos
    
    def _smooth_easing(self, t):
        """
        平滑缓动函数，提供比线性更加平滑的过渡
        t: 0到1之间的值，表示过渡进度
        返回: 平滑过渡的值（0到1之间）
        """
        # 使用三次方函数实现平滑缓动
        return t * t * (3.0 - 2.0 * t) 