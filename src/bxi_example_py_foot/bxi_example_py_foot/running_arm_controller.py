import math
import numpy as np

class RunningArmController:
    """
    实现机器人奔跑时的手臂协调运动控制器。
    """
    def __init__(self, logger, joint_nominal_pos_ref,
                 arm_startup_duration=2.0, arm_shutdown_duration=2.0,
                 arm_amplitude_y=0.8, arm_amplitude_z=0.06, elbow_coeff=0.1,
                 smoothing_factor=0.8):
        self.logger = logger
        self.joint_nominal_pos_ref = joint_nominal_pos_ref
        self.arm_startup_duration = arm_startup_duration
        self.arm_shutdown_duration = arm_shutdown_duration
        self.arm_amplitude_y = arm_amplitude_y  # 肩部Y轴摆动幅度 (前后摆动，进一步减小以使动作更柔和)
        self.arm_amplitude_z = arm_amplitude_z  # 肩部Z轴摆动幅度 (极轻微内外摆动，提高自然度)
        self.elbow_coeff = elbow_coeff  # 肘部运动系数 (与肩部Y轴相关，更小值使弯曲更柔和自然，保持整体平衡)
        self.smoothing_factor = smoothing_factor  # 腿部相位信号的平滑因子 (越大平滑效果越明显)

        self.motion_start_time = None
        self.motion_stop_time = None
        self.is_running_motion = False
        self.is_starting_up = False
        self.is_shutting_down = False

        # 从bxi_example.py的joint_name元组中获取关节索引
        # 左臂关节索引
        self.L_SHLD_Y_IDX = 15  # 左肩Y轴关节
        self.L_SHLD_X_IDX = 16  # 左肩X轴关节 
        self.L_SHLD_Z_IDX = 17  # 左肩Z轴关节
        self.L_ELB_Y_IDX = 18   # 左肘Y轴关节
        # 右臂关节索引
        self.R_SHLD_Y_IDX = 20  # 右肩Y轴关节
        self.R_SHLD_X_IDX = 21  # 右肩X轴关节 
        self.R_SHLD_Z_IDX = 22  # 右肩Z轴关节
        self.R_ELB_Y_IDX = 23   # 右肘Y轴关节

        # Store nominal positions for convenience
        self.nominal_l_shld_y = self.joint_nominal_pos_ref[self.L_SHLD_Y_IDX]
        self.nominal_l_shld_z = self.joint_nominal_pos_ref[self.L_SHLD_Z_IDX]
        self.nominal_l_elb_y = self.joint_nominal_pos_ref[self.L_ELB_Y_IDX]
        self.nominal_r_shld_y = self.joint_nominal_pos_ref[self.R_SHLD_Y_IDX]
        self.nominal_r_shld_z = self.joint_nominal_pos_ref[self.R_SHLD_Z_IDX]
        self.nominal_r_elb_y = self.joint_nominal_pos_ref[self.R_ELB_Y_IDX]

        # 保存当前实际关节角度，用于平滑过渡
        self.current_joint_positions = {
            "l_shld_y": self.nominal_l_shld_y, 
            "l_shld_z": self.nominal_l_shld_z, 
            "l_elb_y": self.nominal_l_elb_y,
            "r_shld_y": self.nominal_r_shld_y, 
            "r_shld_z": self.nominal_r_shld_z, 
            "r_elb_y": self.nominal_r_elb_y,
        }
        
        # 开始关闭时的关节位置，用于平滑过渡
        self.shutdown_start_positions = self.current_joint_positions.copy()
        
        # 添加平滑过滤器
        self.prev_leg_phase_left = 0.0
        self.prev_leg_phase_right = 0.0
        
        # 添加用于自然摆动的低通滤波器
        self.filtered_l_shld_y_swing = 0.0
        self.filtered_l_shld_z_swing = 0.0
        self.filtered_r_shld_y_swing = 0.0
        self.filtered_r_shld_z_swing = 0.0

    def start_running_motion(self, current_sim_time):
        """
        启动奔跑手臂运动。
        
        如果控制器当前未运行或正在关闭，则启动运动过程。
        
        参数:
            current_sim_time (float): 当前仿真时间（秒）
        """
        if not self.is_running_motion or self.is_shutting_down:
            self.is_running_motion = True
            self.is_starting_up = True
            self.is_shutting_down = False
            self.motion_start_time = current_sim_time
            self.motion_stop_time = None

    def stop_running_motion(self, current_sim_time):
        """
        停止奔跑手臂运动。
        
        如果控制器当前正在运行且未处于关闭状态，则启动关闭过程。
        
        参数:
            current_sim_time (float): 当前仿真时间（秒）
        """
        if self.is_running_motion and not self.is_shutting_down:
            self.is_shutting_down = True
            self.is_starting_up = False
            self.motion_stop_time = current_sim_time
            # 保存当前关节位置作为关闭过渡的起点
            self.shutdown_start_positions = self.current_joint_positions.copy()

    def calculate_running_arm_motion(self, base_pos, time_in_seconds, leg_phase_left_signal, leg_phase_right_signal, loop_count_for_log=0):
        """
        计算奔跑时手臂的协调运动。
        
        根据腿部相位信号计算手臂关节角度，实现手臂与腿部的自然协调。
        
        参数:
            base_pos (numpy.ndarray): 基础关节位置数组
            time_in_seconds (float): 当前仿真时间（秒）
            leg_phase_left_signal (float): 左腿相位信号，范围[-1,1]
            leg_phase_right_signal (float): 右腿相位信号，范围[-1,1]
            loop_count_for_log (int): 循环计数，用于控制日志输出频率
            
        返回:
            numpy.ndarray: 更新后的关节位置数组
        """
        pos = base_pos.copy()

        # 如果控制器未激活，直接返回基础位置
        if not self.is_running_motion and not self.is_shutting_down:
            return pos

        # 初始化动作幅度因子
        current_motion_amplitude_factor = 0.0

        # ===== 处理关闭状态 =====
        if self.is_shutting_down:
            if self.motion_stop_time is None:
                self.is_running_motion = False
                self.is_shutting_down = False
                return pos
            
            # 计算关闭过渡的进度
            shutdown_elapsed_time = time_in_seconds - self.motion_stop_time
            if shutdown_elapsed_time >= self.arm_shutdown_duration:
                # 关闭过渡完成，重置所有状态
                current_motion_amplitude_factor = 0.0
                self.is_running_motion = False
                self.is_shutting_down = False
                self.motion_start_time = None
                self.motion_stop_time = None
                
                # 完全回到标称位置
                pos[self.L_SHLD_Y_IDX] = self.nominal_l_shld_y
                pos[self.L_SHLD_Z_IDX] = self.nominal_l_shld_z
                pos[self.L_ELB_Y_IDX] = self.nominal_l_elb_y
                pos[self.R_SHLD_Y_IDX] = self.nominal_r_shld_y
                pos[self.R_SHLD_Z_IDX] = self.nominal_r_shld_z
                pos[self.R_ELB_Y_IDX] = self.nominal_r_elb_y
                
                # 更新当前关节位置
                self.current_joint_positions["l_shld_y"] = self.nominal_l_shld_y
                self.current_joint_positions["l_shld_z"] = self.nominal_l_shld_z
                self.current_joint_positions["l_elb_y"] = self.nominal_l_elb_y
                self.current_joint_positions["r_shld_y"] = self.nominal_r_shld_y
                self.current_joint_positions["r_shld_z"] = self.nominal_r_shld_z
                self.current_joint_positions["r_elb_y"] = self.nominal_r_elb_y
                
                return pos
            else:
                # 使用平滑缓动函数计算过渡因子
                t = shutdown_elapsed_time / self.arm_shutdown_duration
                current_motion_amplitude_factor = 1.0 - self._smooth_easing(t)
        
        # ===== 处理运行状态（包括启动和正常运行） =====
        elif self.is_running_motion:
            if self.motion_start_time is None:
                return pos

            # 计算启动过渡的进度
            startup_elapsed_time = time_in_seconds - self.motion_start_time
            if self.is_starting_up:
                if startup_elapsed_time >= self.arm_startup_duration:
                    # 启动过渡完成，进入正常运行状态
                    current_motion_amplitude_factor = 1.0
                    self.is_starting_up = False
                else:
                    # 使用平滑缓动函数计算过渡因子
                    t = startup_elapsed_time / self.arm_startup_duration
                    current_motion_amplitude_factor = self._smooth_easing(t)
            else: # 正常运行状态
                current_motion_amplitude_factor = 1.0
        
        # 确保动作幅度因子在有效范围内
        current_motion_amplitude_factor = max(0.0, min(1.0, current_motion_amplitude_factor))

        # ===== 计算目标手臂关节位置 =====
        # 基于腿部相位计算手臂运动
        # "迈左腿时右手臂向前挥，迈右腿时左手臂向前挥"
        
        # 平滑处理腿部相位信号，减少突变
        smoothed_leg_phase_left = self._smooth_signal(leg_phase_left_signal, self.prev_leg_phase_left, self.smoothing_factor)
        smoothed_leg_phase_right = self._smooth_signal(leg_phase_right_signal, self.prev_leg_phase_right, self.smoothing_factor)
        
        # 更新上一次的相位值，用于下一次平滑
        self.prev_leg_phase_left = smoothed_leg_phase_left
        self.prev_leg_phase_right = smoothed_leg_phase_right
        
        # 左臂 (与右腿相位相反，实现交叉协调)
        # 如果右腿前伸(leg_phase_right_signal > 0)，左臂后摆
        raw_l_shld_y_swing = -self.arm_amplitude_y * smoothed_leg_phase_right
        raw_l_shld_z_swing = -self.arm_amplitude_z * smoothed_leg_phase_right
        
        # 进一步平滑肩部摆动，提供更柔和的过渡
        self.filtered_l_shld_y_swing = self._smooth_signal(raw_l_shld_y_swing, self.filtered_l_shld_y_swing, 0.3)
        self.filtered_l_shld_z_swing = self._smooth_signal(raw_l_shld_z_swing, self.filtered_l_shld_z_swing, 0.3)
        
        # 计算左肩目标角度
        target_l_shld_y = self.nominal_l_shld_y + self.filtered_l_shld_y_swing * current_motion_amplitude_factor
        target_l_shld_z = self.nominal_l_shld_z + self.filtered_l_shld_z_swing * current_motion_amplitude_factor
        
        # 直臂模式：肘部保持标称位置，大臂小臂成直线
        # elbow_motion_l = self.elbow_coeff * smoothed_leg_phase_right * current_motion_amplitude_factor
        target_l_elb_y = self.nominal_l_elb_y  # 直接设为标称位置，无弯曲
        
        # 右臂 (与左腿相位相反，实现交叉协调)
        # 如果左腿前伸(leg_phase_left_signal > 0)，右臂后摆
        raw_r_shld_y_swing = -self.arm_amplitude_y * smoothed_leg_phase_left
        raw_r_shld_z_swing = -self.arm_amplitude_z * smoothed_leg_phase_left
        
        # 进一步平滑肩部摆动，提供更柔和的过渡
        self.filtered_r_shld_y_swing = self._smooth_signal(raw_r_shld_y_swing, self.filtered_r_shld_y_swing, 0.3)
        self.filtered_r_shld_z_swing = self._smooth_signal(raw_r_shld_z_swing, self.filtered_r_shld_z_swing, 0.3)
        
        # 计算右肩目标角度
        target_r_shld_y = self.nominal_r_shld_y + self.filtered_r_shld_y_swing * current_motion_amplitude_factor
        target_r_shld_z = self.nominal_r_shld_z + self.filtered_r_shld_z_swing * current_motion_amplitude_factor
        
        # 右肘直臂模式：保持标称位置，大臂小臂成直线
        # elbow_motion_r = self.elbow_coeff * smoothed_leg_phase_left * current_motion_amplitude_factor
        target_r_elb_y = self.nominal_r_elb_y  # 直接设为标称位置，无弯曲

        if self.is_shutting_down:
            # 改进关闭过渡逻辑，确保平滑过渡到标称位置
            # 采用目标位置和当前位置之间的线性插值，而不是直接从保存的停止位置过渡
            
            # 计算每个关节的目标位置和当前位置之间的线性插值
            lerp_factor = 1.0 - current_motion_amplitude_factor  # 从当前位置到标称位置的插值因子
            
            # 左臂平滑过渡
            pos[self.L_SHLD_Y_IDX] = self.shutdown_start_positions["l_shld_y"] * current_motion_amplitude_factor + self.nominal_l_shld_y * lerp_factor
            pos[self.L_SHLD_Z_IDX] = self.shutdown_start_positions["l_shld_z"] * current_motion_amplitude_factor + self.nominal_l_shld_z * lerp_factor
            pos[self.L_ELB_Y_IDX] = self.shutdown_start_positions["l_elb_y"] * current_motion_amplitude_factor + self.nominal_l_elb_y * lerp_factor
            
            # 右臂平滑过渡
            pos[self.R_SHLD_Y_IDX] = self.shutdown_start_positions["r_shld_y"] * current_motion_amplitude_factor + self.nominal_r_shld_y * lerp_factor
            pos[self.R_SHLD_Z_IDX] = self.shutdown_start_positions["r_shld_z"] * current_motion_amplitude_factor + self.nominal_r_shld_z * lerp_factor
            pos[self.R_ELB_Y_IDX] = self.shutdown_start_positions["r_elb_y"] * current_motion_amplitude_factor + self.nominal_r_elb_y * lerp_factor
        else: # Startup or active running
            # 应用计算出的关节角度
            pos[self.L_SHLD_Y_IDX] = target_l_shld_y
            pos[self.L_SHLD_Z_IDX] = target_l_shld_z
            pos[self.L_ELB_Y_IDX] = target_l_elb_y
            pos[self.R_SHLD_Y_IDX] = target_r_shld_y
            pos[self.R_SHLD_Z_IDX] = target_r_shld_z
            pos[self.R_ELB_Y_IDX] = target_r_elb_y

            # 更新当前关节位置
            self.current_joint_positions["l_shld_y"] = target_l_shld_y
            self.current_joint_positions["l_shld_z"] = target_l_shld_z
            self.current_joint_positions["l_elb_y"] = target_l_elb_y
            self.current_joint_positions["r_shld_y"] = target_r_shld_y
            self.current_joint_positions["r_shld_z"] = target_r_shld_z
            self.current_joint_positions["r_elb_y"] = target_r_elb_y

        return pos
        
    def _smooth_easing(self, t):
        """
        改进的平滑缓动函数，采用三阶埃尔米特插值
        t: 0到1之间的值，表示过渡进度
        返回: 平滑过渡的值（0到1之间）
        """
        # 确保 t 在 [0, 1] 区间内
        t = max(0.0, min(1.0, t))
        
        # 使用标准三阶埃尔米特插值函数 (更温和的S曲线)
        # 公式: 3t^2 - 2t^3
        return 3.0 * (t**2) - 2.0 * (t**3)
        
    def _smooth_signal(self, new_value, prev_value, smoothing_factor):
        """
        信号平滑函数，用于减少信号噪声和突变
        
        参数:
            new_value (float): 新的信号值
            prev_value (float): 上一个信号值
            smoothing_factor (float): 平滑因子，范围[0,1]，越大平滑效果越强
            
        返回:
            float: 平滑后的信号值
        """
        return prev_value * smoothing_factor + new_value * (1.0 - smoothing_factor)

    # Helper properties for bxi_example.py to check status
    @property
    def is_active(self): # Equivalent to old is_waving for general activity check
        return self.is_running_motion

    @property
    def is_active_or_shutting_down(self): # For checking if calculation is needed
        return self.is_running_motion or self.is_shutting_down 