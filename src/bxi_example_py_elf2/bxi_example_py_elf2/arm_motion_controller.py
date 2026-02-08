import math
import numpy as np

# joint_nominal_pos 需要从主模块或者配置文件中获取
# 为了示例，我们在这里定义一个简化的版本，实际应用中需要正确传递或加载
# 这个值与 bxi_example.py 中的 joint_nominal_pos 一致，特别是手臂部分
# joint_nominal_pos_arm_related = {
#     "l_shld_y": 0.7,
#     "l_shld_z": -0.1,
#     "l_elb_y": -1.5,
# }
# 或者直接传递整个 joint_nominal_pos 数组给控制器


class ArmMotionController:
    def __init__(self, logger, arm_freq=0.3, arm_amp=0.5, arm_base_height_y=-1.2,
                 arm_float_amp=0.4, arm_startup_duration=2.0, joint_nominal_pos_ref=None):
        self.logger = logger
        self.arm_freq = arm_freq  # 手臂挥舞频率
        self.arm_amp = arm_amp  # 挥舞幅度
        self.arm_base_height_y = arm_base_height_y  # 手臂抬起高度（肩关节Y轴）
        self.arm_float_amp = arm_float_amp  # 上下浮动幅度
        self.arm_startup_duration = arm_startup_duration  # 启动过渡时间
        self.arm_shutdown_duration = arm_startup_duration  # 关闭过渡时间
        
        self.arm_wave_start_time = None
        self.arm_wave_stop_time = None
        self.is_waving = False # 内部状态，标记是否应该挥舞 (启动完成到开始关闭前)
        self.is_starting_up = False # 标记是否正在启动过渡
        self.is_shutting_down = False # 标记是否正在关闭过渡

        # 索引常量，基于 bxi_example.py 中的 joint_name
        self.L_SHLD_Y_IDX = 15
        self.L_SHLD_Z_IDX = 17 # l_shld_z_joint
        self.L_ELB_Y_IDX = 18  # l_elb_y_joint
        # 注意：代码中之前对 L_SHLD_Z_IDX 的注释是 l_shld_z_joint，但实际用的是17，对应 l_shld_x_joint。
        # 假设这里控制的是 Y, X, ELB_Y 三个关节，或者 Y, Z, ELB_Y。
        # 根据 joint_name 顺序:
        # 15: "l_shld_y_joint"
        # 16: "l_shld_x_joint"
        # 17: "l_shld_z_joint"
        # 18: "l_elb_y_joint"
        # 如果之前 L_SHLD_Z_IDX = 17 是指 l_shld_z_joint，那么它是正确的。
        # 如果是指 l_shld_x_joint，那么索引是16。
        # 我将假设之前的索引是正确的，即控制 l_shld_y, l_shld_z, l_elb_y。

        self.last_calculated_arm_pos = np.zeros(3) # To store [l_shld_y, l_shld_z, l_elb_y]

        if joint_nominal_pos_ref is None:
            # 提供一个默认值或者抛出错误，因为这个值对于肘部计算很重要
            # 使用一个基于之前观察到的默认值，但这非常不推荐
            self.joint_nominal_l_elb_y = -1.5
        else:
            self.joint_nominal_l_elb_y = joint_nominal_pos_ref[self.L_ELB_Y_IDX]


    def start_waving(self, current_sim_time):
        # 只有在完全停止或正在关闭（此时取消关闭并重新启动）时才启动
        if not self.is_waving or self.is_shutting_down:
            self.is_waving = True # 标记开始挥舞（包含启动过渡）
            self.is_starting_up = True
            self.is_shutting_down = False
            self.arm_wave_start_time = current_sim_time
            self.arm_wave_stop_time = None # 清除停止时间

    def stop_waving(self, current_sim_time):
        # 只有在正在挥舞（包括启动完成）且尚未开始关闭时才触停止
        if self.is_waving and not self.is_shutting_down:
            self.is_shutting_down = True
            self.is_starting_up = False # 如果在启动中被停止，则取消启动状态
            self.arm_wave_stop_time = current_sim_time
            # self.is_waving 保持 True 直到关闭完成

    def calculate_arm_waving(self, base_pos, time_in_seconds, loop_count_for_log=0):
        """
        计算手臂挥舞动作的目标位置。
        
        参数:
            base_pos: 基础关节位置数组 (应为当前机器人的完整 qpos 副本)
            time_in_seconds: 当前仿真时间(秒)
            loop_count_for_log: 用于日志打印节流的循环计数器
        
        返回:
            更新后的关节位置数组，包含挥舞动作
        """
        pos = base_pos.copy() # 操作副本，不直接修改传入的 base_pos

        if not self.is_waving and not self.is_shutting_down: # 如果完全停止，则不进行任何计算
            return pos

        current_wave_amplitude_factor = 0.0

        if self.is_shutting_down:
            if self.arm_wave_stop_time is None: # 安全检查，理论上不应发生
                self.is_waving = False
                self.is_shutting_down = False
                return pos
            
            shutdown_elapsed_time = time_in_seconds - self.arm_wave_stop_time
            if shutdown_elapsed_time >= self.arm_shutdown_duration:
                current_wave_amplitude_factor = 0.0
                self.is_waving = False
                self.is_shutting_down = False
                self.arm_wave_start_time = None
                self.arm_wave_stop_time = None
                return pos # 关闭完成，返回原始位置
            else:
                # 因子从1平滑到0
                current_wave_amplitude_factor = 1.0 - (shutdown_elapsed_time / self.arm_shutdown_duration)
        
        elif self.is_waving: # 包括 is_starting_up 和正常挥舞
            if self.arm_wave_start_time is None: # 安全检查
                return pos # 或者强制启动 self.start_waving(time_in_seconds) 并设置 factor 为 0

            startup_elapsed_time = time_in_seconds - self.arm_wave_start_time
            if self.is_starting_up:
                if startup_elapsed_time >= self.arm_startup_duration:
                    current_wave_amplitude_factor = 1.0
                    self.is_starting_up = False # 启动完成
                else:
                    # 因子从0平滑到1
                    current_wave_amplitude_factor = startup_elapsed_time / self.arm_startup_duration
            else: # 正常挥舞 (启动已完成)
                current_wave_amplitude_factor = 1.0
        
        current_wave_amplitude_factor = max(0.0, min(1.0, current_wave_amplitude_factor))

        # --- 计算目标手臂关节位置 ---
        final_target_l_shld_y = 0.0
        final_target_l_shld_z = 0.0
        final_target_l_elb_y = 0.0

        if self.is_shutting_down:
            shutdown_progress = (time_in_seconds - self.arm_wave_stop_time) / self.arm_shutdown_duration
            shutdown_progress = max(0.0, min(1.0, shutdown_progress)) # clamp to [0,1]

            # 从 last_calculated_arm_pos 插值到 base_pos (策略期望的静止位)
            final_target_l_shld_y = self.last_calculated_arm_pos[0] * (1.0 - shutdown_progress) + base_pos[self.L_SHLD_Y_IDX] * shutdown_progress
            final_target_l_shld_z = self.last_calculated_arm_pos[1] * (1.0 - shutdown_progress) + base_pos[self.L_SHLD_Z_IDX] * shutdown_progress
            final_target_l_elb_y = self.last_calculated_arm_pos[2] * (1.0 - shutdown_progress) + base_pos[self.L_ELB_Y_IDX] * shutdown_progress
        
        elif self.is_waving: # 启动中或正常挥舞
            # 左肩Y轴 (l_shld_y_joint) - 从大腿旁边抬起到固定高度
            current_l_shld_y_from_policy = base_pos[self.L_SHLD_Y_IDX]
            # 将手臂抬到固定高度，不添加上下浮动
            final_target_l_shld_y = current_l_shld_y_from_policy + \
                                  (self.arm_base_height_y - current_l_shld_y_from_policy) * current_wave_amplitude_factor

            # 左肩Z轴 (l_shld_z_joint) - 专注于左右摆动（简单的挥舞）
            # 增加左右摆动幅度，使挥舞动作更明显
            wave_z_movement = self.arm_amp * math.sin(2 * math.pi * self.arm_freq * time_in_seconds) * current_wave_amplitude_factor
            final_target_l_shld_z = base_pos[self.L_SHLD_Z_IDX] + wave_z_movement
            
            # 左肘Y轴 (l_elb_y_joint) - 保持固定的弯曲度
            # 固定肘部弯曲，不随着肩部运动而变化
            elbow_bend_base = -0.6  # 基础弯曲程度，适中
            final_target_l_elb_y = elbow_bend_base * current_wave_amplitude_factor + \
                                 base_pos[self.L_ELB_Y_IDX] * (1.0 - current_wave_amplitude_factor)

            # 存储当前计算的挥舞目标，供关闭时使用
            self.last_calculated_arm_pos[0] = final_target_l_shld_y
            self.last_calculated_arm_pos[1] = final_target_l_shld_z
            self.last_calculated_arm_pos[2] = final_target_l_elb_y
        
        else: # 完全停止 (is_waving is False, is_shutting_down is False)
            return pos # 直接返回原始pos

        pos[self.L_SHLD_Y_IDX] = final_target_l_shld_y
        pos[self.L_SHLD_Z_IDX] = final_target_l_shld_z
        pos[self.L_ELB_Y_IDX] = final_target_l_elb_y

        return pos 