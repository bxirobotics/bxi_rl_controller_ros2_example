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
        self.arm_freq = arm_freq
        self.arm_amp = arm_amp
        self.arm_base_height_y = arm_base_height_y
        self.arm_float_amp = arm_float_amp
        self.arm_startup_duration = arm_startup_duration
        self.arm_shutdown_duration = arm_startup_duration # 可以独立设置，这里复用启动时长
        
        self.arm_wave_start_time = None
        self.arm_wave_stop_time = None
        self.is_waving = False # 内部状态，标记是否应该挥舞 (启动完成到开始关闭前)
        self.is_starting_up = False # 标记是否正在启动过渡
        self.is_shutting_down = False # 标记是否正在关闭过渡

        # 索引常量，基于 bxi_example.py 中的 joint_name
        self.L_SHLD_Y_IDX = 15
        self.L_SHLD_Z_IDX = 17
        self.L_ELB_Y_IDX = 18

        if joint_nominal_pos_ref is None:
            # 提供一个默认值或者抛出错误，因为这个值对于肘部计算很重要
            self.logger.warn("joint_nominal_pos_ref not provided to ArmMotionController, using potentially incorrect defaults for elbow.")
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
            self.logger.info(f"Arm waving initiated at {current_sim_time:.2f}s.")

    def stop_waving(self, current_sim_time):
        # 只有在正在挥舞（包括启动完成）且尚未开始关闭时才触停止
        if self.is_waving and not self.is_shutting_down:
            self.is_shutting_down = True
            self.is_starting_up = False # 如果在启动中被停止，则取消启动状态
            self.arm_wave_stop_time = current_sim_time
            # self.is_waving 保持 True 直到关闭完成
            self.logger.info(f"Arm waving shutdown initiated at {current_sim_time:.2f}s.")

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
                self.logger.warn("ArmMotionController: is_shutting_down is True but arm_wave_stop_time is None. Forcing stop.")
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
                self.logger.info(f"Arm waving shutdown completed at {time_in_seconds:.2f}s.")
                return pos # 关闭完成，返回原始位置
            else:
                # 因子从1平滑到0
                current_wave_amplitude_factor = 1.0 - (shutdown_elapsed_time / self.arm_shutdown_duration)
        
        elif self.is_waving: # 包括 is_starting_up 和正常挥舞
            if self.arm_wave_start_time is None: # 安全检查
                self.logger.warn("ArmMotionController: is_waving is True but arm_wave_start_time is None. Defaulting to no wave.")
                return pos # 或者强制启动 self.start_waving(time_in_seconds) 并设置 factor 为 0

            startup_elapsed_time = time_in_seconds - self.arm_wave_start_time
            if self.is_starting_up:
                if startup_elapsed_time >= self.arm_startup_duration:
                    current_wave_amplitude_factor = 1.0
                    self.is_starting_up = False # 启动完成
                    self.logger.info(f"Arm waving startup completed at {time_in_seconds:.2f}s.")
                else:
                    # 因子从0平滑到1
                    current_wave_amplitude_factor = startup_elapsed_time / self.arm_startup_duration
            else: # 正常挥舞 (启动已完成)
                current_wave_amplitude_factor = 1.0
        
        current_wave_amplitude_factor = max(0.0, min(1.0, current_wave_amplitude_factor))

        # --- 应用 current_wave_amplitude_factor 到所有运动计算 ---
        # 左肩Y轴 (l_shld_y_joint)
        # 目标基础Y抬升位置是从当前策略给出的位置向 arm_base_height_y 过渡
        # 这个过渡本身也受 current_wave_amplitude_factor (代表启动或关闭的整体进度) 影响
        current_l_shld_y_from_policy = base_pos[self.L_SHLD_Y_IDX]
        # target_base_y_lift 是指挥舞动作的Y轴中心，这个中心本身在启动/关闭时平滑变化
        # 当 factor=0, target_base_y_lift = current_l_shld_y_from_policy
        # 当 factor=1, target_base_y_lift = self.arm_base_height_y
        target_base_y_lift = current_l_shld_y_from_policy + \
                             (self.arm_base_height_y - current_l_shld_y_from_policy) * current_wave_amplitude_factor
        
        # 浮动量也受整体因子影响
        float_y_movement = self.arm_float_amp * math.sin(2 * math.pi * self.arm_freq * time_in_seconds) * current_wave_amplitude_factor
        final_target_l_shld_y = target_base_y_lift + float_y_movement
        pos[self.L_SHLD_Y_IDX] = final_target_l_shld_y

        # 左肩Z轴 (l_shld_z_joint)
        # Z轴的摆动也受整体因子影响
        wave_z_movement = 0.5 * self.arm_amp * math.sin(2 * math.pi * self.arm_freq * time_in_seconds + math.pi / 2) * current_wave_amplitude_factor
        # Z轴摆动的中心是其在 base_pos 中的值 (策略输出)
        final_target_l_shld_z = base_pos[self.L_SHLD_Z_IDX] + wave_z_movement
        pos[self.L_SHLD_Z_IDX] = final_target_l_shld_z

        # 左肘Y轴 (l_elb_y_joint)
        # 肘部的跟随运动也受整体因子影响 (通过 float_y_movement 间接实现)
        final_target_l_elb_y = self.joint_nominal_l_elb_y + (0.1 * float_y_movement) # float_y_movement 已包含 current_wave_amplitude_factor
        pos[self.L_ELB_Y_IDX] = final_target_l_elb_y

        if loop_count_for_log % 100 == 0: # 每100次循环（约1秒）打印一次日志
            status_str = "Idle"
            if self.is_starting_up: status_str = "StartingUp"
            elif self.is_shutting_down: status_str = "ShuttingDown"
            elif self.is_waving: status_str = "WavingActive"

            self.logger.info(f"ArmCtrl Debug @ {time_in_seconds:.2f}s (Factor: {current_wave_amplitude_factor:.3f}, Status: {status_str}):")
            self.logger.info(f"  LShY: BasePol={current_l_shld_y_from_policy:.3f}, TgtBaseY={target_base_y_lift:.3f}, Float={float_y_movement:.3f}, FinalTgtY={final_target_l_shld_y:.3f}")
            self.logger.info(f"  LShZ: BasePol={base_pos[self.L_SHLD_Z_IDX]:.3f}, WaveZ={wave_z_movement:.3f}, FinalTgtZ={final_target_l_shld_z:.3f}")
            self.logger.info(f"  LElbY: Nom={self.joint_nominal_l_elb_y:.3f}, FinalTgtY={final_target_l_elb_y:.3f}")
            
        return pos
