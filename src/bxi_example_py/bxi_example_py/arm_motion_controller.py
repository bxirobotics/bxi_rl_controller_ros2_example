import math
import numpy as np

# joint_nominal_pos 需要从主模块或者配置文件中获取
# 为了示例，我们在这里定义一个简化的版本，实际应用中需要正确传递或加载
# 这个值应该与 bxi_example.py 中的 joint_nominal_pos 一致，特别是手臂部分
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
        
        self.arm_wave_start_time = None
        self.is_waving = False # 内部状态，标记是否应该挥舞

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
        if not self.is_waving:
            self.is_waving = True
            self.arm_wave_start_time = current_sim_time
            self.logger.info(f"Arm waving started at {current_sim_time:.2f}s.")

    def stop_waving(self):
        if self.is_waving:
            self.is_waving = False
            self.arm_wave_start_time = None # 重置启动时间
            self.logger.info("Arm waving stopped.")

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

        if not self.is_waving: # 如果没有在挥舞状态，直接返回原始pos
            return pos

        startup_factor = 1.0
        if self.arm_wave_start_time is not None:
            elapsed_time = time_in_seconds - self.arm_wave_start_time
            if elapsed_time < self.arm_startup_duration:
                startup_factor = elapsed_time / self.arm_startup_duration
            else:
                startup_factor = 1.0
            startup_factor = max(0.0, min(1.0, startup_factor)) # 确保在0-1之间
        else:
            # 如果 arm_wave_start_time 为 None 但 is_waving 为 True (理论上不应发生，除非 start_waving 未正确调用)
            # 为安全起见，不进行挥舞
            self.logger.warn("ArmMotionController: is_waving is True but arm_wave_start_time is None. Defaulting to no wave.")
            startup_factor = 0.0
            # 或者可以强制调用 start_waving:
            # self.start_waving(time_in_seconds)
            # startup_factor = 0.0 # 第一次调用，从0开始

        # 左肩Y轴 (l_shld_y_joint)
        current_l_shld_y = base_pos[self.L_SHLD_Y_IDX]
        target_base_y_lift = current_l_shld_y + (self.arm_base_height_y - current_l_shld_y) * startup_factor
        float_y_movement = self.arm_float_amp * math.sin(2 * math.pi * self.arm_freq * time_in_seconds) * startup_factor
        final_target_l_shld_y = target_base_y_lift + float_y_movement
        pos[self.L_SHLD_Y_IDX] = final_target_l_shld_y

        # 左肩Z轴 (l_shld_z_joint)
        current_l_shld_z = base_pos[self.L_SHLD_Z_IDX] # 获取当前Z轴位置
        # 目标Z轴基础位置也应该从当前位置平滑过渡到标称位置（或一个中心摆动位置）
        # 这里简化为从当前 policy/nominal 给出的 base_pos[17] 开始摆动
        wave_z_movement = 0.5 * self.arm_amp * math.sin(2 * math.pi * self.arm_freq * time_in_seconds + math.pi / 2) * startup_factor
        final_target_l_shld_z = current_l_shld_z + (0.0 - current_l_shld_z + wave_z_movement) * startup_factor # 从当前位置向 (标称位置+摆动) 过渡
        # 上述逻辑可能复杂了，简化：直接在当前策略给出的Z轴基础上叠加平滑启动的摆动
        # final_target_l_shld_z = base_pos[self.L_SHLD_Z_IDX] + wave_z_movement
        # 进一步修正：Z轴的摆动中心应该是其标称值，摆动从标称值开始，并应用startup_factor
        # nominal_l_shld_z = joint_nominal_pos_ref[self.L_SHLD_Z_IDX] # 需要传入 joint_nominal_pos
        # 为了简化，我们假设 base_pos[self.L_SHLD_Z_IDX] 已经是策略期望的中心，我们只叠加启动的wave
        final_target_l_shld_z = base_pos[self.L_SHLD_Z_IDX] + wave_z_movement # 保持与之前版本相似的逻辑，在策略输出上叠加
        pos[self.L_SHLD_Z_IDX] = final_target_l_shld_z


        # 左肘Y轴 (l_elb_y_joint)
        # 肘部的目标位置是其标称位置，加上一个与肩部Y轴*最终*浮动量（float_y_movement）成比例的调整
        # 这个调整也应该平滑启动。由于 float_y_movement 已经包含了 startup_factor。
        final_target_l_elb_y = self.joint_nominal_l_elb_y + (0.1 * float_y_movement) # 使用 0.1 作为肘部跟随系数
        pos[self.L_ELB_Y_IDX] = final_target_l_elb_y

        if loop_count_for_log % 100 == 0: # 每100次循环（约1秒）打印一次日志
            self.logger.info(f"ArmCtrl Debug @ {time_in_seconds:.2f}s (SF: {startup_factor:.3f}):")
            self.logger.info(f"  LShY: Cur={current_l_shld_y:.3f}, TgtBaseY={target_base_y_lift:.3f}, Float={float_y_movement:.3f}, FinalTgtY={final_target_l_shld_y:.3f}")
            self.logger.info(f"  LShZ: Base={base_pos[self.L_SHLD_Z_IDX]:.3f}, WaveZ={wave_z_movement:.3f}, FinalTgtZ={final_target_l_shld_z:.3f}")
            self.logger.info(f"  LElbY: Nom={self.joint_nominal_l_elb_y:.3f}, FinalTgtY={final_target_l_elb_y:.3f}")
            
        return pos
