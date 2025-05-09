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
        
        self.last_calculated_handshake_pos = np.zeros(4) # To store [r_shld_y, r_shld_x, r_shld_z, r_elb_y]

        if joint_nominal_pos_ref is None:
            self.logger.warn("joint_nominal_pos_ref not provided to RightArmHandshakeController, using default for target pose which might be suboptimal.")
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

        # Update target_handshake_pose with nominal for r_shld_z if not explicitly set differently
        # self.target_handshake_pose["r_shld_z"] = self.joint_nominal_r_shld_z


    def start_handshake(self, current_sim_time):
        if not self.is_handshaking or self.is_shutting_down:
            self.is_handshaking = True
            self.is_starting_up = True
            self.is_shutting_down = False
            self.handshake_start_time = current_sim_time
            self.handshake_stop_time = None
            self.logger.info(f"Right arm handshake initiated at {current_sim_time:.2f}s.")

    def stop_handshake(self, current_sim_time):
        if self.is_handshaking and not self.is_shutting_down:
            self.is_shutting_down = True
            self.is_starting_up = False
            self.handshake_stop_time = current_sim_time
            self.logger.info(f"Right arm handshake shutdown initiated at {current_sim_time:.2f}s.")

    def calculate_handshake_motion(self, base_pos, time_in_seconds, loop_count_for_log=0):
        pos = base_pos.copy()

        if not self.is_handshaking and not self.is_shutting_down:
            return pos

        current_transition_factor = 0.0

        # Target positions for handshake
        target_r_shld_y = self.target_handshake_pose["r_shld_y"]
        target_r_shld_x = self.target_handshake_pose["r_shld_x"]
        target_r_shld_z = self.target_handshake_pose["r_shld_z"] # Using the value from init
        target_r_elb_y = self.target_handshake_pose["r_elb_y"]

        # Nominal (return) positions
        nominal_r_shld_y = self.joint_nominal_r_shld_y
        nominal_r_shld_x = self.joint_nominal_r_shld_x
        nominal_r_shld_z = self.joint_nominal_r_shld_z
        nominal_r_elb_y = self.joint_nominal_r_elb_y
        
        # If shutting down, interpolate from last calculated/current target to nominal
        if self.is_shutting_down:
            if self.handshake_stop_time is None:
                self.logger.warn("RightArmHandshakeController: is_shutting_down is True but handshake_stop_time is None. Forcing stop.")
                self.is_handshaking = False
                self.is_shutting_down = False
                return pos
            
            shutdown_elapsed_time = time_in_seconds - self.handshake_stop_time
            if shutdown_elapsed_time >= self.handshake_shutdown_duration:
                current_transition_factor = 0.0 # Fully returned to nominal
                self.is_handshaking = False
                self.is_shutting_down = False
                self.handshake_start_time = None
                self.handshake_stop_time = None
                self.logger.info(f"Right arm handshake shutdown completed at {time_in_seconds:.2f}s.")
                # Set to nominal positions directly
                pos[self.R_SHLD_Y_IDX] = nominal_r_shld_y
                pos[self.R_SHLD_X_IDX] = nominal_r_shld_x
                pos[self.R_SHLD_Z_IDX] = nominal_r_shld_z
                pos[self.R_ELB_Y_IDX] = nominal_r_elb_y
                return pos
            else:
                # Factor from 1 (handshake pose) to 0 (nominal pose)
                current_transition_factor = 1.0 - (shutdown_elapsed_time / self.handshake_shutdown_duration)
            
            # Interpolate from target handshake pose to nominal pose
            final_target_r_shld_y = target_r_shld_y * current_transition_factor + nominal_r_shld_y * (1.0 - current_transition_factor)
            final_target_r_shld_x = target_r_shld_x * current_transition_factor + nominal_r_shld_x * (1.0 - current_transition_factor)
            final_target_r_shld_z = target_r_shld_z * current_transition_factor + nominal_r_shld_z * (1.0 - current_transition_factor)
            final_target_r_elb_y = target_r_elb_y * current_transition_factor + nominal_r_elb_y * (1.0 - current_transition_factor)

        # If starting up or active, interpolate from nominal to target handshake pose
        elif self.is_handshaking: # Includes is_starting_up and normal handshaking
            if self.handshake_start_time is None:
                self.logger.warn("RightArmHandshakeController: is_handshaking is True but handshake_start_time is None. Defaulting to nominal.")
                return pos 

            startup_elapsed_time = time_in_seconds - self.handshake_start_time
            if self.is_starting_up:
                if startup_elapsed_time >= self.handshake_startup_duration:
                    current_transition_factor = 1.0 # Fully at handshake pose
                    self.is_starting_up = False
                    self.logger.info(f"Right arm handshake startup completed at {time_in_seconds:.2f}s.")
                else:
                    # Factor from 0 (nominal pose) to 1 (handshake pose)
                    current_transition_factor = startup_elapsed_time / self.handshake_startup_duration
            else: # Normal handshaking (startup completed)
                current_transition_factor = 1.0
            
            current_transition_factor = max(0.0, min(1.0, current_transition_factor))

            # Interpolate from nominal pose to target handshake pose
            final_target_r_shld_y = nominal_r_shld_y * (1.0 - current_transition_factor) + target_r_shld_y * current_transition_factor
            final_target_r_shld_x = nominal_r_shld_x * (1.0 - current_transition_factor) + target_r_shld_x * current_transition_factor
            final_target_r_shld_z = nominal_r_shld_z * (1.0 - current_transition_factor) + target_r_shld_z * current_transition_factor
            final_target_r_elb_y = nominal_r_elb_y * (1.0 - current_transition_factor) + target_r_elb_y * current_transition_factor
            
            # Store current calculated target for smooth shutdown if interrupted
            self.last_calculated_handshake_pos[0] = final_target_r_shld_y
            self.last_calculated_handshake_pos[1] = final_target_r_shld_x
            self.last_calculated_handshake_pos[2] = final_target_r_shld_z
            self.last_calculated_handshake_pos[3] = final_target_r_elb_y
        
        else: # Should not happen if logic is correct, but as a fallback
            return pos

        pos[self.R_SHLD_Y_IDX] = final_target_r_shld_y
        pos[self.R_SHLD_X_IDX] = final_target_r_shld_x
        pos[self.R_SHLD_Z_IDX] = final_target_r_shld_z
        pos[self.R_ELB_Y_IDX] = final_target_r_elb_y

        if loop_count_for_log % 100 == 0:
            status_str = "Idle"
            if self.is_starting_up: status_str = "StartingUp"
            elif self.is_shutting_down: status_str = "ShuttingDown"
            elif self.is_handshaking: status_str = "HandshakingActive"

            self.logger.info(f"RightArmCtrl Debug @ {time_in_seconds:.2f}s (Factor: {current_transition_factor:.3f}, Status: {status_str}):")
            self.logger.info(f"  RShY_final: {final_target_r_shld_y:.3f}, RShX_final: {final_target_r_shld_x:.3f}, RShZ_final: {final_target_r_shld_z:.3f}, RElbY_final: {final_target_r_elb_y:.3f}")
            if self.is_shutting_down:
                 self.logger.info(f"    ShuttingDown: Target=({target_r_shld_y:.2f},{target_r_shld_x:.2f},{target_r_shld_z:.2f},{target_r_elb_y:.2f}), Nominal=({nominal_r_shld_y:.2f},{nominal_r_shld_x:.2f},{nominal_r_shld_z:.2f},{nominal_r_elb_y:.2f})")

        return pos
