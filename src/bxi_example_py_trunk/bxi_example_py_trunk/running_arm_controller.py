import math
import numpy as np

class RunningArmController:
    def __init__(self, logger, joint_nominal_pos_ref,
                 arm_startup_duration=1.5, arm_shutdown_duration=1.5,
                 arm_amplitude_y=0.5, arm_amplitude_z=0.2, elbow_coeff=0.3):
        self.logger = logger
        self.joint_nominal_pos_ref = joint_nominal_pos_ref
        self.arm_startup_duration = arm_startup_duration
        self.arm_shutdown_duration = arm_shutdown_duration
        self.arm_amplitude_y = arm_amplitude_y  # Shoulder Y-axis swing amplitude
        self.arm_amplitude_z = arm_amplitude_z  # Shoulder Z-axis swing amplitude (optional, for more natural motion)
        self.elbow_coeff = elbow_coeff # Coefficient for elbow movement based on shoulder Y

        self.motion_start_time = None
        self.motion_stop_time = None
        self.is_running_motion = False
        self.is_starting_up = False
        self.is_shutting_down = False

        # Joint indices from bxi_example.py's joint_name tuple
        # Left Arm
        self.L_SHLD_Y_IDX = 15  # l_shld_y_joint
        self.L_SHLD_X_IDX = 16  # l_shld_x_joint (might not be used for simple running)
        self.L_SHLD_Z_IDX = 17  # l_shld_z_joint
        self.L_ELB_Y_IDX = 18   # l_elb_y_joint
        # Right Arm
        self.R_SHLD_Y_IDX = 20  # r_shld_y_joint
        self.R_SHLD_X_IDX = 21  # r_shld_x_joint (might not be used)
        self.R_SHLD_Z_IDX = 22  # r_shld_z_joint
        self.R_ELB_Y_IDX = 23   # r_elb_y_joint

        # Store nominal positions for convenience
        self.nominal_l_shld_y = self.joint_nominal_pos_ref[self.L_SHLD_Y_IDX]
        self.nominal_l_shld_z = self.joint_nominal_pos_ref[self.L_SHLD_Z_IDX]
        self.nominal_l_elb_y = self.joint_nominal_pos_ref[self.L_ELB_Y_IDX]
        self.nominal_r_shld_y = self.joint_nominal_pos_ref[self.R_SHLD_Y_IDX]
        self.nominal_r_shld_z = self.joint_nominal_pos_ref[self.R_SHLD_Z_IDX]
        self.nominal_r_elb_y = self.joint_nominal_pos_ref[self.R_ELB_Y_IDX]

        # To store last calculated positions for smooth shutdown
        self.last_calculated_pos = {
            "l_shld_y": self.nominal_l_shld_y, "l_shld_z": self.nominal_l_shld_z, "l_elb_y": self.nominal_l_elb_y,
            "r_shld_y": self.nominal_r_shld_y, "r_shld_z": self.nominal_r_shld_z, "r_elb_y": self.nominal_r_elb_y,
        }

    def start_running_motion(self, current_sim_time):
        if not self.is_running_motion or self.is_shutting_down:
            self.is_running_motion = True
            self.is_starting_up = True
            self.is_shutting_down = False
            self.motion_start_time = current_sim_time
            self.motion_stop_time = None
            self.logger.info(f"RunningArmController: Motion initiated at {current_sim_time:.2f}s.")

    def stop_running_motion(self, current_sim_time):
        if self.is_running_motion and not self.is_shutting_down:
            self.is_shutting_down = True
            self.is_starting_up = False
            self.motion_stop_time = current_sim_time
            self.logger.info(f"RunningArmController: Motion shutdown initiated at {current_sim_time:.2f}s.")

    def calculate_running_arm_motion(self, base_pos, time_in_seconds, leg_phase_left_signal, leg_phase_right_signal, loop_count_for_log=0):
        pos = base_pos.copy()

        if not self.is_running_motion and not self.is_shutting_down:
            return pos

        current_motion_amplitude_factor = 0.0

        if self.is_shutting_down:
            if self.motion_stop_time is None:
                self.logger.warn("RunningArmController: is_shutting_down but motion_stop_time is None. Forcing stop.")
                self.is_running_motion = False
                self.is_shutting_down = False
                return pos
            
            shutdown_elapsed_time = time_in_seconds - self.motion_stop_time
            if shutdown_elapsed_time >= self.arm_shutdown_duration:
                current_motion_amplitude_factor = 0.0
                self.is_running_motion = False
                self.is_shutting_down = False
                self.motion_start_time = None
                self.motion_stop_time = None
                # Reset to nominal positions on full shutdown
                pos[self.L_SHLD_Y_IDX] = self.nominal_l_shld_y
                pos[self.L_SHLD_Z_IDX] = self.nominal_l_shld_z
                pos[self.L_ELB_Y_IDX] = self.nominal_l_elb_y
                pos[self.R_SHLD_Y_IDX] = self.nominal_r_shld_y
                pos[self.R_SHLD_Z_IDX] = self.nominal_r_shld_z
                pos[self.R_ELB_Y_IDX] = self.nominal_r_elb_y
                self.logger.info(f"RunningArmController: Motion shutdown completed at {time_in_seconds:.2f}s.")
                return pos
            else:
                current_motion_amplitude_factor = 1.0 - (shutdown_elapsed_time / self.arm_shutdown_duration)
        
        elif self.is_running_motion: # Includes starting_up and normal running
            if self.motion_start_time is None:
                self.logger.warn("RunningArmController: is_running_motion but motion_start_time is None. Defaulting to no motion.")
                return pos

            startup_elapsed_time = time_in_seconds - self.motion_start_time
            if self.is_starting_up:
                if startup_elapsed_time >= self.arm_startup_duration:
                    current_motion_amplitude_factor = 1.0
                    self.is_starting_up = False
                    self.logger.info(f"RunningArmController: Motion startup completed at {time_in_seconds:.2f}s.")
                else:
                    current_motion_amplitude_factor = startup_elapsed_time / self.arm_startup_duration
            else: # Normal running
                current_motion_amplitude_factor = 1.0
        
        current_motion_amplitude_factor = max(0.0, min(1.0, current_motion_amplitude_factor))

        # --- Calculate target arm joint positions based on leg phases ---
        # "迈左腿时手臂向右挥，迈右腿时手臂向左挥"
        # Left leg forward -> Right arm forward, Left arm backward
        # Right leg forward -> Left arm forward, Right arm backward
        # Assuming leg_phase_signal is positive for forward swing, negative for backward.

        # Left Arm (synchronized with right leg's phase)
        # If right leg is forward (leg_phase_right_signal > 0), left arm swings forward.
        l_shld_y_swing = self.arm_amplitude_y * leg_phase_right_signal
        # Optional: Z-axis swing for shoulder (e.g., slightly inward as arm goes forward)
        # l_shld_z_swing = -self.arm_amplitude_z * leg_phase_right_signal 
        l_shld_z_swing = 0 # Keep it simple for now

        target_l_shld_y = self.nominal_l_shld_y + l_shld_y_swing * current_motion_amplitude_factor
        target_l_shld_z = self.nominal_l_shld_z + l_shld_z_swing * current_motion_amplitude_factor
        target_l_elb_y = self.nominal_l_elb_y - self.elbow_coeff * abs(l_shld_y_swing) * current_motion_amplitude_factor # Elbow flexes as shoulder swings

        # Right Arm (synchronized with left leg's phase)
        # If left leg is forward (leg_phase_left_signal > 0), right arm swings forward.
        r_shld_y_swing = self.arm_amplitude_y * leg_phase_left_signal
        # r_shld_z_swing = self.arm_amplitude_z * leg_phase_left_signal # Z-swing for right arm (e.g. slightly inward)
        r_shld_z_swing = 0 # Keep it simple for now

        target_r_shld_y = self.nominal_r_shld_y + r_shld_y_swing * current_motion_amplitude_factor
        target_r_shld_z = self.nominal_r_shld_z + r_shld_z_swing * current_motion_amplitude_factor
        target_r_elb_y = self.nominal_r_elb_y - self.elbow_coeff * abs(r_shld_y_swing) * current_motion_amplitude_factor


        if self.is_shutting_down:
            shutdown_progress = (time_in_seconds - self.motion_stop_time) / self.arm_shutdown_duration
            shutdown_progress = max(0.0, min(1.0, shutdown_progress))

            pos[self.L_SHLD_Y_IDX] = self.last_calculated_pos["l_shld_y"] * (1.0 - shutdown_progress) + self.nominal_l_shld_y * shutdown_progress
            pos[self.L_SHLD_Z_IDX] = self.last_calculated_pos["l_shld_z"] * (1.0 - shutdown_progress) + self.nominal_l_shld_z * shutdown_progress
            pos[self.L_ELB_Y_IDX] = self.last_calculated_pos["l_elb_y"] * (1.0 - shutdown_progress) + self.nominal_l_elb_y * shutdown_progress
            pos[self.R_SHLD_Y_IDX] = self.last_calculated_pos["r_shld_y"] * (1.0 - shutdown_progress) + self.nominal_r_shld_y * shutdown_progress
            pos[self.R_SHLD_Z_IDX] = self.last_calculated_pos["r_shld_z"] * (1.0 - shutdown_progress) + self.nominal_r_shld_z * shutdown_progress
            pos[self.R_ELB_Y_IDX] = self.last_calculated_pos["r_elb_y"] * (1.0 - shutdown_progress) + self.nominal_r_elb_y * shutdown_progress
        else: # Startup or active running
            pos[self.L_SHLD_Y_IDX] = target_l_shld_y
            pos[self.L_SHLD_Z_IDX] = target_l_shld_z
            pos[self.L_ELB_Y_IDX] = target_l_elb_y
            pos[self.R_SHLD_Y_IDX] = target_r_shld_y
            pos[self.R_SHLD_Z_IDX] = target_r_shld_z
            pos[self.R_ELB_Y_IDX] = target_r_elb_y

            # Store for smooth shutdown
            self.last_calculated_pos["l_shld_y"] = target_l_shld_y
            self.last_calculated_pos["l_shld_z"] = target_l_shld_z
            self.last_calculated_pos["l_elb_y"] = target_l_elb_y
            self.last_calculated_pos["r_shld_y"] = target_r_shld_y
            self.last_calculated_pos["r_shld_z"] = target_r_shld_z
            self.last_calculated_pos["r_elb_y"] = target_r_elb_y

        if loop_count_for_log % 100 == 0:
            status_str = "Idle"
            if self.is_starting_up: status_str = "StartingUp"
            elif self.is_shutting_down: status_str = "ShuttingDown"
            elif self.is_running_motion: status_str = "RunningActive"
            
            self.logger.info(f"RunningArmCtrl @ {time_in_seconds:.2f}s (Factor: {current_motion_amplitude_factor:.3f}, Status: {status_str}):")
            self.logger.info(f"  LegPhL: {leg_phase_left_signal:.3f}, LegPhR: {leg_phase_right_signal:.3f}")
            self.logger.info(f"  LShY: {pos[self.L_SHLD_Y_IDX]:.3f}, LShZ: {pos[self.L_SHLD_Z_IDX]:.3f}, LElbY: {pos[self.L_ELB_Y_IDX]:.3f}")
            self.logger.info(f"  RShY: {pos[self.R_SHLD_Y_IDX]:.3f}, RShZ: {pos[self.R_SHLD_Z_IDX]:.3f}, RElbY: {pos[self.R_ELB_Y_IDX]:.3f}")

        return pos

    # Helper properties for bxi_example.py to check status
    @property
    def is_active(self): # Equivalent to old is_waving for general activity check
        return self.is_running_motion

    @property
    def is_active_or_shutting_down(self): # For checking if calculation is needed
        return self.is_running_motion or self.is_shutting_down
