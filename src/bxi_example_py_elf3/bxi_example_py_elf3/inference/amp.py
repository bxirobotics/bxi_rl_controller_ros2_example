
import collections
import numpy as np
import onnxruntime as ort
from bxi_example_py_elf3.utils.tfs import get_gravity_orientation

class HumanoidGaitPolicyLiteIsaaclab:
    """不带步态输入的AMP行走动作策略管理类"""
    
    def __init__(self, model_onnx_path: str):
        """
        初始化策略
        
        Args:
            model_onnx_path: ONNX模型文件路径
            
        Usage:
            ##1.初始化模型
            self.amp_policy = HumanoidGaitPolicyLite("path/to/model.onnx")
                
            ##2.推理动作
            self.target_dof_pos = self.amp_policy.inference_step(q, dq, quat, omega, cmd_vel)
        """
        
        self.model_onnx_path = model_onnx_path

        self.action_scale = np.array([
            0.231, 0.231, 0.231,
            0.231, 0.231, 0.154,
            0.373, 0.373, 0.213,
            0.231, 0.231, 0.213,
            0.213, 0.373, 0.373,
            0.213, 0.213, 0.373, 
            0.373, 0.231, 
            0.231, 0.373, 
            0.373, 0.213, 0.213, 
            0.373, 0.373, 
            0.23, 0.23
        ])
        
        self.kps = np.array([     # 奔跑的关节kp，和joint_name顺序一一对应
            108.448,162.672,176.421,
            176.421,176.421,54.224,176.421,33.493,21.771,
            176.421,176.421,54.224,176.421,33.493,21.771,
            54.224,54.224,16.747,54.224, 16.747,16.747,16.747,
            54.224,54.224,16.747,54.224, 16.747,16.747,16.747,
            ], 
            dtype=np.float32)

        self.kds = np.array([  # 奔跑的关节kd，和joint_name顺序一一对应
            6.904,10.356,11.231,
            11.231,11.231,3.452,11.231,2.132,1.386,
            11.231,11.231,3.452,11.231,2.132,1.386,
            3.452,3.452,1.066,3.452, 1.066,1.066,1.066,
            3.452,3.452,1.066,3.452, 1.066,1.066,1.066,
            ], 
            dtype=np.float32)
        
        #双臂自然下垂姿势
        self.default_dof_pos = np.array([   # 指定的固定关节角度
            0.0, 0.0, 0.0,
            -0.3,0.0,0.0,0.6,-0.3,0.0,
            -0.3,0.0,0.0,0.6,-0.3,0.0,
            0.2,0.2,0.0,0.6, 0.0,0.0,0.0,     
            0.2,-0.2,0.0,0.6, 0.0,0.0,0.0],    
            dtype=np.float32)

        self.mujoco_to_isaac_idx = [
            15,    # 'l_shoulder_y_joint', 0
            22,    #  'r_shoulder_y_joint', 1
            0,    #  'waist_y_joint', 2
            16,    #  'l_shoulder_x_joint',3 
            23,    #  'r_shoulder_x_joint', 4
            1,    #  'waist_x_joint', 5
            17,    #  'l_shoulder_z_joint',6 
            24,    #  'r_shoulder_z_joint', 7
            2,    #  'waist_z_joint', 8
            18,    #  'l_elbow_y_joint',9 
            25,    #  'r_elbow_y_joint', 10
            3,    #  'l_hip_y_joint', 11
            9,    #  'r_hip_y_joint', 12
            19,    #  'l_wrist_x_joint',13 
            26,    #  'r_wrist_x_joint', 14
            4,    #  'l_hip_x_joint', 15
            10,   #  'r_hip_x_joint', 16
            20,    #  'l_wrist_y_joint', 17
            27,    #  'r_wrist_y_joint', 18
            5,    #  'l_hip_z_joint', 19
            11,   #  'r_hip_z_joint', 20
            21,    #  'l_wrist_z_joint', 21 
            28,    #  'r_wrist_z_joint', 22
            6,    #  'l_knee_y_joint', 23
            12,   #  'r_knee_y_joint', 24
            7,    #  'l_ankle_y_joint', 25
            13,   #  'r_ankle_y_joint', 26
            8,    #  'l_ankle_x_joint', 27
            14,   #  'r_ankle_x_joint',28
        ]
        
        self.isaac_to_mujoco_idx = [
            2,    # "waist_y_joint",
            5,    # "waist_x_joint",
            8,    # "waist_z_joint",
                
            11,    # "l_hip_y_joint",   # 左腿_髋关节_z轴
            15,    # "l_hip_x_joint",   # 左腿_髋关节_x轴
            19,    # "l_hip_z_joint",   # 左腿_髋关节_y轴
            23,    # "l_knee_y_joint",   # 左腿_膝关节_y轴
            25,    # "l_ankle_y_joint",   # 左腿_踝关节_y轴
            27,    # "l_ankle_x_joint",   # 左腿_踝关节_x轴

            12,    # "r_hip_y_joint",   # 右腿_髋关节_z轴    
            16,    # "r_hip_x_joint",   # 右腿_髋关节_x轴
            20,    # "r_hip_z_joint",   # 右腿_髋关节_y轴
            24,    # "r_knee_y_joint",   # 右腿_膝关节_y轴
            26,    # "r_ankle_y_joint",   # 右腿_踝关节_y轴
            28,    # "r_ankle_x_joint",   # 右腿_踝关节_x轴
            0,    # "l_shoulder_y_joint",   # 左臂_肩关节_y轴
            3,    # "l_shoulder_x_joint",   # 左臂_肩关节_x轴
            6,    # "l_shoulder_z_joint",   # 左臂_肩关节_z轴
            9,    # "l_elbow_y_joint",   # 左臂_肘关节_y轴
            13,    # "l_wrist_x_joint",
            17,    # "l_wrist_y_joint",
            21,    # "l_wrist_z_joint",
                
            1,    # "r_shoulder_y_joint",   # 右臂_肩关节_y轴   
            4,    # "r_shoulder_x_joint",   # 右臂_肩关节_x轴
            7,    # "r_shoulder_z_joint",   # 右臂_肩关节_z轴
            10,    # "r_elbow_y_joint",    # 右臂_肘关节_y轴
            14,    # "r_wrist_x_joint",
            18,    # "r_wrist_y_joint",
            22,    # "r_wrist_z_joint",
        ]
        
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])

        # Gait parameters
        self.dt: float = 0.005  #控制周期
        self.gait_air_ratio_l: float = 0.38  #步态空中比率l
        self.gait_air_ratio_r: float = 0.38  #步态空中比率r
        self.gait_phase_offset_l: float = 0.38  #步态相位偏移l
        self.gait_phase_offset_r: float = 0.88  #步态相位偏移r
        self.gait_cycle: float = 0.85    #步态周期（秒）
        
        # Number of actions and observations.
        # 960/1020D policies are full-body models. 540D policies are no-arm
        # models and only command waist + legs; applause overlays the arms.
        self.robot_dof_num = 29
        self.policy_action_dim = 29
        self.controlled_action_dim = 15
        self.num_actions = self.policy_action_dim
        self.obs_action_dim = self.policy_action_dim

        self.full_body_mujoco_idx = np.arange(self.policy_action_dim, dtype=np.int64)
        self.full_body_isaac_idx = np.asarray(self.isaac_to_mujoco_idx, dtype=np.int64)
        self.controlled_mujoco_idx = np.arange(self.controlled_action_dim, dtype=np.int64)
        self.controlled_isaac_idx = np.asarray(
            [self.isaac_to_mujoco_idx[i] for i in self.controlled_mujoco_idx],
            dtype=np.int64,
        )
        self.isaac_order_mujoco_idx = np.asarray(self.mujoco_to_isaac_idx, dtype=np.int64)
        self.noarm_policy_isaac_idx = np.asarray(
            [i for i, mujoco_idx in enumerate(self.isaac_order_mujoco_idx) if mujoco_idx < self.controlled_action_dim],
            dtype=np.int64,
        )
        self.noarm_policy_mujoco_idx = self.isaac_order_mujoco_idx[self.noarm_policy_isaac_idx]
        self.output_mujoco_idx = self.full_body_mujoco_idx
        self.last_action_isaac_idx = self.full_body_isaac_idx
        self.controlled_action_scale = self.action_scale[self.full_body_isaac_idx]
        
        # 【动态】以下将在initialize_model中根据实际模型输入维度调整
        self.num_obs = 960  # 默认值，将被动态调整
        self.obs_history_len = 10
        self.single_obs_dim = 3 + 3 + 3 + self.num_actions*3  # 默认96维，将被动态调整
        self.extra_obs_dim = 0  # 额外观测维度（用于1020维模型），默认0
        
        self.initialize_model(self.model_onnx_path)

        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.vae_vel = np.zeros(3, dtype=np.float32)
        self.max_vel = 0.0
        
    # 初始化部分（完整版）
    def initialize_model(self, onnx_path):
        # 加载运动数据
            
        # 配置执行提供者（根据硬件选择最优后端）
        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
            sess_options=options
        )
        
        # 预存输入输出信息
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        print(self.input_info)
        print(self.output_info)
        
        # 【关键】自动检测模型输入维度，兼容960和1020
        input_shape = self.input_info.shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
            actual_input_dim = input_shape[-1]
        else:
            actual_input_dim = 960  # 默认值
        
        print(f"\n[AutoDetect] Model input dimension: {actual_input_dim}")
        
        # 根据输入维度动态调整
        if actual_input_dim == 540:
            # 540维模型：54维/步 × 10步，elf3_tang_noarm训练时使用
            # Isaac全身关节顺序删掉手臂后的15维策略顺序。
            self.single_obs_dim = 54  # 3+3+3+15+15+15
            self.obs_action_dim = self.controlled_action_dim
            # self.output_mujoco_idx = self.noarm_policy_mujoco_idx
            # self.last_action_isaac_idx = self.noarm_policy_isaac_idx
            # self.controlled_action_scale = self.action_scale[self.noarm_policy_isaac_idx]
            self.output_mujoco_idx = self.controlled_mujoco_idx
            self.last_action_isaac_idx = self.controlled_isaac_idx
            self.controlled_action_scale = self.action_scale[self.controlled_isaac_idx]
            self.extra_obs_dim = 0
            print(f"[AutoDetect] Configured for 540D input (54D per step, no-arm Isaac joint order)")
            print(f"[AutoDetect] 540D output MuJoCo indices: {self.output_mujoco_idx.tolist()}")
        elif actual_input_dim == 960:
            # 960维模型：96维/步 × 10步
            self.single_obs_dim = 96  # 3+3+3+29+29+29
            self.obs_action_dim = self.policy_action_dim
            self.output_mujoco_idx = self.full_body_mujoco_idx
            self.last_action_isaac_idx = self.full_body_isaac_idx
            self.controlled_action_scale = self.action_scale[self.full_body_isaac_idx]
            self.extra_obs_dim = 0
            print(f"[AutoDetect] Configured for 960D input (96D per step, full body)")
        elif actual_input_dim == 1020:
            # 1020维模型：102维/步 × 10步
            self.single_obs_dim = 102  # 96 + 6 extra
            self.obs_action_dim = self.policy_action_dim
            self.output_mujoco_idx = self.full_body_mujoco_idx
            self.last_action_isaac_idx = self.full_body_isaac_idx
            self.controlled_action_scale = self.action_scale[self.full_body_isaac_idx]
            self.extra_obs_dim = 6
            print(f"[AutoDetect] Configured for 1020D input (102D per step, full body, +6 extra)")
        else:
            raise ValueError(
                f"Unsupported AMP model input dimension: {actual_input_dim}. "
                "Expected 540, 960, or 1020."
            )
      
        # 更新总观测维度
        self.num_obs = self.single_obs_dim * self.obs_history_len
        print(f"[AutoDetect] Total observation dimension: {self.num_obs}\n")
        
        # 预分配输入内存（可选，适合固定输入尺寸）
        self.input_buffer = np.zeros(
            self.input_info.shape[1],
            dtype=np.float32
        )
        
        # Initialize variables
        self.action = np.zeros(self.policy_action_dim, dtype=np.float32)
        self.last_action = np.zeros(self.policy_action_dim, dtype=np.float32)
        self.last_controlled_action = np.zeros(self.controlled_action_dim, dtype=np.float32)
        self.target_dof_pos = self.default_dof_pos.copy()
        
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.single_obs_dim, dtype=np.float32))
        
        # Prepare full observation vector
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        if self.extra_obs_dim > 0:
            self.episode_length_buf = 0
            self.gait_phase = np.zeros(2) #步态相位
            self.gait_cycle = self.gait_cycle
            self.phase_ratio = np.array([self.gait_air_ratio_l, self.gait_air_ratio_r]) #步态空中比率[0.38,0.38]
            self.phase_offset = np.array([self.gait_phase_offset_l, self.gait_phase_offset_r]) #步态相位偏移[0.38,0.88]

        print("preparing initial observation history...")
        self.reset_observation_history(
            self.default_dof_pos,
            np.zeros_like(self.default_dof_pos),
            np.array([1.0, 0.0, 0.0, 0.0]),  # 单位四元数
            np.zeros(3), # 初始角速度
            np.array([0.0, 0.0, 0.0])  # 初始命令速度
        )
        print("AMP model init finished!!!")
    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega, cmd_vel):
        # 如果启用了额外观测（步态），先更新步态相位，使观测包含当前相位
        if self.extra_obs_dim > 0:
            # 计数并计算当前相位
            self.episode_length_buf += 1
            t = self.episode_length_buf * 0.02 / self.gait_cycle
            self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
            self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

        # Update observation (现在观测会使用最新的 gait_phase)
        self.obs_tensor = self.compute_observation(q, dq, quat, omega, cmd_vel)
        np.copyto(self.input_buffer, self.obs_tensor.reshape(-1))  # 比直接赋值更安全
        raw_action = self.session.run(
            [self.output_info.name],
            {self.input_info.name: self.obs_tensor},
        )[0][0]
        self.action = raw_action.astype(np.float32, copy=False)

        out_len = self.action.shape[0]
        output_action_dim = self.output_mujoco_idx.shape[0]
        if out_len < output_action_dim:
            raise ValueError(
                f"ONNX action dim is {out_len}, expected at least {output_action_dim}"
            )

        if out_len >= self.policy_action_dim:
            policy_action_isaac = self.action[:self.policy_action_dim]
            controlled_action = policy_action_isaac[self.last_action_isaac_idx]
            self.last_action = policy_action_isaac.copy()
            vel_start = self.policy_action_dim
        else:
            controlled_action = self.action[:output_action_dim]
            self.last_action = np.zeros(self.policy_action_dim, dtype=np.float32)
            self.last_action[self.last_action_isaac_idx] = controlled_action
            vel_start = output_action_dim
        self.last_controlled_action = controlled_action.copy()

        self.vae_vel = np.zeros(3, dtype=np.float32)
        if out_len > vel_start:
            vel_len = min(3, out_len - vel_start)
            self.vae_vel[:vel_len] = self.action[vel_start:vel_start + vel_len]

        self.target_dof_pos = self.default_dof_pos.copy()
        self.target_dof_pos[self.output_mujoco_idx] += (
            controlled_action * self.controlled_action_scale
        )
        self.target_dof_pos = self.target_dof_pos.astype(np.float32)

        # 极简推理（比原版快5-15%）
        return self.target_dof_pos, self.vae_vel

    # 创建观测输入   
    def reset_runtime_state(self, qj, dqj, quat, omega, cmd_vel):
        self.action = np.zeros_like(self.action, dtype=np.float32)
        self.last_action = np.zeros(self.policy_action_dim, dtype=np.float32)
        self.last_controlled_action = np.zeros(self.controlled_action_dim, dtype=np.float32)
        self.vae_vel = np.zeros(3, dtype=np.float32)
        self.command_vel = np.asarray(cmd_vel, dtype=np.float32)
        if self.extra_obs_dim > 0:
            self.episode_length_buf = 0
        self.reset_observation_history(qj, dqj, quat, omega, self.command_vel)

    def reset_observation_history(self, qj, dqj, quat, omega, cmd_vel):
        """Fill history with the current stable observation before control starts."""
        single_obs = self._build_single_observation(qj, dqj, quat, omega, cmd_vel)
        self.obs_history.clear()
        for _ in range(self.obs_history_len):
            self.obs_history.append(single_obs.copy())
        self._refresh_observation_buffer()

    def compute_observation(self, qj, dqj, quat, omega, cmd_vel):
        """Compute the observation vector from current state"""
        single_obs = self._build_single_observation(qj, dqj, quat, omega, cmd_vel)
        self.obs_history.append(single_obs)
        self._refresh_observation_buffer()

        return np.expand_dims(self.obs, axis=0)

    def _build_single_observation(self, qj, dqj, quat, omega, cmd_vel):
        gravity_orientation = get_gravity_orientation(quat)
        qj = np.asarray(qj, dtype=np.float32)
        dqj = np.asarray(dqj, dtype=np.float32)
        self.command_vel = np.asarray(cmd_vel, dtype=np.float32)

        if qj.shape[0] != self.robot_dof_num:
            raise ValueError(f"qj dim is {qj.shape[0]}, expected {self.robot_dof_num}")
        if dqj.shape[0] != self.robot_dof_num:
            raise ValueError(f"dqj dim is {dqj.shape[0]}, expected {self.robot_dof_num}")
        
        # Create single observation with dynamic dimensions
        single_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        
        # 【标准】omega + gravity + cmd_vel + joint_pos + joint_vel + last_action
        single_obs[0:3] = omega                                         # 3维
        single_obs[3:6] = gravity_orientation                           # 3维
        single_obs[6:9] = self.command_vel                              # 3维
        if self.obs_action_dim == self.controlled_action_dim:
            obs_q_idx = self.output_mujoco_idx
            obs_dq_idx = self.output_mujoco_idx
            obs_last_action = self.last_controlled_action
        else:
            obs_q_idx = self.mujoco_to_isaac_idx
            obs_dq_idx = self.mujoco_to_isaac_idx
            obs_last_action = self.last_action

        single_obs[9:9+self.obs_action_dim] = (qj - self.default_dof_pos)[obs_q_idx]
        single_obs[9+self.obs_action_dim:9+2*self.obs_action_dim] = dqj[obs_dq_idx]
        single_obs[9+2*self.obs_action_dim:9+3*self.obs_action_dim] = obs_last_action
        
        # 【兼容】如果需要额外维度（如1020维模型），补零
        if self.extra_obs_dim > 0:
            # 对于102维（1020维模型），在末尾补6维零
            # single_obs[96:102] = 0  (已在初始化时设为0，无需显式赋值)        
            single_obs[9+3*self.num_actions:11+3*self.num_actions] = np.sin(2 * np.pi * self.gait_phase)  # 2 #步态相位正弦值
            single_obs[11+3*self.num_actions:13+3*self.num_actions] = np.cos(2 * np.pi * self.gait_phase)  # 2 #步态相位余弦值
            single_obs[13+3*self.num_actions:15+3*self.num_actions] = self.phase_ratio  # 2 #步态空中比率
        
        return single_obs

    def _refresh_observation_buffer(self):
        # Construct full observation with history
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * self.single_obs_dim
            end_idx = start_idx + self.single_obs_dim
            self.obs[start_idx:end_idx] = hist_obs
    
