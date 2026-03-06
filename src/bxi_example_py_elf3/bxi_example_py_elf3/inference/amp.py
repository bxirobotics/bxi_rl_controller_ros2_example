
import collections
import numpy as np
import onnxruntime as ort
from bxi_example_py_elf3.utils.tfs import get_gravity_orientation

class HumanoidGaitPolicyLite:
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

        self.action_scale = np.array([+++++++++++
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
        
        # Number of actions and observations
        self.num_actions = 29
        
        self.num_obs = 960  # 96 * 10 (observation dimension * history length)
    
        self.obs_history_len = 10
        
        self.single_obs_dim = 3 + 3 + 3 + self.num_actions*3 #96
        
        self.initialize_model(self.model_onnx_path)
        
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
        # 预分配输入内存（可选，适合固定输入尺寸）
        self.input_buffer = np.zeros(
            self.input_info.shape[1],
            dtype=np.float32
        )
        
        # Initialize variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.single_obs_dim, dtype=np.float32))
        
        # Prepare full observation vector
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        # 进行一次初始推理，填充obs_history
        print("preparing initial inference to fill obs_history...")
        self.inference_step(  # 初始化一次，填充obs_history
            self.default_dof_pos,
            np.zeros_like(self.default_dof_pos),
            np.array([1.0, 0.0, 0.0, 0.0]),  # 单位四元数
            np.zeros(3), # 初始角速度
            np.array([0.0, 0.0, 0.0])  # 初始命令速度
        )
        print("AMP model init finished!!!")
    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega, cmd_vel):
         # Update observation
        self.obs_tensor = self.compute_observation(q, dq, quat, omega, cmd_vel)        
        np.copyto(self.input_buffer, self.obs_tensor)  # 比直接赋值更安全
        self.action = self.session.run(["actions"], {"obs": self.obs_tensor})[0][0]
    
        self.last_action = self.action.copy()

        self.target_dof_pos = self.action * self.action_scale
        
        self.target_dof_pos = self.target_dof_pos[self.isaac_to_mujoco_idx] + self.default_dof_pos
        
        # 极简推理（比原版快5-15%）
        return self.target_dof_pos

    # 创建观测输入   
    def compute_observation(self,qj, dqj, quat, omega, cmd_vel):
        """Compute the observation vector from current state"""
        gravity_orientation = get_gravity_orientation(quat)
        self.command_vel = cmd_vel  # Placeholder for commanded velocity
        # print(single_obs_dim)#94
        
        # Create single observation
        single_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        single_obs[0:3] = omega                                         #3
        single_obs[3:6] = gravity_orientation                           #3
        single_obs[6:9] = self.command_vel                                   #3
        single_obs[9:9+self.num_actions] = (qj - self.default_dof_pos)[self.mujoco_to_isaac_idx]                           #29
        single_obs[9+self.num_actions:9+2*self.num_actions] = dqj[self.mujoco_to_isaac_idx] #0.05 #29
        single_obs[9+2*self.num_actions:9+3*self.num_actions] = self.last_action  # Assuming action has at least 15 elements #29
        
        self.obs_history.append(single_obs)
        
        # Construct full observation with history
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * self.single_obs_dim
            end_idx = start_idx + self.single_obs_dim
            self.obs[start_idx:end_idx] = hist_obs
            
        return np.expand_dims(self.obs, axis=0)
    
