
import collections
import numpy as np
import onnxruntime as ort
from bxi_example_py_elf3.utils.tfs import get_gravity_orientation

class TumbleRecoverPolicy:
    """host 23dof跌到恢复动作策略管理类,补偿手部腰部关节动作"""
    
    def __init__(self, model_onnx_path: str):
        """
        初始化策略
        
        Args:
            model_onnx_path: ONNX模型文件路径
            
        Usage:
            ##1.初始化模型
            self.host_policy = TumbleRecoverPolicy("path/to/model.onnx")
                
            ##2.推理动作
            self.target_dof_pos = self.host_policy.inference_step(q, dq, quat, omega)
        """
        
        self.model_onnx_path = model_onnx_path
           
        self.action_scale = 0.3

        self.ang_vel_scale = 0.25
        
        self.dof_pos_scale = 1.0
        
        self.dof_vel_scale = 0.05

        # Number of actions and observations
        self.num_actions = 23
        
        self.num_obs = 456  # 76 * 6 (observation dimension * history length)

        self.obs_history_len = 6
        
        self.single_obs_dim = 3 + 3 + self.num_actions*3 +1 #76
        
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
        # print(self.input_info)
        # print(self.output_info)
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

        print("host model init finished!!!")

    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega):
        #29dof转换为23dof,去掉waist x y 以及wrist y z
        qj =  np.zeros(self.num_actions, dtype=np.float32)
        qj[:1] = q[2:3].copy()
        qj[1:13] = q[3:15].copy()
        qj[13:18] = q[15:20].copy()
        qj[18:23] = q[22:27].copy()
        
        qj_23=qj.copy()
        
        dqj =  np.zeros(self.num_actions, dtype=np.float32)
        dqj[:1] = dq[2:3].copy()
        dqj[1:13] = dq[3:15].copy()
        dqj[13:18] = dq[15:20].copy()
        dqj[18:23] = dq[22:27].copy()
        
         # Update observation
        self.obs_tensor = self.compute_observation(qj, dqj, quat, omega)        
        np.copyto(self.input_buffer, self.obs_tensor)  # 比直接赋值更安全
        self.action = self.session.run(["output"], {"input": self.obs_tensor})[0][0]
    
        self.last_action = self.action.copy()

        self.target_dof_pos = self.action * self.action_scale
        
        #还原到机器人关节顺序
        target_dof_pos_10=self.target_dof_pos[:10].copy()
        target_dof_pos_13=self.target_dof_pos[10:23].copy()
        self.target_dof_pos[:13]=target_dof_pos_13.copy()
        self.target_dof_pos[13:23]=target_dof_pos_10.copy()
        
        self.target_dof_pos = self.target_dof_pos + qj_23.copy()
        
        self.target_dof_29_pos =  np.zeros(len(q), dtype=np.float32)
        self.target_dof_29_pos[:2] = np.array([0,0])  
        # self.target_dof_29_pos[:2] = q[:2].copy()  
        self.target_dof_29_pos[2:3] = self.target_dof_pos[:1].copy()
        self.target_dof_29_pos[3:20] = self.target_dof_pos[1:18].copy()
        self.target_dof_29_pos[20:22] = np.array([0,0])
        # self.target_dof_29_pos[20:22] = q[20:22].copy() 
        self.target_dof_29_pos[22:27] = self.target_dof_pos[18:23].copy()
        self.target_dof_29_pos[27:29] = np.array([0,0])
        # self.target_dof_29_pos[27:29] = q[27:29].copy() 
         
        # 极简推理（比原版快5-15%）
        # return self.target_dof_pos
        return self.target_dof_29_pos

    # 创建观测输入   
    def compute_observation(self,qj, dqj, quat, omega):
        """Compute the observation vector from current state"""
        
        #手部十个关节提前
        qj_13=qj[:13].copy()
        # print(len(qj_13))
        qj_10=qj[13:23].copy()
        # print(len(qj_10))
        qj[:10]=qj_10[:10].copy()
        qj[10:23]=qj_13[:13].copy()
        
        # dqj = dq.copy()
        dqj_13=dqj[:13].copy()
        dqj_10=dqj[13:23].copy()
        dqj[:10]=dqj_10.copy()
        dqj[10:23]=dqj_13.copy()
        
        gravity_orientation = get_gravity_orientation(quat)
        
        # print(single_obs_dim)#76
        
        # Create single observation
        single_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        single_obs[0:3] = omega * self.ang_vel_scale#0.25 #3
        single_obs[3:6] = gravity_orientation             #3
        single_obs[6:6+self.num_actions] = qj * self.dof_pos_scale    #23
        single_obs[6+self.num_actions:6+2*self.num_actions] = dqj * self.dof_vel_scale#0.05 #23
        single_obs[6+2*self.num_actions:6+3*self.num_actions] = self.last_action  # Assuming action has at least 15 elements
        single_obs[6+3*self.num_actions:7+3*self.num_actions] = self.action_scale  #np.array([height_cmd])     #1
        
        self.obs_history.append(single_obs)
        
        # Construct full observation with history
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * self.single_obs_dim
            end_idx = start_idx + self.single_obs_dim
            self.obs[start_idx:end_idx] = hist_obs
            
        return np.expand_dims(self.obs, axis=0)

class TumbleRecoverPolicy_29_Dof:
    """host 29dof跌到恢复动作策略管理类"""
    
    def __init__(self, model_onnx_path: str):
        """
        初始化策略
        
        Args:
            model_onnx_path: ONNX模型文件路径
            
        Usage:
            ##1.初始化模型
            self.host_policy = TumbleRecoverPolicy("path/to/model.onnx")
                
            ##2.推理动作
            self.target_dof_pos = self.host_policy.inference_step(q, dq, quat, omega)
        """
        
        self.model_onnx_path = model_onnx_path
           
        self.action_scale = 0.3

        self.ang_vel_scale = 0.25
        
        self.dof_pos_scale = 1.0
        
        self.dof_vel_scale = 0.05

        # Number of actions and observations
        self.num_actions = 29
        
        self.num_obs = 564  # 94 * 6 (observation dimension * history length)

        self.obs_history_len = 6
        
        self.single_obs_dim = 3 + 3 + self.num_actions*3 +1 #94
        
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

        print("model init finished!!!")

    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega):
        qj =  q.copy()
        dqj =  dq.copy()
         # Update observation
        self.obs_tensor = self.compute_observation(qj, dqj, quat, omega)        
        np.copyto(self.input_buffer, self.obs_tensor)  # 比直接赋值更安全
        self.action = self.session.run(["output"], {"input": self.obs_tensor})[0][0]
    
        self.last_action = self.action.copy()

        self.target_dof_pos = self.action * self.action_scale
        
        #还原到机器人关节顺序
        target_dof_pos_14=self.target_dof_pos[:14].copy()
        target_dof_pos_15=self.target_dof_pos[14:29].copy()
        self.target_dof_pos[:14]=target_dof_pos_15.copy()
        self.target_dof_pos[14:29]=target_dof_pos_14.copy()
        
        self.target_dof_pos = self.target_dof_pos + dqj.copy()
        
        # 极简推理（比原版快5-15%）
        return self.target_dof_pos

    # 创建观测输入   
    def compute_observation(self,qj, dqj, quat, omega):
        """Compute the observation vector from current state"""
        #关节顺序调整
        qj_15=qj[:15].copy()
        # print(len(qj_13))
        qj_14=qj[15:29].copy()
        # print(len(qj_10))
        qj[:14]=qj_14[:14].copy()
        qj[14:29]=qj_15[14:29].copy()
        
        # dqj = dq.copy()
        dqj_15=dqj[:15].copy()
        dqj_14=dqj[15:29].copy()
        dqj[:14]=dqj_14[:14].copy()
        dqj[14:29]=dqj_15[14:29].copy()

        gravity_orientation = get_gravity_orientation(quat)
        
        # print(single_obs_dim)#94
        
        # Create single observation
        single_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        single_obs[0:3] = omega * self.ang_vel_scale#0.25 #3
        single_obs[3:6] = gravity_orientation             #3
        single_obs[6:6+self.num_actions] = qj * self.dof_pos_scale    #29
        single_obs[6+self.num_actions:6+2*self.num_actions] = dqj * self.dof_vel_scale#0.05 #29
        single_obs[6+2*self.num_actions:6+3*self.num_actions] = self.last_action  # Assuming action has at least 15 elements #29
        single_obs[6+3*self.num_actions:7+3*self.num_actions] = self.action_scale  #np.array([height_cmd])     #1
        
        self.obs_history.append(single_obs)
        
        # Construct full observation with history
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * self.single_obs_dim
            end_idx = start_idx + self.single_obs_dim
            self.obs[start_idx:end_idx] = hist_obs
            
        return np.expand_dims(self.obs, axis=0)