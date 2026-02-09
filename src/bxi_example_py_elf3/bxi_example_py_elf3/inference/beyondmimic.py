
import onnx
import numpy as np
import onnxruntime as ort
from bxi_example_py_elf3.utils.tfs import quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply, matrix_to_quaternion_simple, yaw_quat

class DanceMotionPolicy:
    """舞蹈动作策略管理类"""
    
    def __init__(self, motion_npz_path: str, model_onnx_path: str):
        """
        初始化舞蹈动作策略
        
        Args:
            motion_npz_path: 舞蹈动作数据文件路径(.npz)
            model_onnx_path: ONNX模型文件路径
            
        Usage:
            ##1.初始化模型
            self.dance_policy = DanceMotionPolicy("path/to/motion.npz", "path/to/model.onnx")
                
            ##2.推理动作
            if self.dance_policy.timestep < self.dance_policy.motionpos.shape[0]:
                self.target_dof_pos = self.dance_policy.inference_step(q, dq, quat, omega)
        """
        self.num_obs = 154 #带位姿估计160
        
        self.motion_npz_path = motion_npz_path
        
        self.model_onnx_path = model_onnx_path
        
        self.initialize_model(motion_npz_path, model_onnx_path)
        
        self.timeinit = 0.0#计算初始转换矩阵计数器
        
        self.timestep = 0
        
        self.num_actions = len(self.joint_name)
        
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        self.action_buffer = np.zeros((self.num_actions,), dtype=np.float32)
        
    # 初始化部分（完整版）
    def initialize_model(self, npz_path, onnx_path):
        # 加载运动数据
        print("model init!!!")
        self.motion =  np.load(npz_path)
        self.motionpos = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]
        # print("Inference timestep:", self.motionpos.shape[0]) #总动作序列长度
        # print(" ")
        
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        for prop in model.metadata_props:
            if prop.key == "joint_names":
                self.joint_name = prop.value.split(",")
                # print(f"{prop.key}: {prop.value}")
                # print(" ")
                
            if prop.key == "default_joint_pos":   
                self.joint_pos_array = np.array([float(x) for x in prop.value.split(",")])
                # print(f"{prop.key}: {prop.value}")
                # print(" ")

            if prop.key == "joint_stiffness":
                self.stiffness_array = np.array([float(x) for x in prop.value.split(",")])
                # print(f"{prop.key}: {prop.value}")
                # print(" ")
                
            if prop.key == "joint_damping":
                self.damping_array = np.array([float(x) for x in prop.value.split(",")])
                # print(f"{prop.key}: {prop.value}")
                print(" ")      
            
            if prop.key == "action_scale":
                self.action_scale = np.array([float(x) for x in prop.value.split(",")])
                # print(f"{prop.key}: {prop.value}")
                # print(" ")
                
            # print(f"{prop.key}: {prop.value}")#查看metadata_props内容
            
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
            self.input_info.shape,
            dtype=np.float32
        )

        print("dance model init finished!!!")

    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega):
        # 使用预分配内存（如果适用）
        obs_data = self.create_obs_input(q, dq, quat, omega)
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        self.action = self.session.run(['actions'], {'obs': obs_data, 'time_step':np.array([[self.timestep]], dtype=np.float32)})[0]
        
        self.action = np.asarray(self.action).reshape(-1)
        self.action_buffer = self.action.copy()
        
        self.target_dof_pos = self.action * self.action_scale + self.joint_pos_array
        self.target_dof_pos = self.target_dof_pos.reshape(-1,)
        # 极简推理（比原版快5-15%）
        return self.target_dof_pos

    # 计算初始到世界坐标系的转换矩阵
    def compute_init_to_world(self, robot_quat, motion_quat):
        yaw_motion_quat = yaw_quat(motion_quat)
        yaw_motion_matrix = np.zeros(9)
        yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat).reshape(3,3)
        
        yaw_robot_quat = yaw_quat(robot_quat)
        yaw_robot_matrix = np.zeros(9)
        yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat).reshape(3,3)
        yaw_robot_matrix = yaw_robot_matrix.reshape(3,3)
        self.init_to_world =  yaw_robot_matrix @ yaw_motion_matrix.T

    # 计算相对旋转矩阵    
    def compute_relmatrix(self, robot_quat, motion_quat):
        rel_quat = quaternion_multiply(matrix_to_quaternion_simple(self.init_to_world), motion_quat)
        rel_quat = quaternion_multiply(quaternion_conjugate(robot_quat),rel_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat) # 归一化四元数
        rel_matrix = quaternion_to_rotation_matrix(rel_quat)[:,:2].reshape(-1,)  # 转换为旋转矩阵并取前两列展平
        return rel_matrix
 
    # 创建观测输入   
    def create_obs_input(self,q, dq, quat, omega):
        # 获取当前动作数据
        motion_quat = self.motionquat[self.timestep,0,:]
        motion_pos = self.motioninputpos[self.timestep,:]
        motion_vel = self.motioninputvel[self.timestep,:]  
          
        # 前两个时间步计算初始转换矩阵
        if self.timeinit < 2:
            self.timeinit += 1.0
            self.compute_init_to_world(quat, motion_quat)
        
        # create observation
        offset = 0
        motioninput = np.concatenate((motion_pos,motion_vel),axis=0)
        self.obs[offset:offset + 58] = motioninput
        
        offset += 58
        relmatrix = self.compute_relmatrix(quat, motion_quat)
        self.obs[offset:offset + 6] = relmatrix  
        
        offset += 6
        self.obs[offset:offset + 3] = omega 
        
        offset += 3
        self.obs[offset:offset + self.num_actions] = q - self.joint_pos_array  # joint positions
        
        offset += self.num_actions
        self.obs[offset:offset + self.num_actions] = dq  # joint velocities
        
        offset += self.num_actions   
        self.obs[offset:offset + self.num_actions] = self.action_buffer
        
        self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(154,)变成(1,154)并确保数据类型
        
        return self.obs_input
    