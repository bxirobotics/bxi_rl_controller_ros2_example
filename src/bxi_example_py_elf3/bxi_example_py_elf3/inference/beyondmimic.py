
import onnx
import numpy as np
import onnxruntime as ort
from bxi_example_py_elf3.utils.tfs import quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply, matrix_to_quaternion_simple, yaw_quat, get_gravity_orientation

class DanceMotionPolicyMjlab:
    """舞蹈动作策略管理类"""
    
    def __init__(self, motion_npz_path: str, model_onnx_path: str, start_frame: int = 0, fixed_pos: bool = False):
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
        
        self.fixed_pos = fixed_pos #是否固定位置，固定位置适合舞蹈动作
        
        self.timeinit = 0.0#计算初始转换矩阵计数器
        
        self.start_frame = start_frame
        
        self.end_frame = self.motionpos.shape[0] - 1
        
        self.timestep = self.start_frame
        
        self.num_actions = len(self.joint_name)
        
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        self.action_buffer = np.zeros((self.num_actions,), dtype=np.float32)
        
    # 初始化部分（完整版）
    def initialize_model(self, npz_path, onnx_path):
        # 加载运动数据
        # print("model init!!!")
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
                self.kps = self.stiffness_array.copy()
                # print(f"{prop.key}: {prop.value}")
                # print(" ")
                
            if prop.key == "joint_damping":
                self.damping_array = np.array([float(x) for x in prop.value.split(",")])
                self.kds = self.damping_array.copy()
                # print(f"{prop.key}: {prop.value}")
                # print(" ")      
            
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

        # print("BYDMIMIC model init finished!!!")
        # print("########################################################################")

    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega):
        # 使用预分配内存（如果适用）
        obs_data = self.create_obs_input(q, dq, quat, omega)
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        self.action = self.session.run(['actions'], {'obs': obs_data, 'time_step':np.array([[int(self.timestep)]], dtype=np.float32)})[0]
        
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
        motion_quat = self.motionquat[int(self.timestep),0,:]
        motion_pos = self.motioninputpos[int(self.timestep),:]
        motion_vel = self.motioninputvel[int(self.timestep),:]  
        
        if self.fixed_pos:  
            # 前两个时间步计算初始转换矩阵
            if self.timeinit < 2:
                self.timeinit += 1.0
                self.compute_init_to_world(quat, motion_quat)# 计算初始转换矩阵（每次都计算，确保准确性）
        else:
            self.compute_init_to_world(quat, motion_quat)# 计算初始转换矩阵（每次都计算，确保准确性）
        
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
    

class DanceMotionPolicyGravityMjlab:
    """舞蹈动作策略管理类"""
    
    def __init__(self, motion_npz_path: str, model_onnx_path: str, start_frame: int = 0, fixed_pos: bool = False):
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
    
        self.num_obs = 157 #带位姿估计163(3pos+3lin_vel)
        
        self.motion_npz_path = motion_npz_path
        
        self.model_onnx_path = model_onnx_path

        self.initialize_model(motion_npz_path, model_onnx_path)
        
        self.fixed_pos = fixed_pos #是否固定位置，固定位置适合舞蹈动作
        
        self.timeinit = 0.0#计算初始转换矩阵计数器
        
        self.start_frame = start_frame
        
        self.end_frame = self.motionpos.shape[0] - 1
        
        self.timestep = self.start_frame
        
        self.num_actions = len(self.joint_name)
        
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        self.action_buffer = np.zeros((self.num_actions,), dtype=np.float32)
        
    # 初始化部分（完整版）
    def initialize_model(self, npz_path, onnx_path):
        # 加载运动数据
        # print("model init!!!")
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
                self.kps = self.stiffness_array.copy()
                # print(f"{prop.key}: {prop.value}")
                # print(" ")
                
            if prop.key == "joint_damping":
                self.damping_array = np.array([float(x) for x in prop.value.split(",")])
                self.kds = self.damping_array.copy()
                # print(f"{prop.key}: {prop.value}")
                # print(" ")      
            
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

        # print("BYDMIMIC model init finished!!!")
        # print("########################################################################")

    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega):
        # 使用预分配内存（如果适用）
        obs_data = self.create_obs_input(q, dq, quat, omega)
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        self.action = self.session.run(['actions'], {'obs': obs_data, 'time_step':np.array([[int(self.timestep)]], dtype=np.float32)})[0]
        
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
        motion_quat = self.motionquat[int(self.timestep),0,:]
        motion_pos = self.motioninputpos[int(self.timestep),:]
        motion_vel = self.motioninputvel[int(self.timestep),:]  
        
        if self.fixed_pos:  
            # 前两个时间步计算初始转换矩阵
            if self.timeinit < 2:
                self.timeinit += 1.0
                self.compute_init_to_world(quat, motion_quat)# 计算初始转换矩阵（每次都计算，确保准确性）
        else:
            self.compute_init_to_world(quat, motion_quat)# 计算初始转换矩阵（每次都计算，确保准确性）
        
        # create observation
        offset = 0
        motioninput = np.concatenate((motion_pos,motion_vel),axis=0)
        self.obs[offset:offset + 58] = motioninput
        offset += 58
        
        relmatrix = self.compute_relmatrix(quat, motion_quat)
        self.obs[offset:offset + 6] = relmatrix  
        offset += 6
        
        projected_gravity = get_gravity_orientation(quat)
        self.obs[offset:offset + 3] = projected_gravity
        offset += 3
        
        self.obs[offset:offset + 3] = omega 
        offset += 3
        
        self.obs[offset:offset + self.num_actions] = q - self.joint_pos_array  # joint positions
        offset += self.num_actions
        
        self.obs[offset:offset + self.num_actions] = dq  # joint velocities
        offset += self.num_actions   
        
        self.obs[offset:offset + self.num_actions] = self.action_buffer
        
        self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(157,)变成(1,157)并确保数据类型
        
        return self.obs_input
    
    
    
class DanceMotionPolicyGravityIsaaclab:
    """舞蹈动作策略管理类"""
    
    def __init__(self, motion_npz_path: str, model_onnx_path: str, start_frame: int = 0, fixed_pos: bool = False):
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
        
        self.num_obs = 157 #带位姿估计163(3pos+3lin_vel)
        
        self.motion_npz_path = motion_npz_path
        
        self.model_onnx_path = model_onnx_path

        self.initialize_model(motion_npz_path, model_onnx_path)
        
        self.fixed_pos = fixed_pos #是否固定位置，固定位置适合舞蹈动作
        
        self.timeinit = 0.0#计算初始转换矩阵计数器
        
        self.start_frame = start_frame
        
        self.end_frame = self.motionpos.shape[0] - 1
        
        self.timestep = self.start_frame
        
        self.num_actions = len(self.joint_name)
        
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        self.action_buffer = np.zeros((self.num_actions,), dtype=np.float32)
        
    # 初始化部分（完整版）
    def initialize_model(self, npz_path, onnx_path):
        # 加载运动数据
        # print("model init!!!")
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
                print(f"{prop.key}: {prop.value}")
                print(" ")
                
            if prop.key == "default_joint_pos":   
                self.joint_pos_array = np.array([float(x) for x in prop.value.split(",")])
                self.default_dof_pos = self.joint_pos_array[self.isaac_to_mujoco_idx].copy() #默认的动作位置（双臂自然下垂姿势）
                print(f"{prop.key}: {prop.value}")
                print(" ")

            # if prop.key == "joint_stiffness":
            #     self.stiffness_array = np.array([float(x) for x in prop.value.split(",")])
            #     # self.kps = self.stiffness_array.copy()
            #     print(f"{prop.key}: {prop.value}")
            #     print(" ")
                
            # if prop.key == "joint_damping":
            #     self.damping_array = np.array([float(x) for x in prop.value.split(",")])
            #     # self.kds = self.damping_array.copy()
            #     print(f"{prop.key}: {prop.value}")
            #     print(" ")      
            
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

        # print("BYDMIMIC model init finished!!!")
        # print("########################################################################")

    # 循环推理部分（极速版）
    def inference_step(self, q, dq, quat, omega):
        # 使用预分配内存（如果适用）
        obs_data = self.create_obs_input(q, dq, quat, omega)
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        self.action = self.session.run(['actions'], {'obs': obs_data, 'time_step':np.array([[int(self.timestep)]], dtype=np.float32)})[0]
        
        self.action = np.asarray(self.action).reshape(-1)
        self.action_buffer = self.action.copy()
        
        self.target_dof_pos = self.action[self.isaac_to_mujoco_idx] * self.action_scale[self.isaac_to_mujoco_idx]  + self.default_dof_pos
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
        motion_quat = self.motionquat[int(self.timestep),0,:]
        motion_pos = self.motioninputpos[int(self.timestep),:]#[self.mujoco_to_isaac_idx]
        motion_vel = self.motioninputvel[int(self.timestep),:]#[self.mujoco_to_isaac_idx]  
        
        if self.fixed_pos:  
            # 前两个时间步计算初始转换矩阵
            if self.timeinit < 2:
                self.timeinit += 1.0
                self.compute_init_to_world(quat, motion_quat)# 计算初始转换矩阵（每次都计算，确保准确性）
        else:
            self.compute_init_to_world(quat, motion_quat)# 计算初始转换矩阵（每次都计算，确保准确性）
        
        # create observation
        offset = 0
        motioninput = np.concatenate((motion_pos,motion_vel),axis=0)
        self.obs[offset:offset + 58] = motioninput
        offset += 58
        
        relmatrix = self.compute_relmatrix(quat, motion_quat)
        self.obs[offset:offset + 6] = relmatrix  
        offset += 6
        
        projected_gravity = get_gravity_orientation(quat)
        self.obs[offset:offset + 3] = projected_gravity
        offset += 3
        
        self.obs[offset:offset + 3] = omega 
        offset += 3
        
        self.obs[offset:offset + self.num_actions] = (q - self.default_dof_pos)[self.mujoco_to_isaac_idx]  # joint positions
        offset += self.num_actions
        
        self.obs[offset:offset + self.num_actions] = dq[self.mujoco_to_isaac_idx]  # joint velocities
        offset += self.num_actions   
        
        self.obs[offset:offset + self.num_actions] = self.action_buffer
        
        self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(157,)变成(1,157)并确保数据类型
        
        return self.obs_input
    
    

class DanceMotionPolicyGravityIsaaclabV2:
    """适配 reference residual action + command window + 10帧历史的 IsaacLab ONNX 部署类。"""

    def __init__(self, motion_npz_path: str, model_onnx_path: str, start_frame: int = 0, fixed_pos: bool = False):
        self.kps = np.array([
            108.448,162.672,176.421,
            176.421,176.421,54.224,176.421,33.493,21.771,
            176.421,176.421,54.224,176.421,33.493,21.771,
            54.224,54.224,16.747,54.224,16.747,16.747,16.747,
            54.224,54.224,16.747,54.224,16.747,16.747,16.747,
        ], dtype=np.float32)

        self.kds = np.array([
            6.904,10.356,11.231,
            11.231,11.231,3.452,11.231,2.132,1.386,
            11.231,11.231,3.452,11.231,2.132,1.386,
            3.452,3.452,1.066,3.452,1.066,1.066,1.066,
            3.452,3.452,1.066,3.452,1.066,1.066,1.066,
        ], dtype=np.float32)

        self.mujoco_to_isaac_idx = [
            15,22,0,16,23,1,17,24,2,18,25,3,9,19,26,
            4,10,20,27,5,11,21,28,6,12,7,13,8,14,
        ]

        self.isaac_to_mujoco_idx = [
            2,5,8,11,15,19,23,25,27,12,16,20,24,26,28,
            0,3,6,9,13,17,21,1,4,7,10,14,18,22,
        ]

        self.motion_npz_path = motion_npz_path
        self.model_onnx_path = model_onnx_path
        self.fixed_pos = fixed_pos
        self.timeinit = 0.0

        self.initialize_model(motion_npz_path, model_onnx_path)

        self.start_frame = start_frame
        self.end_frame = self.motioninputpos.shape[0] - 1
        self.timestep = self.start_frame
        self.num_actions = len(self.joint_name)

        self.action_buffer = np.zeros((self.num_actions,), dtype=np.float32)
        self.history_buffers = {}

    def initialize_model(self, npz_path, onnx_path):
        self.motion = np.load(npz_path)
        self.motionpos = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]

        model = onnx.load(onnx_path)
        meta = {prop.key: prop.value for prop in model.metadata_props}

        self.joint_name = meta["joint_names"].split(",")
        self.default_joint_pos_isaac = np.array(
            [float(x) for x in meta["default_joint_pos"].split(",")],
            dtype=np.float32,
        )
        self.default_dof_pos = self.default_joint_pos_isaac[self.isaac_to_mujoco_idx].copy()

        self.action_scale = np.array(
            [float(x) for x in meta["action_scale"].split(",")],
            dtype=np.float32,
        )

        self.command_window_offsets = np.array(
            [
                int(float(x))
                for x in meta.get("command_window_offsets", "-2,0,2,5,10").split(",")
                if x.strip()
            ],
            dtype=np.int64,
        )

        self.observation_names = meta.get(
            "observation_names",
            "command,motion_anchor_ori_b,projected_gravity,base_ang_vel,joint_pos,joint_vel,actions",
        ).split(",")

        self.observation_history_lengths = [
            int(float(x))
            for x in meta.get(
                "observation_history_lengths",
                "1,1,10,10,10,10,10",
            ).split(",")
        ]

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(onnx_path, providers=providers, sess_options=options)
        self.input_info = self.session.get_inputs()[0]
        self.num_obs = int(self.input_info.shape[1])
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

    def inference_step(self, q, dq, quat, omega, base_lin_vel_b=None, robot_pos_w=None, advance=False):
        obs_data = self.create_obs_input(q, dq, quat, omega, base_lin_vel_b, robot_pos_w)
        feed = {
            "obs": obs_data,
            "time_step": np.array([[int(self.timestep)]], dtype=np.float32),
        }

        action, ref_joint_pos = self.session.run(["actions", "joint_pos"], feed)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        ref_joint_pos = np.asarray(ref_joint_pos, dtype=np.float32).reshape(-1)

        self.action_buffer = action.copy()

        target_isaac = ref_joint_pos + action * self.action_scale
        target_mujoco = target_isaac[self.isaac_to_mujoco_idx].reshape(-1)

        if advance:
            self.timestep = min(int(self.timestep) + 1, self.end_frame)

        return target_mujoco

    def create_obs_input(self, q, dq, quat, omega, base_lin_vel_b=None, robot_pos_w=None):
        t = min(int(self.timestep), self.end_frame)
        motion_quat = self.motionquat[t, 0, :]

        if self.fixed_pos:
            if self.timeinit < 2:
                self.timeinit += 1.0
                self.compute_init_to_world(quat, motion_quat)
        else:
            self.compute_init_to_world(quat, motion_quat)

        features = {
            "command": self.build_command_window(t),
            "motion_anchor_ori_b": self.compute_relmatrix(quat, motion_quat).astype(np.float32),
            "projected_gravity": get_gravity_orientation(quat).astype(np.float32),
            "base_ang_vel": np.asarray(omega, dtype=np.float32),
            "joint_pos": (np.asarray(q, dtype=np.float32) - self.default_dof_pos)[self.mujoco_to_isaac_idx],
            "joint_vel": np.asarray(dq, dtype=np.float32)[self.mujoco_to_isaac_idx],
            "actions": self.action_buffer.astype(np.float32),
        }

        if "base_lin_vel" in self.observation_names:
            if base_lin_vel_b is None:
                raise ValueError("当前 ONNX 需要 base_lin_vel，请传入机身坐标系线速度 base_lin_vel_b。")
            features["base_lin_vel"] = np.asarray(base_lin_vel_b, dtype=np.float32)

        if "motion_anchor_pos_b" in self.observation_names:
            raise ValueError("当前部署类建议使用 Wo-State-Estimation 模型；此 ONNX 需要 motion_anchor_pos_b。")

        obs_parts = []
        for name, history_len in zip(self.observation_names, self.observation_history_lengths):
            value = features[name]
            if history_len > 1:
                value = self.push_history(name, value, history_len)
            obs_parts.append(value.reshape(-1))

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        if obs.shape[0] != self.num_obs:
            raise ValueError(f"obs dim mismatch: got {obs.shape[0]}, expected {self.num_obs}")

        return obs.reshape(1, -1)

    def build_command_window(self, timestep):
        frames = np.clip(
            timestep + self.command_window_offsets,
            0,
            self.end_frame,
        ).astype(np.int64)

        motion_pos = self.motioninputpos[frames]
        motion_vel = self.motioninputvel[frames]
        return np.concatenate((motion_pos, motion_vel), axis=-1).reshape(-1).astype(np.float32)

    def push_history(self, name, value, history_len):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if name not in self.history_buffers:
            self.history_buffers[name] = np.repeat(value[None, :], history_len, axis=0)
        else:
            self.history_buffers[name][:-1] = self.history_buffers[name][1:]
            self.history_buffers[name][-1] = value
        return self.history_buffers[name].reshape(-1)

    def compute_init_to_world(self, robot_quat, motion_quat):
        yaw_motion_quat = yaw_quat(motion_quat)
        yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat).reshape(3, 3)

        yaw_robot_quat = yaw_quat(robot_quat)
        yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat).reshape(3, 3)

        self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

    def compute_relmatrix(self, robot_quat, motion_quat):
        rel_quat = quaternion_multiply(matrix_to_quaternion_simple(self.init_to_world), motion_quat)
        rel_quat = quaternion_multiply(quaternion_conjugate(robot_quat), rel_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat).clip(min=1e-8)
        return quaternion_to_rotation_matrix(rel_quat)[:, :2].reshape(-1)

    

class DanceMotionPolicyGravityIsaaclabV3:
    """适配 default_joint_pos residual action + command window + 10帧历史的 IsaacLab ONNX 部署类。"""

    def __init__(self, motion_npz_path: str, model_onnx_path: str, start_frame: int = 0, fixed_pos: bool = False):
        self.kps = np.array([
            108.448,162.672,176.421,
            176.421,176.421,54.224,176.421,33.493,21.771,
            176.421,176.421,54.224,176.421,33.493,21.771,
            54.224,54.224,16.747,54.224,16.747,16.747,16.747,
            54.224,54.224,16.747,54.224,16.747,16.747,16.747,
        ], dtype=np.float32)

        self.kds = np.array([
            6.904,10.356,11.231,
            11.231,11.231,3.452,11.231,2.132,1.386,
            11.231,11.231,3.452,11.231,2.132,1.386,
            3.452,3.452,1.066,3.452,1.066,1.066,1.066,
            3.452,3.452,1.066,3.452,1.066,1.066,1.066,
        ], dtype=np.float32)

        self.mujoco_to_isaac_idx = [
            15,22,0,16,23,1,17,24,2,18,25,3,9,19,26,
            4,10,20,27,5,11,21,28,6,12,7,13,8,14,
        ]

        self.isaac_to_mujoco_idx = [
            2,5,8,11,15,19,23,25,27,12,16,20,24,26,28,
            0,3,6,9,13,17,21,1,4,7,10,14,18,22,
        ]

        self.motion_npz_path = motion_npz_path
        self.model_onnx_path = model_onnx_path
        self.fixed_pos = fixed_pos
        self.timeinit = 0.0

        self.initialize_model(motion_npz_path, model_onnx_path)

        self.start_frame = start_frame
        self.end_frame = self.motioninputpos.shape[0] - 1
        self.timestep = self.start_frame
        self.num_actions = len(self.joint_name)

        self.action_buffer = np.zeros((self.num_actions,), dtype=np.float32)
        self.history_buffers = {}

    def initialize_model(self, npz_path, onnx_path):
        self.motion = np.load(npz_path)
        self.motionpos = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]

        model = onnx.load(onnx_path)
        meta = {prop.key: prop.value for prop in model.metadata_props}

        self.joint_name = meta["joint_names"].split(",")
        self.default_joint_pos_isaac = np.array(
            [float(x) for x in meta["default_joint_pos"].split(",")],
            dtype=np.float32,
        )
        # self.default_joint_pos_isaac[[25, 26]] -= 0.05 #ankle_y微调，解决部分动作脚尖穿地问题
        # self.default_joint_pos_isaac[[27, 28]] += 0.05 #ankle_x微调，解决部分动作脚尖穿地问题
        
        self.default_dof_pos = self.default_joint_pos_isaac[self.isaac_to_mujoco_idx].copy()

        self.action_scale = np.array(
            [float(x) for x in meta["action_scale"].split(",")],
            dtype=np.float32,
        )
        # print(f"action_scale: {self.action_scale}")

        self.command_window_offsets = np.array(
            [
                int(float(x))
                for x in meta.get("command_window_offsets", "-2,0,2,5,10").split(",")
                if x.strip()
            ],
            dtype=np.int64,
        )

        self.observation_names = meta.get(
            "observation_names",
            "command,motion_anchor_ori_b,projected_gravity,base_ang_vel,joint_pos,joint_vel,actions",
        ).split(",")

        self.observation_history_lengths = [
            int(float(x))
            for x in meta.get(
                "observation_history_lengths",
                "1,1,10,10,10,10,10",
            ).split(",")
        ]

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(onnx_path, providers=providers, sess_options=options)
        self.input_info = self.session.get_inputs()[0]
        self.num_obs = int(self.input_info.shape[1])
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

    def inference_step(self, q, dq, quat, omega, base_lin_vel_b=None, robot_pos_w=None, advance=False):
        obs_data = self.create_obs_input(q, dq, quat, omega, base_lin_vel_b, robot_pos_w)
        feed = {
            "obs": obs_data,
            "time_step": np.array([[int(self.timestep)]], dtype=np.float32),
        }

        action = self.session.run(["actions"], feed)[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        self.action_buffer = action.copy()

        # self.action_scale[[7, 8, 13, 14]] *= 1.2
        # self.default_joint_pos_isaac[[25, 27, 26, 28]] += 0.05
        
        target_isaac = self.default_joint_pos_isaac + action * self.action_scale
        target_mujoco = target_isaac[self.isaac_to_mujoco_idx].reshape(-1)

        if advance:
            self.timestep = min(int(self.timestep) + 1, self.end_frame)

        return target_mujoco

    def create_obs_input(self, q, dq, quat, omega, base_lin_vel_b=None, robot_pos_w=None):
        t = min(int(self.timestep), self.end_frame)
        motion_quat = self.motionquat[t, 0, :]

        if self.fixed_pos:
            if self.timeinit < 2:
                self.timeinit += 1.0
                self.compute_init_to_world(quat, motion_quat)
        else:
            self.compute_init_to_world(quat, motion_quat)

        features = {
            "command": self.build_command_window(t),
            "motion_anchor_ori_b": self.compute_relmatrix(quat, motion_quat).astype(np.float32),
            "projected_gravity": get_gravity_orientation(quat).astype(np.float32),
            "base_ang_vel": np.asarray(omega, dtype=np.float32),
            "joint_pos": (np.asarray(q, dtype=np.float32) - self.default_dof_pos)[self.mujoco_to_isaac_idx],
            "joint_vel": np.asarray(dq, dtype=np.float32)[self.mujoco_to_isaac_idx],
            "actions": self.action_buffer.astype(np.float32),
        }

        if "base_lin_vel" in self.observation_names:
            if base_lin_vel_b is None:
                raise ValueError("当前 ONNX 需要 base_lin_vel，请传入机身坐标系线速度 base_lin_vel_b。")
            features["base_lin_vel"] = np.asarray(base_lin_vel_b, dtype=np.float32)

        if "motion_anchor_pos_b" in self.observation_names:
            raise ValueError("当前部署类建议使用 Wo-State-Estimation 模型；此 ONNX 需要 motion_anchor_pos_b。")

        obs_parts = []
        for name, history_len in zip(self.observation_names, self.observation_history_lengths):
            value = features[name]
            if history_len > 1:
                value = self.push_history(name, value, history_len)
            obs_parts.append(value.reshape(-1))

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        if obs.shape[0] != self.num_obs:
            raise ValueError(f"obs dim mismatch: got {obs.shape[0]}, expected {self.num_obs}")

        return obs.reshape(1, -1)

    def build_command_window(self, timestep):
        frames = np.clip(
            timestep + self.command_window_offsets,
            0,
            self.end_frame,
        ).astype(np.int64)

        motion_pos = self.motioninputpos[frames]
        motion_vel = self.motioninputvel[frames]
        return np.concatenate((motion_pos, motion_vel), axis=-1).reshape(-1).astype(np.float32)

    def push_history(self, name, value, history_len):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if name not in self.history_buffers:
            self.history_buffers[name] = np.repeat(value[None, :], history_len, axis=0)
        else:
            self.history_buffers[name][:-1] = self.history_buffers[name][1:]
            self.history_buffers[name][-1] = value
        return self.history_buffers[name].reshape(-1)

    def compute_init_to_world(self, robot_quat, motion_quat):
        yaw_motion_quat = yaw_quat(motion_quat)
        yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat).reshape(3, 3)

        yaw_robot_quat = yaw_quat(robot_quat)
        yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat).reshape(3, 3)

        self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

    def compute_relmatrix(self, robot_quat, motion_quat):
        rel_quat = quaternion_multiply(matrix_to_quaternion_simple(self.init_to_world), motion_quat)
        rel_quat = quaternion_multiply(quaternion_conjugate(robot_quat), rel_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat).clip(min=1e-8)
        return quaternion_to_rotation_matrix(rel_quat)[:, :2].reshape(-1)
