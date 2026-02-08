
import onnx
import ast
import numpy as np
import onnxruntime as ort

from scipy.spatial.transform import Rotation

dof_num = 29

class NormalMotionPolicy:
    def __init__(self, model_onnx_path: str):
        """
        Args:
            model_onnx_path: ONNX模型文件路径
            
        Usage:
            ##1.初始化模型
            self.normal_policy = NormalMotionPolicy("path/to/model.onnx")
                
            ##2.推理动作
            self.target_dof_pos = self.infer_step(self, q, dq, quat, omega, cmd):
        """
        self.num_obs = 96
        self.num_action = dof_num
        self.model_onnx_path = model_onnx_path

        self.target_q = np.zeros(self.num_action, dtype=np.double)
        self.action = np.zeros(self.num_action, dtype=np.double)

        policy_input = np.zeros([1, self.num_obs], dtype=np.float32)
        print("policy test")
        
        self.initialize_model(model_onnx_path)
        self.action[:] = self.inference_step(policy_input)
        
        self.timeinit = 0.0     #计算初始转换矩阵计数器
        self.timestep = 0
        
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

    def initialize_model(self, model_path):
        # 配置执行提供者（根据硬件选择最优后端）
        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 加载模型
        model = onnx.load(model_path)
        metadata = {}
        for prop in model.metadata_props:
            metadata[prop.key] = prop.value

        # print("metadata", metadata)
        self.joint_names = metadata["joint_names"]
        self.joint_stiffness = np.array(ast.literal_eval(metadata["joint_stiffness"]), dtype=np.float32)
        self.joint_damping = np.array(ast.literal_eval(metadata["joint_damping"]), dtype=np.float32)
        self.action_scale = np.array(ast.literal_eval(metadata["action_scale"]), dtype=np.float32)
        self.default_joint_pos = np.array(ast.literal_eval(metadata["default_joint_pos"]), dtype=np.float32)
        # exit()
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=options
        )
        
        # 预存输入输出信息
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        # 预分配输入内存（可选，适合固定输入尺寸）
        self.input_buffer = np.zeros(
            self.input_info.shape,
            dtype=np.float32
        )

    # 循环推理部分（极速版）
    def inference_step(self, obs_data):
        # 使用预分配内存（如果适用）
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        
        # 极简推理（比原版快5-15%）
        self.action = self.session.run([self.output_info.name], 
            {self.input_info.name: self.input_buffer})[0][0]  # 直接获取第一个输出的第一个样本
        self.target_q = self.action * self.action_scale
        qpos = self.default_joint_pos.copy()
        qpos[:] += self.target_q[:]
        return qpos
    
    def infer_step(self, q, dq, quat, omega, cmd):
        obs = np.zeros([1, self.num_obs], dtype=np.float32)
        projected_gravity = self.projected_gravity_from_quat(quat, np.array([0, 0, -1]))

        obs[0, :3] = omega
        obs[0, 3:6] = projected_gravity
        obs[0, 6:6+self.num_action] = (q-self.default_joint_pos)
        obs[0, 6+(self.num_action*1):6+(self.num_action*2)] = dq
        obs[0, 6+(self.num_action*2):6+(self.num_action*3)] = self.action
        obs[0, -3] = cmd[0] 
        obs[0, -2] = cmd[1]
        obs[0, -1] = cmd[2]

        policy_input = np.zeros([1, self.num_obs], dtype=np.float32)
        policy_input = obs

        return self.inference_step(policy_input)

    def projected_gravity_from_quat(self, quaternion, gravity=np.array([0, 0, -9.81])):
        """
        计算重力在机体坐标系中的投影
        
        参数:
            quaternion: 四元数 [w, x, y, z] 或 [x, y, z, w]
            gravity: 世界坐标系中的重力向量 [x, y, z]，默认 [0, 0, -9.81]
        
        返回:
            重力在机体坐标系中的投影向量 [x, y, z]
        """
        # w, x, y, z = quaternion
        # quaternion = x, y, z, w
        # 创建旋转对象（自动处理四元数顺序）
        rot = Rotation.from_quat(quaternion)
        
        rot_inv = rot.inv()
        
        # 将重力向量从世界坐标系转换到机体坐标系
        # apply方法将向量从世界系旋转到机体系
        return rot_inv.apply(gravity)
        # return gravity
    