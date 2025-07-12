
import torch
import os.path as osp
import numpy as np
from typing import Dict
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.inference.exp_filter import expFilter
import onnxruntime as ort

class humanoid_hurdle_onnx_Agent(baseAgent):
    def __init__(self, policy_path):
        self.num_actions = 23
        self.num_prop_obs_input = 240
        self.include_history_steps = 5

        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.onnx_session = ort.InferenceSession(
            policy_path,
            providers=providers,
            sess_options=options
        )
        
        self.default_dof_pos = [
            0.,0.,0.,-1.57,0.,
            0.,0.,0.,-1.57,0.,
            0.,
            0.,0.,-0.5,1.0,-0.5,0.,
            0.,0.,-0.5,1.0,-0.5,0.,
        ]
        self.default_dof_pos = np.array(self.default_dof_pos)

        self.obs_scale={
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "ang_vel": 0.25,
            "height_measurements": 5.0,
        }
        self.action_scale = 0.25
        self.prop_obs_history = np.zeros((self.include_history_steps, self.num_prop_obs_input))
        self.clip_observation = 100.
        self.clip_action = 100.

        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter = expFilter(0.6)
        self.bootstrap()

    def bootstrap(self):
        "预热用"
        obs_group={
            "dof_pos":np.zeros(23,dtype=float),
            "dof_vel":np.zeros(23,dtype=float),
            "angular_velocity":np.zeros(3,dtype=float),
            "commands":np.zeros(3,dtype=float),
            "projected_gravity":np.zeros(3,dtype=float),
            "height_map":np.zeros(18*9),
        }
        self.inference(obs_group)

    def build_observations(self, obs_group):
        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"]
        obs_height_map = (obs_group["height_map"] - 1.0) * self.obs_scale["height_measurements"]

        # 本体感知proprioception 9+69+162=240
        prop_obs = np.concatenate((
            np.array([obs_commands[2], 0, obs_commands[0]]), # 3
            obs_base_ang_vel, # 3
            obs_projected_gravity, # 3
            obs_dof_pos, # 23
            obs_dof_vel, # 23
            obs_last_actions, # 23
            obs_height_map, # 18x9=162
        ),axis=-1)

        self.prop_obs_history=np.roll(self.prop_obs_history,shift=-1,axis=0)
        self.prop_obs_history[-1,:] = prop_obs

        return self.prop_obs_history.flatten()
    
    def inference(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        prop_obs = self.build_observations(obs_group)

        input_feed = {
            "input": prop_obs.flatten()[None,:].astype(np.float32),
        }
        actions = np.squeeze(self.onnx_session.run(["output"], input_feed)) # test
        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        return dof_pos_target_urdf
    
    def reset(self):
        self.prop_obs_history = np.zeros((self.include_history_steps, self.num_prop_obs_input))
        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter.reset()

if __name__=="__main__":
    a=humanoid_hurdle_onnx_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/BxiParkour_0712.onnx")
    obs_group={
        "dof_pos":np.zeros(23),
        "dof_vel":np.zeros(23),
        "angular_velocity":np.zeros(3),
        "projected_gravity":np.zeros(3),
        "commands":np.zeros(3),
        "height_map":np.zeros(18*9)
    }
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    for i in range(100):
        print(a.inference(obs_group))
