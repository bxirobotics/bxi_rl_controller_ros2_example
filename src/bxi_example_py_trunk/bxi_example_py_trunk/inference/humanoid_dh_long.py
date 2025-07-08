import torch
import numpy as np
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.inference.exp_filter import expFilter
import time

class humanoid_dh_long_Agent(baseAgent):
    def __init__(self, policy_path, device):
        self.device = device
        self.num_prop_obs_input = 47
        self.num_estimator_input = 42
        self.include_history_steps = 5
        self.long_history = 64
        self.num_actions = 12

        self.script_module = torch.jit.load(policy_path).to(self.device)

        # dof_pos归一化范围
        dof_pos_limits=[[-0.7850,  0.7850],
                        [-0.5230,  0.7850],
                        [-2.0000,  1.0000],
                        [ 0.0000,  2.3550],
                        [-2.0000,  2.0000],
                        [-0.3490,  0.3490],
                        [-0.7850,  0.7850],
                        [-0.7850,  0.5230],
                        [-2.0000,  1.0000],
                        [ 0.0000,  2.3550],
                        [-2.0000,  2.0000],
                        [-0.3490,  0.3490]]
        self.dof_pos_limits=torch.tensor(dof_pos_limits,device=self.device)

        self.default_dof_pos = [0., 0., -0.3, 0.6, -0.3, 0.,
                                0., 0., -0.3, 0.6, -0.3, 0.]
        self.default_dof_pos = torch.tensor(self.default_dof_pos,device=self.device)

        l_low=self.default_dof_pos-self.dof_pos_limits[:,0]
        l_up=self.dof_pos_limits[:,1]-self.default_dof_pos
        dof_scale=1.0/torch.max(l_low,l_up)
        self.obs_scale={
            "dof_pos": dof_scale,
            "dof_vel": 0.1,
            "ang_vel": 0.1,
        }
        self.action_scale = 1.0

        self.prop_obs_history = torch.zeros((self.num_prop_obs_input,self.long_history),
                                             device=self.device,requires_grad=False)
        self.estimator_obs_history = torch.zeros((self.num_estimator_input,self.include_history_steps),
                                             device=self.device,requires_grad=False)   

        self.last_actions_buf = torch.zeros(self.num_actions,device=self.device,requires_grad=False)
        self.inference_count = 0
        self.dt = 0.01
        self.gait_period = 0.6
        self.exp_filter = expFilter(0.6)
        self.bootstrap()

    def bootstrap(self):
        "预热用"
        obs_group={
            "dof_pos":torch.zeros(12,dtype=float,device="cuda"),
            "dof_vel":torch.zeros(12,dtype=float,device="cuda"),
            "angular_velocity":torch.zeros(3,dtype=float,device="cuda"),
            "commands":torch.zeros(3,dtype=float,device="cuda"),
            "projected_gravity":torch.zeros(3,dtype=float,device="cuda"),
        }
        self.inference(obs_group)

    def get_phase(self):
        phase_1 = self.inference_count * self.dt / self.gait_period
        phase = torch.tensor([phase_1], device=self.device)
        sin_phase = torch.sin(2 * torch.pi * phase)
        cos_phase= torch.cos(2 * torch.pi * phase)
        obs = torch.cat([sin_phase,cos_phase],dim=-1)
        return obs

    def build_observations(self, obs_group):
        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"]
        obs_commands[...,2] *= self.obs_scale["ang_vel"]
        obs_phase = self.get_phase()

        # 本体感知proprioception 47
        prop_obs = torch.cat((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
            obs_commands,
            obs_phase
        ),dim=-1)

        # estimator 42
        estimator_obs = torch.cat((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
        ),dim=-1)

        self.prop_obs_history=self.prop_obs_history.roll(shifts=-1,dims=1)
        self.prop_obs_history[:,-1] = prop_obs

        self.estimator_obs_history=self.estimator_obs_history.roll(shifts=-1,dims=1)
        self.estimator_obs_history[:,-1] = estimator_obs

        return self.prop_obs_history, self.estimator_obs_history
    
    def inference(self, obs_group):
        start = time.time()
        for obs in obs_group.values():
            obs.clamp_(-10., 10.)

        prop_obs_history, estimator_obs_history = self.build_observations(obs_group)

        with torch.inference_mode():
            actions = self.script_module.inference(prop_obs_history.flatten().unsqueeze(0),
                                                    estimator_obs_history.flatten().unsqueeze(0),
                                                    ).squeeze(0)
            actions.clip_(-1.,1.)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos

        self.inference_count += 1
        dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        end = time.time()
        print(f"推理时间:{(end-start)*1000:.1f}ms")
        return dof_pos_target_urdf
    
    def reset(self):
        self.prop_obs_history = torch.zeros((self.num_prop_obs_input,self.long_history),
                                             device=self.device,requires_grad=False)
        self.estimator_obs_history = torch.zeros((self.num_estimator_input,self.include_history_steps),
                                             device=self.device,requires_grad=False)   
        self.exp_filter.reset()

if __name__=="__main__":
    a=humanoid_dh_long_Agent("cpu")
    obs_group={
        "dof_pos":torch.zeros(12),
        "dof_vel":torch.zeros(12),
        "angular_velocity":torch.zeros(3),
        "projected_gravity":torch.zeros(3),
        "commands":torch.zeros(3),
    }
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    
    start = time.time()
    for i in range(100):
        print(a.inference(obs_group).cpu().numpy())
    end = time.time()
    print("推理时间:",(end-start)/100.)
