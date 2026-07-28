import time
import numpy as np
from bxi_example_py_elf3.framework.mod_api.geometry import get_gravity_orientation
from bxi_example_py_elf3.framework.joints import JointTargetView
from .joints import ELF3_POLICY_JOINTS

from bxi_example_py_elf3.framework.inference.api import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.inference.contract import PolicyJointContract
from bxi_example_py_elf3.framework.inference.history import HistoryBuffer
from bxi_example_py_elf3.framework.inference.model import ModelSpec
from bxi_example_py_elf3.framework.inference.policy import JointPolicy
from bxi_example_py_elf3.framework.inference.runtime import InferenceRuntime, default_runtime


class HumanoidGaitPolicyLiteIsaaclab(JointPolicy):
    """不带步态输入的AMP行走动作策略管理类"""

    joint_contract = PolicyJointContract(
        observation=ELF3_POLICY_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        model: str | ModelSpec,
        *,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ):
        """
        初始化策略

        Args:
            model: 模型路径或包含多个后端模型的 ModelSpec

        Usage:
            ##1.初始化模型
            self.amp_policy = HumanoidGaitPolicyLite("path/to/model.onnx")

            ##2.推理动作
            target, velocity = policy.step(q, dq, quat, omega, cmd_vel)
        """

        super().__init__()
        self._runtime = runtime or default_runtime()
        self._policy_name = "amp"

        self._action_scale = np.array(
            [
                0.231,
                0.231,
                0.231,
                0.231,
                0.231,
                0.154,
                0.373,
                0.373,
                0.213,
                0.231,
                0.231,
                0.213,
                0.213,
                0.373,
                0.373,
                0.213,
                0.213,
                0.373,
                0.373,
                0.231,
                0.231,
                0.373,
                0.373,
                0.213,
                0.213,
                0.373,
                0.373,
                0.23,
                0.23,
            ],
            dtype=np.float32,
        )

        self._kp = np.array(
            [  # 奔跑的关节kp，和joint_name顺序一一对应
                108.448,
                162.672,
                176.421,
                176.421,
                176.421,
                54.224,
                176.421,
                33.493,
                21.771,
                176.421,
                176.421,
                54.224,
                176.421,
                33.493,
                21.771,
                54.224,
                54.224,
                16.747,
                54.224,
                16.747,
                16.747,
                16.747,
                54.224,
                54.224,
                16.747,
                54.224,
                16.747,
                16.747,
                16.747,
            ],
            dtype=np.float32,
        )

        self._kd = np.array(
            [  # 奔跑的关节kd，和joint_name顺序一一对应
                6.904,
                10.356,
                11.231,
                11.231,
                11.231,
                3.452,
                11.231,
                2.132,
                1.386,
                11.231,
                11.231,
                3.452,
                11.231,
                2.132,
                1.386,
                3.452,
                3.452,
                1.066,
                3.452,
                1.066,
                1.066,
                1.066,
                3.452,
                3.452,
                1.066,
                3.452,
                1.066,
                1.066,
                1.066,
            ],
            dtype=np.float32,
        )

        # 双臂自然下垂姿势
        self._default_position = np.array(
            [  # 指定的固定关节角度
                0.0,
                0.0,
                0.0,
                -0.3,
                0.0,
                0.0,
                0.6,
                -0.3,
                0.0,
                -0.3,
                0.0,
                0.0,
                0.6,
                -0.3,
                0.0,
                0.2,
                0.2,
                0.0,
                0.6,
                0.0,
                0.0,
                0.0,
                0.2,
                -0.2,
                0.0,
                0.6,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.mujoco_to_isaac_idx = [
            15,  # 'l_shoulder_y_joint', 0
            22,  #  'r_shoulder_y_joint', 1
            0,  #  'waist_y_joint', 2
            16,  #  'l_shoulder_x_joint',3
            23,  #  'r_shoulder_x_joint', 4
            1,  #  'waist_x_joint', 5
            17,  #  'l_shoulder_z_joint',6
            24,  #  'r_shoulder_z_joint', 7
            2,  #  'waist_z_joint', 8
            18,  #  'l_elbow_y_joint',9
            25,  #  'r_elbow_y_joint', 10
            3,  #  'l_hip_y_joint', 11
            9,  #  'r_hip_y_joint', 12
            19,  #  'l_wrist_x_joint',13
            26,  #  'r_wrist_x_joint', 14
            4,  #  'l_hip_x_joint', 15
            10,  #  'r_hip_x_joint', 16
            20,  #  'l_wrist_y_joint', 17
            27,  #  'r_wrist_y_joint', 18
            5,  #  'l_hip_z_joint', 19
            11,  #  'r_hip_z_joint', 20
            21,  #  'l_wrist_z_joint', 21
            28,  #  'r_wrist_z_joint', 22
            6,  #  'l_knee_y_joint', 23
            12,  #  'r_knee_y_joint', 24
            7,  #  'l_ankle_y_joint', 25
            13,  #  'r_ankle_y_joint', 26
            8,  #  'l_ankle_x_joint', 27
            14,  #  'r_ankle_x_joint',28
        ]

        self.isaac_to_mujoco_idx = [
            2,  # "waist_y_joint",
            5,  # "waist_x_joint",
            8,  # "waist_z_joint",
            11,  # "l_hip_y_joint",   # 左腿_髋关节_z轴
            15,  # "l_hip_x_joint",   # 左腿_髋关节_x轴
            19,  # "l_hip_z_joint",   # 左腿_髋关节_y轴
            23,  # "l_knee_y_joint",   # 左腿_膝关节_y轴
            25,  # "l_ankle_y_joint",   # 左腿_踝关节_y轴
            27,  # "l_ankle_x_joint",   # 左腿_踝关节_x轴
            12,  # "r_hip_y_joint",   # 右腿_髋关节_z轴
            16,  # "r_hip_x_joint",   # 右腿_髋关节_x轴
            20,  # "r_hip_z_joint",   # 右腿_髋关节_y轴
            24,  # "r_knee_y_joint",   # 右腿_膝关节_y轴
            26,  # "r_ankle_y_joint",   # 右腿_踝关节_y轴
            28,  # "r_ankle_x_joint",   # 右腿_踝关节_x轴
            0,  # "l_shoulder_y_joint",   # 左臂_肩关节_y轴
            3,  # "l_shoulder_x_joint",   # 左臂_肩关节_x轴
            6,  # "l_shoulder_z_joint",   # 左臂_肩关节_z轴
            9,  # "l_elbow_y_joint",   # 左臂_肘关节_y轴
            13,  # "l_wrist_x_joint",
            17,  # "l_wrist_y_joint",
            21,  # "l_wrist_z_joint",
            1,  # "r_shoulder_y_joint",   # 右臂_肩关节_y轴
            4,  # "r_shoulder_x_joint",   # 右臂_肩关节_x轴
            7,  # "r_shoulder_z_joint",   # 右臂_肩关节_z轴
            10,  # "r_elbow_y_joint",    # 右臂_肘关节_y轴
            14,  # "r_wrist_x_joint",
            18,  # "r_wrist_y_joint",
            22,  # "r_wrist_z_joint",
        ]

        # Initial command vel
        self._command = np.zeros(3, dtype=np.float32)

        # Gait parameters
        self.dt: float = 0.005  # 控制周期
        self.gait_air_ratio_l: float = 0.38  # 步态空中比率l
        self.gait_air_ratio_r: float = 0.38  # 步态空中比率r
        self.gait_phase_offset_l: float = 0.38  # 步态相位偏移l
        self.gait_phase_offset_r: float = 0.88  # 步态相位偏移r
        self.gait_cycle: float = 0.85  # 步态周期（秒）

        # Number of actions and observations.
        # 960/1020D policies are full-body models. 540D policies are no-arm
        # models and only command waist + legs; applause overlays the arms.
        self.robot_dof_num = self.joint_contract.observation.dof_num
        self.policy_action_dim = self.joint_contract.action.dof_num
        self.controlled_action_dim = 15
        self.num_actions = self.policy_action_dim
        self.obs_action_dim = self.policy_action_dim

        self.full_body_mujoco_idx = np.arange(self.policy_action_dim, dtype=np.int64)
        self.full_body_isaac_idx = np.asarray(self.isaac_to_mujoco_idx, dtype=np.int64)
        self.controlled_mujoco_idx = np.arange(
            self.controlled_action_dim, dtype=np.int64
        )
        self.controlled_isaac_idx = np.asarray(
            [self.isaac_to_mujoco_idx[i] for i in self.controlled_mujoco_idx],
            dtype=np.int64,
        )
        self.isaac_order_mujoco_idx = np.asarray(
            self.mujoco_to_isaac_idx, dtype=np.int64
        )
        self.noarm_policy_isaac_idx = np.asarray(
            [
                i
                for i, mujoco_idx in enumerate(self.isaac_order_mujoco_idx)
                if mujoco_idx < self.controlled_action_dim
            ],
            dtype=np.int64,
        )
        self.noarm_policy_mujoco_idx = self.isaac_order_mujoco_idx[
            self.noarm_policy_isaac_idx
        ]
        self.output_mujoco_idx = self.full_body_mujoco_idx
        self.last_action_isaac_idx = self.full_body_isaac_idx
        self.controlled_action_scale = self._action_scale[self.full_body_isaac_idx]

        # The model input width selects the 540D, 960D or 1020D layout.
        self.num_obs = 960  # 默认值，将被动态调整
        self.obs_history_len = 10
        self.single_obs_dim = 3 + 3 + 3 + self.num_actions * 3  # 默认96维，将被动态调整
        self.extra_obs_dim = 0  # 额外观测维度（用于1020维模型），默认0

        self._initialize_model(model, backend)

        self._estimated_velocity = np.zeros(3, dtype=np.float32)
        self.default_target = JointTargetView(
            self.joint_contract.action,
            self._default_position,
            self._kp,
            self._kd,
        )
        self.publish_output(
            self._target,
            self._kp,
            self._kd,
            estimated_velocity=self._estimated_velocity,
        )

    def _initialize_model(self, model, backend):
        spec = (
            model
            if isinstance(model, ModelSpec)
            else ModelSpec.portable_onnx(
                model,
                input_names=("obs",),
                output_names=("actions",),
            )
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)

        # 【关键】自动检测模型输入维度，兼容960和1020
        input_shape = self._backend.input_shape("obs")
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
            actual_input_dim = input_shape[-1]
        else:
            actual_input_dim = 960  # 默认值

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
            self.controlled_action_scale = self._action_scale[self.controlled_isaac_idx]
            self.extra_obs_dim = 0
        elif actual_input_dim == 960:
            # 960维模型：96维/步 × 10步
            self.single_obs_dim = 96  # 3+3+3+29+29+29
            self.obs_action_dim = self.policy_action_dim
            self.output_mujoco_idx = self.full_body_mujoco_idx
            self.last_action_isaac_idx = self.full_body_isaac_idx
            self.controlled_action_scale = self._action_scale[self.full_body_isaac_idx]
            self.extra_obs_dim = 0
        elif actual_input_dim == 1020:
            # 1020维模型：102维/步 × 10步
            self.single_obs_dim = 102  # 96 + 6 extra
            self.obs_action_dim = self.policy_action_dim
            self.output_mujoco_idx = self.full_body_mujoco_idx
            self.last_action_isaac_idx = self.full_body_isaac_idx
            self.controlled_action_scale = self._action_scale[self.full_body_isaac_idx]
            self.extra_obs_dim = 6
        else:
            raise ValueError(
                f"Unsupported AMP model input dimension: {actual_input_dim}. "
                "Expected 540, 960, or 1020."
            )

        # 更新总观测维度
        self.num_obs = self.single_obs_dim * self.obs_history_len
        self._input = np.zeros((1, self.num_obs), dtype=np.float32)
        self._inputs = {"obs": self._input}

        # Initialize variables
        output_dim = int(self._backend.output_shape("actions")[-1])
        self._action = np.zeros(output_dim, dtype=np.float32)
        self._previous_action = np.zeros(self.policy_action_dim, dtype=np.float32)
        self._previous_controlled_action = np.zeros(
            self.output_mujoco_idx.size,
            dtype=np.float32,
        )
        self._target = self._target_buffer.position
        np.copyto(self._target, self._default_position)
        self._single_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        self._joint_delta = np.empty(self.robot_dof_num, dtype=np.float32)
        self._scaled_action = np.empty(self.output_mujoco_idx.size, dtype=np.float32)
        self._history = HistoryBuffer(
            self.obs_history_len,
            (self.single_obs_dim,),
            dtype=np.float32,
        )
        self._obs = self._input[0]

        if self.extra_obs_dim > 0:
            self.episode_length_buf = 0
            self.gait_phase = np.zeros(2)  # 步态相位
            self.gait_cycle = self.gait_cycle
            self.phase_ratio = np.array(
                [self.gait_air_ratio_l, self.gait_air_ratio_r]
            )  # 步态空中比率[0.38,0.38]
            self.phase_offset = np.array(
                [self.gait_phase_offset_l, self.gait_phase_offset_r]
            )  # 步态相位偏移[0.38,0.88]

        self._fill_history(
            self._default_position,
            np.zeros_like(self._default_position),
            np.array([1.0, 0.0, 0.0, 0.0]),  # 单位四元数
            np.zeros(3),  # 初始角速度
            np.array([0.0, 0.0, 0.0]),  # 初始命令速度
        )

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        joints = self.bind_joints(frame)
        cmd_vel = frame.command
        if cmd_vel is None:
            raise ValueError("HumanoidGaitPolicyLiteIsaaclab requires frame.command")
        monitor = self._runtime.options.monitor_enabled
        if monitor:
            total_started = time.perf_counter_ns()
        # 如果启用了额外观测（步态），先更新步态相位，使观测包含当前相位
        if self.extra_obs_dim > 0 and advance:
            # 计数并计算当前相位
            self.episode_length_buf += 1
            t = self.episode_length_buf * dt / self.gait_cycle
            self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
            self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

        self._update_observation(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
            cmd_vel,
            advance=advance,
        )
        if monitor:
            input_finished = time.perf_counter_ns()
        raw_action = self._backend.run(self._inputs)["actions"]
        if monitor:
            backend_finished = time.perf_counter_ns()
        np.copyto(self._action, np.asarray(raw_action).reshape(-1))

        out_len = self._action.shape[0]
        output_action_dim = self.output_mujoco_idx.shape[0]
        if out_len < output_action_dim:
            raise ValueError(
                f"ONNX action dim is {out_len}, expected at least {output_action_dim}"
            )

        if out_len >= self.policy_action_dim:
            np.copyto(self._previous_action, self._action[: self.policy_action_dim])
            np.take(
                self._previous_action,
                self.last_action_isaac_idx,
                out=self._previous_controlled_action,
            )
            controlled_action = self._previous_controlled_action
            vel_start = self.policy_action_dim
        else:
            np.copyto(
                self._previous_controlled_action,
                self._action[:output_action_dim],
            )
            controlled_action = self._previous_controlled_action
            self._previous_action.fill(0.0)
            self._previous_action[self.last_action_isaac_idx] = controlled_action
            vel_start = output_action_dim

        self._estimated_velocity.fill(0.0)
        if out_len > vel_start:
            vel_len = min(3, out_len - vel_start)
            self._estimated_velocity[:vel_len] = self._action[
                vel_start : vel_start + vel_len
            ]

        np.copyto(self._target, self._default_position)
        np.multiply(
            controlled_action,
            self.controlled_action_scale,
            out=self._scaled_action,
        )
        self._target[:output_action_dim] += self._scaled_action

        if monitor:
            output_finished = time.perf_counter_ns()
            self._runtime.monitor.record(
                self._policy_name,
                input_finished - total_started,
                backend_finished - input_finished,
                output_finished - backend_finished,
                output_finished - total_started,
            )

        return self.output

    def reset(self, frame: InferenceFrame) -> None:
        joints = self.bind_joints(frame)
        cmd_vel = frame.command
        if cmd_vel is None:
            raise ValueError("HumanoidGaitPolicyLiteIsaaclab requires frame.command")
        self._action.fill(0.0)
        self._previous_action.fill(0.0)
        self._previous_controlled_action.fill(0.0)
        self._estimated_velocity.fill(0.0)
        np.copyto(self._command, cmd_vel, casting="unsafe")
        if self.extra_obs_dim > 0:
            self.episode_length_buf = 0
        np.copyto(self._target, self._default_position)
        self._fill_history(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
            self._command,
        )
        self.publish_output(
            self._target,
            self._kp,
            self._kd,
            estimated_velocity=self._estimated_velocity,
        )

    def _fill_history(self, qj, dqj, quat, omega, cmd_vel):
        single_obs = self._build_single_observation(qj, dqj, quat, omega, cmd_vel)
        self._history.fill(single_obs)
        self._refresh_observation_buffer()

    def _update_observation(
        self, qj, dqj, quat, omega, cmd_vel, *, advance: bool
    ):
        single_obs = self._build_single_observation(qj, dqj, quat, omega, cmd_vel)
        if advance:
            self._history.append(single_obs)
            self._refresh_observation_buffer()
        else:
            self._history.preview_append_into(single_obs, self._obs)

    def _build_single_observation(self, qj, dqj, quat, omega, cmd_vel):
        gravity_orientation = get_gravity_orientation(quat)
        qj = np.asarray(qj, dtype=np.float32)
        dqj = np.asarray(dqj, dtype=np.float32)
        np.copyto(self._command, cmd_vel, casting="unsafe")

        if qj.shape[0] != self.robot_dof_num:
            raise ValueError(f"qj dim is {qj.shape[0]}, expected {self.robot_dof_num}")
        if dqj.shape[0] != self.robot_dof_num:
            raise ValueError(
                f"dqj dim is {dqj.shape[0]}, expected {self.robot_dof_num}"
            )

        # Create single observation with dynamic dimensions
        single_obs = self._single_obs
        single_obs.fill(0.0)

        # 【标准】omega + gravity + cmd_vel + joint_pos + joint_vel + last_action
        single_obs[0:3] = omega  # 3维
        single_obs[3:6] = gravity_orientation  # 3维
        single_obs[6:9] = self._command  # 3维
        if self.obs_action_dim == self.controlled_action_dim:
            obs_q_idx = self.output_mujoco_idx
            obs_dq_idx = self.output_mujoco_idx
            obs_last_action = self._previous_controlled_action
        else:
            obs_q_idx = self.mujoco_to_isaac_idx
            obs_dq_idx = self.mujoco_to_isaac_idx
            obs_last_action = self._previous_action

        np.subtract(qj, self._default_position, out=self._joint_delta)
        np.take(
            self._joint_delta,
            obs_q_idx,
            out=single_obs[9 : 9 + self.obs_action_dim],
        )
        np.take(
            dqj,
            obs_dq_idx,
            out=single_obs[9 + self.obs_action_dim : 9 + 2 * self.obs_action_dim],
        )
        single_obs[
            9 + 2 * self.obs_action_dim : 9 + 3 * self.obs_action_dim
        ] = obs_last_action

        # 【兼容】如果需要额外维度（如1020维模型），补零
        if self.extra_obs_dim > 0:
            # 对于102维（1020维模型），在末尾补6维零
            # single_obs[96:102] = 0  (已在初始化时设为0，无需显式赋值)
            single_obs[9 + 3 * self.num_actions : 11 + 3 * self.num_actions] = np.sin(
                2 * np.pi * self.gait_phase
            )  # 2 #步态相位正弦值
            single_obs[11 + 3 * self.num_actions : 13 + 3 * self.num_actions] = np.cos(
                2 * np.pi * self.gait_phase
            )  # 2 #步态相位余弦值
            single_obs[
                13 + 3 * self.num_actions : 15 + 3 * self.num_actions
            ] = self.phase_ratio  # 2 #步态空中比率

        return single_obs

    def _refresh_observation_buffer(self):
        self._history.write_into(self._obs)

    def close(self):
        self._backend.close()
