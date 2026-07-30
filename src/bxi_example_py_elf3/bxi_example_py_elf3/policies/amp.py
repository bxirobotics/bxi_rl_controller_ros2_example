import numpy as np
from bxi_example_py_elf3.framework.mod_api.geometry import get_gravity_orientation
from bxi_example_py_elf3.framework.joints import CompiledJointMap, JointTargetView
from .joints import (
    ELF3_ISAAC_PARAMETERS,
    ELF3_LOWER_BODY_JOINTS,
)

from bxi_example_py_elf3.framework.inference.api import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.inference.contract import PolicyJointContract
from bxi_example_py_elf3.framework.inference.history import HistoryBuffer
from bxi_example_py_elf3.framework.inference.model import ModelSpec
from bxi_example_py_elf3.framework.inference.policy import JointPolicy
from bxi_example_py_elf3.framework.inference.runtime import (
    InferenceRuntime,
    default_runtime,
)


class HumanoidGaitPolicyLiteIsaaclab(JointPolicy):
    """不带步态输入的AMP行走动作策略管理类"""

    joint_contract = PolicyJointContract(
        observation=ELF3_ISAAC_PARAMETERS.layout,
        action=ELF3_ISAAC_PARAMETERS.layout,
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
        self._parameters = ELF3_ISAAC_PARAMETERS

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
        self._full_count = self._parameters.layout.dof_num

        # The model input width selects the 540D, 960D or 1020D layout.
        self.num_obs = 960  # 默认值，将被动态调整
        self.obs_history_len = 10
        self.single_obs_dim = 96
        self.extra_obs_dim = 0  # 额外观测维度（用于1020维模型），默认0

        self._initialize_model(model, backend)

        self._estimated_velocity = np.zeros(3, dtype=np.float32)
        self.default_target = JointTargetView(
            self.joint_contract.action,
            self._parameters.default_position,
            self._parameters.kp,
            self._parameters.kd,
        )
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
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

        # The input width selects the model's named joint subset.
        if actual_input_dim == 540:
            self.single_obs_dim = 54
            model_layout = ELF3_LOWER_BODY_JOINTS
            self.extra_obs_dim = 0
        elif actual_input_dim == 960:
            self.single_obs_dim = 96
            model_layout = self._parameters.layout
            self.extra_obs_dim = 0
        elif actual_input_dim == 1020:
            self.single_obs_dim = 102
            model_layout = self._parameters.layout
            self.extra_obs_dim = 6
        else:
            raise ValueError(
                f"Unsupported AMP model input dimension: {actual_input_dim}. "
                "Expected 540, 960, or 1020."
            )

        self._model_layout = model_layout
        self._model_binding = CompiledJointMap.compile(
            self._parameters.layout,
            model_layout,
        )
        self._model_indices = self._model_binding.indices
        self._model_parameters = self._parameters.select(model_layout)
        self._model_count = model_layout.dof_num

        # 更新总观测维度
        self.num_obs = self.single_obs_dim * self.obs_history_len
        self._input = np.zeros((1, self.num_obs), dtype=np.float32)
        self._inputs = {"obs": self._input}

        # Initialize variables
        output_dim = int(self._backend.output_shape("actions")[-1])
        self._action = np.zeros(output_dim, dtype=np.float32)
        self._previous_action = np.zeros(self._full_count, dtype=np.float32)
        self._previous_model_action = np.zeros(self._model_count, dtype=np.float32)
        # ``step(..., advance=False)`` is used by running transitions as a
        # side-effect-free preview.  Keep reusable checkpoints so previewing
        # never feeds its inferred action back into the next preview.
        self._previous_action_checkpoint = np.empty_like(self._previous_action)
        self._previous_model_action_checkpoint = np.empty_like(
            self._previous_model_action
        )
        self._target = self._target_buffer.position
        np.copyto(self._target, self._parameters.default_position)
        self._single_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        self._scaled_action = np.empty(self._model_count, dtype=np.float32)
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
            self._parameters.default_position,
            np.zeros_like(self._parameters.default_position),
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
        if not advance:
            np.copyto(self._previous_action_checkpoint, self._previous_action)
            np.copyto(
                self._previous_model_action_checkpoint,
                self._previous_model_action,
            )
        joints = self.bind_joints(frame)
        cmd_vel = frame.command
        if cmd_vel is None:
            raise ValueError("HumanoidGaitPolicyLiteIsaaclab requires frame.command")
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
        raw_action = self._backend.run(self._inputs)["actions"]
        np.copyto(self._action, np.asarray(raw_action).reshape(-1))

        out_len = self._action.shape[0]
        if out_len < self._model_count:
            raise ValueError(
                f"model action dim is {out_len}, expected at least {self._model_count}"
            )

        if out_len >= self._full_count:
            np.copyto(self._previous_action, self._action[: self._full_count])
            if self._model_binding.is_identity:
                controlled_action = self._previous_action
            else:
                np.take(
                    self._previous_action,
                    self._model_indices,
                    out=self._previous_model_action,
                )
                controlled_action = self._previous_model_action
            vel_start = self._full_count
        else:
            np.copyto(
                self._previous_model_action,
                self._action[: self._model_count],
            )
            controlled_action = self._previous_model_action
            self._previous_action.fill(0.0)
            self._previous_action[self._model_indices] = controlled_action
            vel_start = self._model_count

        self._estimated_velocity.fill(0.0)
        if out_len > vel_start:
            vel_len = min(3, out_len - vel_start)
            self._estimated_velocity[:vel_len] = self._action[
                vel_start : vel_start + vel_len
            ]

        np.multiply(
            controlled_action,
            self._model_parameters.action_scale,
            out=self._scaled_action,
        )
        if self._model_binding.is_identity:
            np.add(
                self._parameters.default_position,
                self._scaled_action,
                out=self._target,
            )
        else:
            np.copyto(self._target, self._parameters.default_position)
            np.add(
                self._model_parameters.default_position,
                self._scaled_action,
                out=self._scaled_action,
            )
            self._target[self._model_indices] = self._scaled_action

        if not advance:
            np.copyto(self._previous_action, self._previous_action_checkpoint)
            np.copyto(
                self._previous_model_action,
                self._previous_model_action_checkpoint,
            )

        return self.output

    def reset(self, frame: InferenceFrame) -> None:
        joints = self.bind_joints(frame)
        cmd_vel = frame.command
        if cmd_vel is None:
            raise ValueError("HumanoidGaitPolicyLiteIsaaclab requires frame.command")
        self._action.fill(0.0)
        self._previous_action.fill(0.0)
        self._previous_model_action.fill(0.0)
        self._estimated_velocity.fill(0.0)
        np.copyto(self._command, cmd_vel, casting="unsafe")
        if self.extra_obs_dim > 0:
            self.episode_length_buf = 0
        np.copyto(self._target, self._parameters.default_position)
        self._fill_history(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
            self._command,
        )
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
            estimated_velocity=self._estimated_velocity,
        )

    def _fill_history(self, qj, dqj, quat, omega, cmd_vel):
        single_obs = self._build_single_observation(qj, dqj, quat, omega, cmd_vel)
        self._history.fill(single_obs)
        self._refresh_observation_buffer()

    def _update_observation(self, qj, dqj, quat, omega, cmd_vel, *, advance: bool):
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

        if qj.shape[0] != self._full_count:
            raise ValueError(f"qj dim is {qj.shape[0]}, expected {self._full_count}")
        if dqj.shape[0] != self._full_count:
            raise ValueError(f"dqj dim is {dqj.shape[0]}, expected {self._full_count}")

        # Create single observation with dynamic dimensions
        single_obs = self._single_obs
        single_obs.fill(0.0)

        # 【标准】omega + gravity + cmd_vel + joint_pos + joint_vel + last_action
        single_obs[0:3] = omega  # 3维
        single_obs[3:6] = gravity_orientation  # 3维
        single_obs[6:9] = self._command  # 3维
        joint_position_obs = single_obs[9 : 9 + self._model_count]
        joint_velocity_obs = single_obs[
            9 + self._model_count : 9 + 2 * self._model_count
        ]
        if self._model_binding.is_identity:
            np.subtract(
                qj,
                self._parameters.default_position,
                out=joint_position_obs,
            )
            np.copyto(joint_velocity_obs, dqj)
            obs_last_action = self._previous_action
        else:
            np.take(qj, self._model_indices, out=joint_position_obs)
            joint_position_obs -= self._model_parameters.default_position
            np.take(dqj, self._model_indices, out=joint_velocity_obs)
            obs_last_action = self._previous_model_action
        single_obs[
            9 + 2 * self._model_count : 9 + 3 * self._model_count
        ] = obs_last_action

        # 【兼容】如果需要额外维度（如1020维模型），补零
        if self.extra_obs_dim > 0:
            # 对于102维（1020维模型），在末尾补6维零
            # single_obs[96:102] = 0  (已在初始化时设为0，无需显式赋值)
            single_obs[9 + 3 * self._full_count : 11 + 3 * self._full_count] = np.sin(
                2 * np.pi * self.gait_phase
            )  # 2 #步态相位正弦值
            single_obs[11 + 3 * self._full_count : 13 + 3 * self._full_count] = np.cos(
                2 * np.pi * self.gait_phase
            )  # 2 #步态相位余弦值
            single_obs[
                13 + 3 * self._full_count : 15 + 3 * self._full_count
            ] = self.phase_ratio  # 2 #步态空中比率

        return single_obs

    def _refresh_observation_buffer(self):
        self._history.write_into(self._obs)

    def close(self):
        self._backend.close()
