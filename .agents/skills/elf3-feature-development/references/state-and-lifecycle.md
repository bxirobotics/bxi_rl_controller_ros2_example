# 状态设计与生命周期

## 目录

- [公共 API 边界](#公共-api-边界)
- [选择状态基类](#选择状态基类)
- [生命周期](#生命周期)
- [一个直接状态的骨架](#一个直接状态的骨架)
- [策略状态](#策略状态)
- [ROS callback 与线程边界](#ros-callback-与线程边界)
- [性能与日志](#性能与日志)
- [安全行为](#安全行为)

## 公共 API 边界

状态、Resource 和 Transition 集成只从 `bxi_example_py_elf3.framework.mod_api` 导入；策略还可使用 `bxi_example_py_elf3.framework.inference`。不要导入 `framework.runtime` 内部实现，也不要耦合具体的 `BxiExample` 控制器类。

状态统一继承 `RobotControlState` 或其便捷子类。状态日志使用 `self.logger`；`ctx.ros_node` 只用于创建 ROS 实体。

## 选择状态基类

| 类型 | 开发者主要负责 | 注意点 |
| --- | --- | --- |
| `PoseState` | `target_position()`、增益 | 适合固定姿态；确认从零力矩进入时增益不是意外的 0 |
| `ProceduralState` | `compute_frame(ctx, elapsed)` | 仅真实更新推进时间 |
| `PolicyState` | 策略 reset、进入位置、推理 | 策略必须来自 ResourceHandle |
| `MotionReplayState` | 策略资源、结束状态和切换配置 | 显式给 `finish_state`，复用暂停/完成逻辑 |
| `RobotControlState` | 全部生命周期和输出时序 | 适合传感器、复杂组合或特殊故障处理 |

从便捷类切换到直接实现核心类时，保留 Mod id、本地状态名、事件和 routes，避免破坏外部契约。

## 生命周期

| 方法 | 何时调用 | 应做什么 |
| --- | --- | --- |
| `__init__` | 构建状态 | 保存纯配置、handle 和小型私有数据；不访问 ROS node 或加载模型 |
| `on_bind(ctx)` | 构建后一次 | 创建状态拥有的 subscription、publisher、client、service、timer |
| `is_available(ctx)` | 请求进入前 | 非阻塞、无副作用地检查最新健康快照 |
| `on_prepare(ctx, from_state)` | Transition Session 创建前 | 解析布局、建缓冲、选择动作、reset/preheat 已就绪模型；不发电机命令 |
| `on_prepare_cancel(...)` | 准备后的切换取消 | 撤销 prepare 的运行期副作用 |
| `on_enter(ctx)` | Transition 完成后 | 重置播放、相位和会话变量；确保首帧连续 |
| `on_update(ctx, dt)` | 当前状态每个控制周期 | 读取一致快照并产生本周期唯一电机输出 |
| `on_action(ctx, name)` | action 触发 | 处理后返回 `True`，未知 action 返回 `False` |
| `on_exit(ctx)` | 成功切出 | 结束运行期会话；不要销毁 bind 级实体 |
| `on_unbind(ctx)` | 节点关闭 | 对称销毁 `on_bind` 创建的所有 ROS 实体 |

`on_bind()` 发生在首个机器人状态之前，不能读取 `ctx.robot_layout`。需要布局的缓存和索引放在 `on_prepare()`，或没有 prepare 阶段时首次使用时惰性创建并缓存。

## 一个直接状态的骨架

```python
import numpy as np

from bxi_example_py_elf3.framework.mod_api import (
    EntryFrameProvider,
    RobotControlState,
    RunningFrameProvider,
)


class ExampleState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(self, name, state_id):
        super().__init__(name, state_id)
        self._layout = None
        self._base = None
        self._joint_index = None
        self._elapsed = 0.0

    def on_prepare(self, ctx, from_state):
        layout = ctx.robot_layout
        if self._layout is not layout:
            self._joint_index = layout.index("r_elbow_y_joint")
            self._base = np.empty(layout.dof_num, dtype=np.float32)
            self._layout = layout
        np.copyto(self._base, ctx.last_motor_frame.qpos)

    def on_enter(self, ctx):
        self._elapsed = 0.0

    def _calculate(self, ctx, elapsed):
        last = ctx.last_motor_frame
        frame = self._motor_frame(ctx, self._base, last.kp, last.kd)
        frame.qpos[self._joint_index] = self._base[self._joint_index]
        return frame

    def get_entry_frame(self, ctx):
        return self._calculate(ctx, 0.0)

    def sample_running_frame(self, ctx, dt, *, advance):
        frame = self._calculate(ctx, self._elapsed)
        if advance:
            self._elapsed += dt
        return frame

    def on_update(self, ctx, dt):
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
```

只有 route 选用的 Transition 需要 Provider 时才实现它们。`advance=False` 必须不推进时间、模型 history、相位、播放帧、输入消费游标或其他可观察状态。

## 策略状态

策略应有长期复用的具名 `PolicyOutput`/`JointTargetView`，并把模型 observation/action 的真实布局固化在 policy contract 中。状态从 `ResourceHandle` 获取策略：

```python
class WalkState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(self, name, state_id, policy):
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy

    @property
    def policy(self):
        return self._policy.get()

    def on_prepare(self, ctx, from_state):
        ctx.preheat_model(self.policy, command=self.get_cmd_vel(ctx))

    def get_entry_frame(self, ctx):
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(self, ctx, dt, *, advance):
        self.get_cmd_vel(ctx)
        output = self.policy.step(ctx.inference_frame, dt, advance=advance)
        return self._motor_frame_from_target(ctx, output.joints)
```

固定动作若符合 replay 契约，优先继承 `MotionReplayState`，不要重复实现预热、暂停、完成检测和返回逻辑。

## ROS callback 与线程边界

在 `on_bind()` 创建实体，在 `on_unbind()` 对称释放。callback 不直接发电机命令，不直接修改控制周期正在读取的数组：

```text
ROS/SDK callback -> 校验消息 -> 带锁或双缓冲的最新快照
控制周期         -> 读取一致快照 -> 更新状态长期缓冲 -> 形成 MotorFrame
```

callback 要检查有限数、消息长度、编码、时间戳/新鲜度和业务范围。传感器状态还要区分：

- 尚未收到首帧；
- 已收到但已 stale；
- 格式或内参非法；
- Resource 尚未 ready；
- 外部节点 faulted/unavailable。

进入前由 `is_available()` 拒绝不健康状态；运行中超时则明确请求 normal、pd_brake 或 zero_torque 等经过设计的状态。不要让 Composer、ResourceHandle 或旧缓存替你猜超时策略。

## 性能与日志

- `JointLayout`、索引、映射、Composer 和大数组只在类定义/准备/首次绑定布局时创建。
- 控制周期使用 `np.copyto`、切片赋值和 `out=` 原地操作。
- 不在每周期做文件 IO、模型创建、名称查找、深拷贝、字符串拼接或日志输出。
- 高频异常只记录一次或节流；持续数据放状态快照或统计信息。
- `self.logger` 在 `on_bind()` 前已注入，不使用 `ctx.ros_node.get_logger()` 冒充状态 logger。

## 安全行为

对运动状态明确考虑：姿态异常、传感器断流、模型完成、外部参考丢失、action 暂停、节点启动失败和 Transition 中断。`ctx.request_state()` 返回 `True` 只表示请求已接受或正在准备资源，不表示切换已经完成。

`force=True` 只用于明确的维护/诊断，它只绕过目标 `is_available()`，不会绕过资源、节点、配置和生命周期错误。普通业务状态不要默认使用。
