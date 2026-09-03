# 关节命令、Transition 与输入

## 目录

- [名称是语义主键](#名称是语义主键)
- [MotorFrame 是完整 MIT 命令](#motorframe-是完整-mit-命令)
- [多来源合成](#多来源合成)
- [Transition 选择](#transition-选择)
- [自定义 Transition](#自定义-transition)
- [Route、action 和主动请求](#routeaction-和主动请求)
- [速度和遥控输入](#速度和遥控输入)
- [平台 override 与 State Composer](#平台-override-与-state-composer)

## 名称是语义主键

模型、策略、状态、机器人消息和硬件协议的布局是不同边界：

```text
Incoming Message Layout
  -> Robot Layout
  -> Policy/Model Observation Layout
  -> Model/Policy Action Layout
  -> State Output Layout
  -> 完整 Robot MotorFrame
  -> 可选 Hardware Layout
```

它们不要求关节数相等或顺序一致。只在准备/绑定阶段按名称编译数字索引，控制周期只用缓存后的映射。禁止“取前 29 个”“最后两个是头部”或把 MuJoCo 顺序当模型顺序。

第一条合法具名状态建立稳定 Robot Layout。进程运行中关节集合改变应停止并重启，不动态猜测新布局。

## MotorFrame 是完整 MIT 命令

五个等长 `float32` 字段始终存在：

| MotorFrame | MIT | 含义 |
| --- | --- | --- |
| `qpos` | `p_des` | 目标位置 |
| `vel` | `v_des` | 目标速度 |
| `kp` | `kp` | 位置增益 |
| `kd` | `kd` | 速度增益 |
| `torque` | `t_ff` | 前馈力矩 |

状态只给 `qpos/kp/kd` 时，`vel/torque` 每次明确为 0，不是 `None`，也不继承旧帧。

状态输出交给 `ctx.set_motor_target()` 后按名称解析：

- 同布局走 fast path；
- 同集合不同顺序则重排；
- 状态输出少于机器人时，平台必须为每个缺失硬件关节提供 `JointCommandDefaults`，其中 `position/kp/kd` 都显式给出；
- 状态输出多于机器人时裁剪额外输出，并对新布局 warning 一次；
- Defaults 是固定平台兜底，不能承载话题、IK 或轨迹等动态命令。

## 多来源合成

每个生产者维护自己的 `JointTargetBuffer`，Composer 只合成：

```python
from bxi_example_py_elf3.framework.mod_api import (
    JointCommandComposer,
    JointCommandLayer,
    JointLayout,
    JointTargetBuffer,
)


HEAD = JointLayout(("head_y_joint", "head_z_joint"))
self._head = JointTargetBuffer(HEAD)
self._head.kp[:] = (16.747, 16.747)
self._head.kd[:] = (1.066, 1.066)

self._composer = JointCommandComposer(
    STATE_OUTPUT,
    (
        JointCommandLayer("policy", self.policy.output.joints),
        JointCommandLayer("arm_override", self._arm.view, override=True),
        JointCommandLayer("head", self._head.view),
    ),
)
```

构造时会检查：输出布局被完整覆盖、Layer 不含额外关节、普通 Layer 不重叠、覆盖显式授权、每个来源有完整 position/kp/kd。

Layer、Buffer 和 Composer 在 prepare 阶段创建。callback 先更新线程安全快照，控制周期再把快照复制到 Buffer；不要让 callback 与 `compose()` 并发写同一数组。

## Transition 选择

Transition 处理 route 已决定的两状态之间临时电机输出：

| 类型 | 用途 | 状态能力 |
| --- | --- | --- |
| `instant` | 明确立即切换，常见于紧急断力 | 无 |
| `hold` | 短暂保持最后完整输出 | 无 |
| `entry_gain_ramp` | 朝目标进入帧建立增益 | 目标 `EntryFrameProvider` |
| `running_blend` | 混合两端动态运行帧 | 目标 Entry；被采样侧 Running |
| `sequence` | 顺序组合多个步骤 | 各步骤对应能力 |

常用 profile：

```yaml
transition_profiles:
  safe_entry:
    type: sequence
    steps:
      - type: hold
        duration: 0.02
      - type: entry_gain_ramp
        duration: 0.8
        kp_from: zero
        kd_from: target
```

`running_blend` 混合 `qpos/vel/kp/kd/torque`。两侧自然帧可能是 29、31 或任意 N，必须先分别 `ctx.resolve_motor_frame()` 到完整 Robot Layout，才能逐数组插值。

`sample_running_frame(ctx, dt, advance=False)` 必须是无副作用预览。若 Transition 让目标 `advance_to: true`，目标 `on_enter()` 不能又把时间/history 重置到旧帧而造成跳变；有这种重置时在 route 中改为 `advance_to: false` 并检查完成帧与第一运行帧连续。

来源动作完成后已经无法生成有效帧时，使用 `sample_from: false` 从切换前最后完整帧开始混合。

## 自定义 Transition

只有内置类型不能表达需求时才编写。使用 `ConfigReader` 严格读取字段并调用 `finish()`；用 `validate_states()` 检查 Provider；Session 数据在 `on_start()` 初始化；通过同一 `plugin:create_mod` 返回值的 `transition_plugins` 显式注册。

自定义类型名包含 Mod id，避免全局冲突。不要仅靠 import 副作用注册，也不要为同一 Mod 定义第二个 `create_mod()`。

## Route、action 和主动请求

- route：切换状态，可配置 transition 和 delay；
- action：调用当前状态 `on_action()`，不切换状态；
- `ctx.request_state()`：状态逻辑主动请求完成/故障切换。

`request_state()` 返回 `True` 表示请求被接受、排队或正在准备 Resource，不表示 Transition 已完成；返回 `False` 表示目标不可用或目标节点启动失败。配置错误仍应抛出，不被布尔值掩盖。

## 速度和遥控输入

状态只消费抽象速度和事件：

```text
设备 Driver -> raw signals -> controls -> outputs
             -> MotionCommands -> Mod event/route/action
```

状态不要知道具体键盘、Xbox 或 CRSF 按键。`speed_profile` 配置缩放和限幅，状态调用 `self.get_cmd_vel(ctx)`；需要状态特有滤波时覆盖 `process_cmd_vel()`。

组合键和设备差异应修改 remote controller YAML，而不是写新 Driver。只有接入全新协议/设备时才扩展 Driver；Driver 只发布 raw signals，不直接知道 Mod 状态名。

## 平台 override 与 State Composer

运行时 `/simulation|hardware/actuators_cmds_override` 适合诊断新增关节、方向或外部控制器。长期产品逻辑应放在 State 内，用 Composer 明确所有权，使它参与状态生命周期和 Transition。不要把全局 override 当作 Mod 的长期接口。
