# 控制器与状态扩展指南

本文说明如何添加新的遥控器/输入控制器，以及如何添加新的机器人业务状态。

`communication/msg/MotionCommands` 不改。`btn_1..btn_10` 只是兼容传输层，业务层通过状态机配置把它们解释成有名字的事件。

## 1. 总体架构

配置按人读起来的顺序分三层：

```text
sources   声明输入从哪里来：手柄、键盘、CRSF、串口、UDP
controls  把 source 解释成统一控件：analog / bool / enum
outputs   把 control 条件转成 MotionCommands：analog / level / edge
```

运行时数据流：

```text
joystick / keyboard / CRSF / serial / UDP
        |
        v
source: js.axis.3 / gamepad.x / keyboard.normal / crsf.ch5
        |
        v
control: move.vx / button.west / trigger.left / mode.switch
        |
        v
output: vel_des / yawdot_des / btn_N / system.action
        |
        v
RobotStateMachine
```

相关文件：

- 遥控器配置：`src/remote_controller/config/xbox_default.yaml`
- 遥控器 YAML 解析：`src/remote_controller/src/config.cpp`
- 输入映射：`src/remote_controller/src/input_mapper.cpp`
- 遥控器节点：`src/remote_controller/src/main.cpp`
- 状态机配置：`src/bxi_example_py_elf3/config/elf3_state_machine.yaml`
- 状态机引擎：`src/bxi_example_py_elf3/bxi_example_py_elf3/state_machine.py`
- 机器人状态类：`src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py`

`remote_controller` 的核心解析和映射代码已经做成 `remote_controller_core` 库。以后写 CRSF、串口、UDP 节点，可以复用 `RemoteConfig` 和 `InputMapper`，只需要把协议数据喂成 source。

## 2. 遥控器配置

配置文件顺序建议保持：

```yaml
sources:
  ...

controls:
  ...

outputs:
  ...

system:
  ...
```

### 2.1 sources

`sources` 只声明输入来源和别名，不写业务含义。

Linux joystick 示例：

```yaml
sources:
  gamepad:
    type: joystick
    device: /dev/input/js0
    axes:
      left_y: 3
      trigger_left: 5
    buttons:
      rb: 7
      x: 3
```

字段：

- `sources.<name>`：输入源分组名，例如 `gamepad`、`keyboard`、`crsf`。
- `sources.<name>.type`：输入源类型。内置支持 `joystick` 和 `keyboard`；其他类型可作为自定义 source 声明使用。
- `sources.<name>.device` / `sources.<name>.js`：joystick 设备路径。
- `sources.<name>.axes.<alias>`：轴别名。joystick 下可以写数字 `3`，等价于 `js.axis.3`。
- `sources.<name>.buttons.<alias>`：按钮别名。joystick 下可以写数字 `7`，等价于 `js.button.7`。
- `sources.<name>.signals.<alias>`：自定义输入源的 source 别名，例如 `crsf.ch5`。

这些别名后续用 `<分组名>.<别名>` 引用，例如 `gamepad.left_y`、`gamepad.x`。

键盘示例：

```yaml
sources:
  keyboard:
    type: keyboard
    poll_timeout_us: 20000
    hold_ms: 200
    axes:
      vx: keyboard.axis.vx
      vy: keyboard.axis.vy
      yaw: keyboard.axis.yaw
    movement:
      forward: w
      backward: s
      yaw_left: a
      yaw_right: d
      strafe_left: q
      strafe_right: e
      stop: space
    keys:
      normal: {key: "1", source: key.normal}
```

键盘字段：

- `poll_timeout_us`：读取键盘的轮询超时，单位微秒。
- `hold_ms`：terminal 键盘没有真实释放事件，按一次键后 source 保持按下的毫秒数。
- `axes`：给键盘模拟移动轴起别名。
- `movement.forward/backward/yaw_left/yaw_right/strafe_left/strafe_right/stop`：移动键。
- `keys.<alias>.key`：键盘按键。
- `keys.<alias>.source`：按下该键时产生的 source。

CRSF 示例：

```yaml
sources:
  crsf:
    type: crsf
    signals:
      throttle: crsf.ch1
      mode: crsf.ch5
```

配置解析器只负责声明别名。真正读取 CRSF 协议的节点需要调用：

```cpp
mapper.set_signal("crsf.ch1", normalized_ch1);
mapper.set_signal("crsf.ch5", normalized_ch5);
```

### 2.2 controls

`controls` 把 source 解释成统一控件。业务绑定只看 control，不关心它来自 Xbox、键盘还是 CRSF。

字段：

- `controls.<name>`：控件名，例如 `move.vx`、`button.west`、`mode.switch`。
- `type`：`analog`、`bool`、`enum`。
- `source`：单个 source 或 source 别名。
- `sources`：多个 source。多个输入同时存在时，取绝对值最大的输入。
- `direction`：方向，默认 `1`，写 `-1` 可反向。
- `scale`：倍率，默认 `1.0`。

`analog` 字段：

- `deadzone`：死区。source 已归一化，通常写 `0.02..0.05`。
- `min`：负方向最大输出。
- `max`：正方向最大输出。
- `alpha`：一阶低通滤波系数。

`bool` 字段：

- `threshold`：按下阈值。
- `hysteresis`：迟滞量，避免模拟通道在阈值附近抖动。

`enum` 字段：

- `positions`：档位表，每个档位写 `[min, max]`。
- `hysteresis`：档位迟滞量。

示例：

```yaml
controls:
  move.vx:
    type: analog
    sources:
      - source: gamepad.left_y
        direction: -1
      - source: keyboard.vx
    deadzone: 0.03
    min: -1.0
    max: 1.0
    alpha: 0.03

  trigger.left:
    type: bool
    source: gamepad.trigger_left
    threshold: 0.85
    hysteresis: 0.05

  mode.switch:
    type: enum
    source: crsf.mode
    hysteresis: 0.05
    positions:
      low: [-1.0, -0.35]
      mid: [-0.35, 0.35]
      high: [0.35, 1.0]
```

### 2.3 outputs

`outputs` 是最终输出层，分三类：

- `outputs.analog`：连续量，写到 `MotionCommands.vel_des` 或 `yawdot_des`。
- `outputs.level`：电平量，条件满足就保持，条件不满足就回 `0`。`btn_*` 推荐用这个。
- `outputs.edge`：边沿量，条件从不满足变成满足时触发一次。`system.*` 必须用这个。

analog 示例：

```yaml
outputs:
  analog:
    vx: move.vx
    vy: move.vy
    yaw: move.yaw
```

analog 字段：

- `outputs.analog.vx`：写到 `MotionCommands.vel_des.x`。
- `outputs.analog.vy`：写到 `MotionCommands.vel_des.y`。
- `outputs.analog.yaw`：写到 `MotionCommands.yawdot_des`。
- 值可以直接写 control，也可以写 `{control: move.vx, offset: 0.0}`。
- 如果写 `controls: [...]`，多个 control 取绝对值最大的值。

level / edge 示例：

```yaml
outputs:
  edge:
    - output: system.start
      when: [system.start]

  level:
    - output: btn_1=1
      when:
        any:
          - [shoulder.right, button.west]
          - [key.normal]
```

binding 字段：

- `output`：支持 `btn_N`、`btn_N=value`、`system.<action>`。
- `when`：触发条件。
- `mode`：可选，通常不需要写；放在 `outputs.level` 下默认是 `level`，放在 `outputs.edge` 下默认是 `edge`。

条件写法：

- `button.west`：等价于 `{pressed: button.west}`。
- `mode.switch=high`：enum control 等于 `high`。
- `{pressed: button.west}`：control 被按下。
- `{released: shoulder.right}`：control 未按下。
- `{equals: {control: mode.switch, value: high}}`：enum control 等于指定值。
- `{range: {control: throttle, min: 0.2, max: 1.0}}`：analog control 在范围内。
- `when: [a, b]`：所有条件都满足。
- `when: {all: [a, b]}`：所有条件都满足。
- `when: {any: [[a, b], [c]]}`：任意一组条件满足。

### 2.4 btn_N 变化逻辑

`btn_N` 是 level 输出，不自动发布一帧后清零，也不需要 `release_outputs`。

规则：

- `output: btn_N` 等价于 `output: btn_N=1`。
- 条件满足时，`MotionCommands.btn_N` 保持指定值。
- 条件不满足时，`MotionCommands.btn_N` 自动变为 `0`。
- 扳机、三档开关、键盘键都只是 control，走同一套规则。

例子：

```text
RB + X 按住      -> btn_1 = 1
RB 或 X 松开     -> btn_1 = 0

LT + X 按住      -> btn_10 = 1
LT 或 X 松开     -> btn_10 = 0
```

状态机按边沿触发业务事件：

```text
btn_10: 0 -> 3  触发 applause
btn_10: 3 -> 3  不重复触发
btn_10: 3 -> 0  不触发 applause
btn_10: 0 -> 3  再次触发 applause
```

如果其他节点直接发布 `MotionCommands`，也应该遵守这个 level 语义：条件满足时保持非零，条件不满足时设回 `0`。

### 2.5 system

`system` 定义 `system.<action>` 要执行的命令。

字段：

- `system.<action>`：shell 命令列表。
- `system_mutexes.<name>.acquire`：获取互斥锁的 action。
- `system_mutexes.<name>.release`：释放互斥锁的 action。
- `system_reset_motion_after`：这些 action 执行后清空遥控运动输出。

示例：

```yaml
outputs:
  edge:
    - output: system.stop
      when: [system.stop]

system:
  stop:
    - "killall -SIGINT hardware_elf3"
    - "killall -SIGINT bxi_example_py_elf3"

system_mutexes:
  robot_process:
    acquire: start
    release: stop
```

## 3. 添加非 joystick 输入源

复杂输入源建议做独立节点，例如 `crsf_remote_controller`、`udp_remote_controller`。

原则：

1. 新节点负责读取协议。
2. 把协议通道归一化为 `-1.0..1.0` 或 `0.0/1.0` source。
3. 复用 `remote_controller_core::InputMapper`。
4. 发布同一个 `/motion_commands`。

CRSF 三档开关示例：

```yaml
sources:
  crsf:
    type: crsf
    signals:
      throttle: crsf.ch1
      mode: crsf.ch5

controls:
  throttle:
    type: analog
    source: crsf.throttle
    deadzone: 0.02
    min: -1.0
    max: 1.0
    alpha: 0.05

  mode.switch:
    type: enum
    source: crsf.mode
    hysteresis: 0.05
    positions:
      low: [-1.0, -0.35]
      mid: [-0.35, 0.35]
      high: [0.35, 1.0]

outputs:
  level:
    - output: btn_10=1
      when: [mode.switch=high]
```

节点伪代码：

```cpp
auto outputs = mapper.set_signal("crsf.ch1", normalize(channel_1));
dispatch_outputs(outputs);

outputs = mapper.set_signal("crsf.ch5", normalize(channel_5));
dispatch_outputs(outputs);

communication::msg::MotionCommands msg;
mapper.fill_message(msg);
publisher->publish(msg);
```

不要让两个节点同时发布同一个 `/motion_commands`。

## 4. 当前能力边界

已经实现：

- source 别名声明。
- joystick source 自动归一化。
- keyboard `hold_ms` 模拟释放。
- `analog`、`bool`、`enum` control。
- 多 source 混合，策略是取绝对值最大的输入。
- `outputs.analog`、`outputs.level`、`outputs.edge`。
- `when` 支持 `all`、`any`、`pressed`、`released`、`equals`、`range`。

没有内置实现：

- CRSF 协议读取节点本身。
- 串口/UDP 协议读取节点本身。
- 更复杂的混合策略，例如加权求和、优先级锁定、曲线映射。

这些可以继续在 `remote_controller_core` 上扩展，不需要改业务状态机。

## 5. 添加新的机器人状态

一个机器人状态对应一个 Python 类。状态类放在：

```text
src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py
```

状态机配置在：

```text
src/bxi_example_py_elf3/config/elf3_state_machine.yaml
```

### 5.1 状态机配置字段

`initial_state`：

- 状态机初始化后进入的第一个状态名。必须是 `states` 下已定义的状态。

`remote_events`：

- `remote_events.<event>.slot`：读取哪个 `MotionCommands.btn_N`。
- `remote_events.<event>.value`：期望值。只有槽位从其他值变到该值时才触发事件。
- `remote_events.<event>: btn_N`：兼容旧简写，不建议新配置使用。

示例：

```yaml
remote_events:
  normal:
    slot: btn_1
    value: 1
  applause:
    slot: btn_10
    value: 3
```

`transition_profiles`：

- `transition_profiles.<profile>.duration`：过渡时长，单位秒。`0.0` 表示立即切换。
- `transition_profiles.<profile>.exit_behavior`：过渡期间旧状态默认行为。
- `transition_profiles.<profile>.enter_behavior`：过渡期间新状态默认行为。
- `hold_last_motor`：保持上一帧电机目标。
- `none`：不做额外处理。

`speed_profiles`：

- `speed_profiles.<profile>.vx_scale`：`vel_des.x` 倍率。
- `speed_profiles.<profile>.vx_min`：缩放后的 `vx` 下限。
- `speed_profiles.<profile>.vx_max`：缩放后的 `vx` 上限。
- `speed_profiles.<profile>.vy_scale`：`vel_des.y` 倍率。
- `speed_profiles.<profile>.yaw_scale`：`yawdot_des` 倍率。

`states`：

- `states.<state>.behavior`：状态类名，必须在 `robot_states.py` 中存在。
- `states.<state>.id`：可选固定数字 ID。通常不要写，不写时按 YAML 顺序自动分配。
- `states.<state>.params`：可选构造参数，会作为关键字参数传给状态类。
- `states.<state>.speed_profile`：可选速度 profile 名。
- `states.<state>.transitions.on_event`：事件触发转移表。
- `states.<state>.transitions.on_event.<event>: <target>`：简写，立即切换。
- `states.<state>.transitions.on_event.<event>.to`：目标状态。
- `states.<state>.transitions.on_event.<event>.transition`：过渡 profile，默认 `instant`。
- `states.<state>.transitions.on_event.<event>.delay`：延迟多少秒后执行。
- `states.<state>.transitions.on_event.<event>.action`：只执行 action，不切换状态。
- `states.<state>.transitions.after`：进入该状态后自动触发的规则列表。
- `states.<state>.transitions.after[].seconds`：进入该状态多少秒后触发；也支持写成 `after`。
- `states.<state>.transitions.after[].to`：自动转移目标状态。
- `states.<state>.transitions.after[].transition`：自动转移使用的过渡 profile。
- `states.<state>.transitions.after[].action`：到时只执行 action，不切换状态。

### 5.2 状态生命周期

状态类可以实现这些方法：

```python
class MyState(RobotControlState):
    def on_prepare_enter(self, ctx, from_state, transition):
        pass

    def on_enter(self, ctx):
        pass

    def on_update(self, ctx, dt):
        pass

    def on_exit(self, ctx):
        pass

    def on_exit_transition(self, ctx, to_state, progress, transition):
        pass

    def on_enter_transition(self, ctx, from_state, progress, transition):
        pass

    def on_action(self, ctx, action_name):
        return False
```

调用顺序：

```text
触发切换事件
  -> 旧状态.on_exit()
  -> 新状态.on_prepare_enter()
  -> 过渡期间每个控制周期:
       旧状态.on_exit_transition(progress)
       新状态.on_enter_transition(progress)
  -> 新状态.on_enter()
  -> 新状态.on_update()
```

如果 `duration: 0.0`，状态机会直接进入新状态，不跑逐帧过渡钩子。

### 5.3 新增状态示例

新增 `WaveState`：

```python
class WaveState(RobotControlState):
    def on_prepare_enter(self, ctx, from_state, transition) -> None:
        ctx.preheat_model(ctx.dance)

    def on_enter(self, ctx) -> None:
        self.reset_loop(ctx)
        ctx.dance.timestep = 100

    def on_update(self, ctx, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("zero_torque", trigger="safety")
            return

        qpos = ctx.dance.inference_step(
            ctx.current_q,
            ctx.current_dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )
        ctx.set_motor_target(qpos, ctx.dance.stiffness_array, ctx.dance.damping_array)
```

注册状态：

```yaml
states:
  wave:
    behavior: WaveState
```

新增遥控事件：

```yaml
remote_events:
  wave:
    slot: btn_10
    value: 5
```

允许从 `normal` 进入：

```yaml
states:
  normal:
    transitions:
      on_event:
        wave:
          to: wave
          transition: soft_switch
```

遥控器绑定：

```yaml
outputs:
  level:
    - output: btn_10=5
      when: [mode.switch=high]
```

### 5.4 action

如果配置里写：

```yaml
toggle_wave_pause:
  action: toggle_wave_pause
```

优先在状态类内部处理，让状态私有变量留在自己内部：

```python
class WaveState(RobotControlState):
    def __init__(self, name, state_id):
        super().__init__(name, state_id)
        self.paused = False

    def on_action(self, ctx, action_name):
        if action_name != "toggle_wave_pause":
            return False

        self.paused = not self.paused
        return True
```

`on_action()` 返回 `True` 表示 action 已被该状态处理；返回 `False` 时状态机会继续查全局 `action_handlers`。

## 6. 构建与验证

构建：

```bash
colcon build \
  --paths src/remote_controller src/bxi_example_py_elf3 \
  --packages-select remote_controller bxi_example_py_elf3 \
  --symlink-install --merge-install
```

启动遥控器：

```bash
source install/setup.bash
ros2 launch remote_controller remote_controller.launch.py
```

键盘模式：

```bash
ros2 launch remote_controller remote_controller_keyboard.launch.py
```

观察输出：

```bash
ros2 topic echo /motion_commands
```
