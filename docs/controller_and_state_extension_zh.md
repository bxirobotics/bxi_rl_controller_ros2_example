# 控制器与状态扩展指南

本文说明当前架构下如何添加新的遥控器/输入控制器，以及如何添加新的机器人业务状态。

当前设计保持 `communication/msg/MotionCommands` 不变。`MotionCommands.btn_1..btn_10` 仍作为兼容传输层，业务代码不直接理解这些数字槽位，而是通过配置把它们映射成有名字的事件。

## 1. 总体架构

数据流如下：

```text
遥控器/键盘原始输入
        |
        v
remote_controller
  - config.cpp: 读取原生 YAML 配置
  - input_mapper.cpp: 统一不同遥控器的按钮、轴、组合键
  - main.cpp: ROS2 发布 MotionCommands
        |
        v
MotionCommands
  - vel_des / yawdot_des
  - btn_1..btn_10
        |
        v
bxi_example_demo.py
  - RemoteEventAdapter: btn_N -> 业务事件名
  - RobotStateMachine: 管理状态、事件、延迟、过渡
  - robot_states.py: 每个状态自己的运行代码
        |
        v
ActuatorCmds
```

相关文件：

- 遥控器配置：`src/remote_controller/config/xbox_default.yaml`
- 遥控器 YAML 解析：`src/remote_controller/src/config.cpp`
- 输入映射：`src/remote_controller/src/input_mapper.cpp`
- 遥控器 ROS 节点：`src/remote_controller/src/main.cpp`
- 状态机配置：`src/bxi_example_py_elf3/config/elf3_state_machine.yaml`
- 状态机引擎：`src/bxi_example_py_elf3/bxi_example_py_elf3/state_machine.py`
- 机器人状态类：`src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py`
- 主控制节点：`src/bxi_example_py_elf3/bxi_example_py_elf3/bxi_example_demo.py`

## 2. 添加新的遥控器/输入控制器

如果新的遥控器仍然是 Linux joystick 设备，例如 `/dev/input/js0`，通常只需要新增一个 YAML 配置文件，不需要改 C++。

### 2.1 先确认按钮和轴编号

可以用系统工具观察手柄事件：

```bash
jstest /dev/input/js0
```

或者：

```bash
evtest
```

记录这些信息：

- 前进/后退轴编号
- 左右平移轴编号
- 转向轴编号
- A/B/X/Y 或 Cross/Circle/Square/Triangle 按钮编号
- LB/RB 或 L1/R1 按钮编号
- Start/Stop 按钮编号

当前约定使用一套“标准控件名”屏蔽物理遥控器差异：

```text
button.south    # Xbox A / PS Cross
button.east     # Xbox B / PS Circle
button.west     # Xbox X / PS Square
button.north    # Xbox Y / PS Triangle
shoulder.left   # LB / L1
shoulder.right  # RB / R1
trigger.left    # LT / L2，作为轴按钮使用
trigger.right   # RT / R2，作为轴按钮使用
system.start
system.stop
```

### 2.2 新增遥控器配置文件

复制默认配置：

```bash
cp src/remote_controller/config/xbox_default.yaml \
   src/remote_controller/config/my_controller.yaml
```

配置结构如下：

```yaml
device:
  js: /dev/input/js0
  vel_offset: 0.0

axes:
  vx:
    index: 3
    direction: -1
    deadzone: 1000.0
    min: -1.0
    max: 1.0
    alpha: 0.03

buttons:
  system.stop: 11
  system.start: 14
  shoulder.left: 6
  shoulder.right: 7
  button.south: 0
  button.east: 1
  button.west: 3
  button.north: 4

modifiers:
  - shoulder.left
  - shoulder.right
  - trigger.left
  - trigger.right

axis_buttons:
  trigger.left:
    index: 5
    direction: 1
    threshold: 0.85
    release_outputs: [btn_10=0]

bindings:
  - output: btn_1
    when: [shoulder.right, button.west]
  - output: btn_10=1
    when: [trigger.left, button.west]
```

字段说明：

- `device.js`：Linux joystick 设备路径。
- `device.vel_offset`：给 `vel_des.x` 增加固定偏置，默认 `0.0`。
- `axes.<name>.index`：轴编号。
- `axes.<name>.direction`：方向，`1` 保持原始方向，`-1` 反向。
- `axes.<name>.deadzone`：死区，原始 joystick 轴范围是 `-32767..32767`。
- `axes.<name>.min/max`：归一化后的速度范围。
- `axes.<name>.alpha`：一阶低通滤波系数，越大响应越快，越小越平滑。
- `buttons`：标准控件名到物理按钮编号的映射。
- `modifiers`：组合键修饰键，例如 `shoulder.left/right`。
- `axis_buttons`：把扳机这类轴输入抽象成“按下/松开”的标准控件。
- `bindings`：标准控件组合到 `MotionCommands` 输出槽位的映射。
- `release_outputs`：轴按钮松开时自动输出，例如松开 LT 后把 `btn_10` 复位为 `0`。

`bindings.when` 的顺序有意义：最后一个控件是触发键，修饰键写在前面。例如：

```yaml
- output: btn_1
  when: [shoulder.right, button.west]
```

含义是：按住 `shoulder.right`，再按下 `button.west` 时触发 `btn_1`。

### 2.3 配置组合键输出

当前默认业务事件约定如下：

```text
btn_1 -> normal
btn_2 -> zero_torque
btn_3 -> pd_brake
btn_4 -> initial_pos
btn_5 -> dance
btn_6 -> recover
btn_7 -> normal_run
btn_8 -> amp_run
btn_9 -> toggle_dance_pause
btn_10 == 1 -> back_flip
btn_10 == 2 -> forward_flip
btn_10 == 3 -> applause
```

这些业务含义不是写在 `remote_controller` 里的，而是在状态机配置里解释：

```yaml
remote_events:
  normal: btn_1
  zero_torque: btn_2
  toggle_dance_pause: btn_9
  back_flip:
    slot: btn_10
    value: 1
  forward_flip:
    slot: btn_10
    value: 2
  applause:
    slot: btn_10
    value: 3
```

所以新增遥控器时，通常只需要保证输出仍然是相同的 `btn_N`。如果一个槽位承载多个业务动作，使用 `btn_N=value`，状态机里用 `slot/value` 区分。

### 2.4 配置键盘映射

键盘配置和手柄配置在同一个 YAML 里：

```yaml
keyboard:
  timeout_us: 150000
  movement:
    forward: w
    backward: s
    yaw_left: a
    yaw_right: d
    strafe_left: q
    strafe_right: e
    stop: space
  bindings:
    "1": btn_1
    "2": btn_6
    "3": btn_5
    "6": btn_10=1
    "7": btn_10=2
    "8": btn_10=3
    "0": btn_10=0
```

移动键直接控制速度轴。`bindings` 里的键会触发 `btn_N`，和手柄输出走同一套状态机。

### 2.5 让 launch 使用新配置

`remote_controller` 不再用 ROS2 parameter YAML 读取遥控器配置，而是通过命令行参数传原生 YAML：

```python
arguments=["--config", remote_config, "__log_level:=debug"]
```

键盘模式：

```python
arguments=["--keyboard", "--config", remote_config, "__log_level:=debug"]
```

你可以新建一个 launch 文件，例如：

```python
import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    remote_config = os.path.join(
        get_package_share_path("remote_controller"),
        "config/my_controller.yaml",
    )

    return LaunchDescription([
        Node(
            package="remote_controller",
            executable="remote_controller",
            name="remote_controller",
            output="screen",
            emulate_tty=True,
            arguments=["--config", remote_config, "__log_level:=debug"],
        )
    ])
```

### 2.6 重新构建和测试

```bash
colcon build --packages-select remote_controller --symlink-install --merge-install
source install/setup.bash
ros2 launch remote_controller remote_controller.launch.py
```

观察输出：

```bash
ros2 topic echo /motion_commands
```

按组合键时，应该能看到对应的 `btn_N` 在 `0/1` 之间切换。

### 2.7 如果不是 Linux joystick 设备

如果输入源不是 `/dev/input/js*`，例如网络手柄、串口遥控器、蓝牙自定义协议，不建议把协议逻辑写进 `InputMapper`。

推荐做法：

1. 保持 `InputMapper` 不变。
2. 新增一个输入读取器，把外部协议转换成标准调用：

```cpp
auto axis_outputs = mapper_.set_axis(axis_index, raw_value);
auto button_outputs = mapper_.handle_button(button_index, pressed);
```

3. 或者直接输出标准业务槽位：

```cpp
mapper_.apply_button_output("btn_1");
```

这样 `MotionCommands` 和 Python 状态机都不用改。

## 3. 添加新的机器人状态

一个状态对应一个 Python 类。用户需要定义进入状态、运行状态、退出状态，以及过渡期间执行的代码。

状态类放在：

```text
src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py
```

状态机基类在：

```text
src/bxi_example_py_elf3/bxi_example_py_elf3/state_machine.py
```

### 3.1 状态生命周期

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
```

调用顺序：

```text
触发切换事件
  -> 当前状态.on_exit()
  -> 目标状态.on_prepare_enter()
  -> 过渡期间每个控制周期:
       当前状态.on_exit_transition(progress)
       目标状态.on_enter_transition(progress)
  -> 目标状态.on_enter()
  -> 目标状态.on_update()
```

说明：

- `on_prepare_enter`：在过渡开始前执行，适合预热模型、准备缓存。
- `on_enter`：真正进入目标状态时执行，适合重置计数器、设置起始帧。
- `on_update`：状态运行时每个控制周期执行。
- `on_exit`：离开当前状态时执行，适合保存最后一帧电机目标。
- `on_exit_transition`：过渡期间，由旧状态执行的代码。
- `on_enter_transition`：过渡期间，由新状态执行的代码。
- `progress`：过渡进度，范围 `0.0..1.0`。
- `transition`：配置里的过渡 profile，例如 `soft_switch`。

如果 `duration: 0.0`，状态机会直接进入新状态，不跑逐帧过渡钩子。

### 3.2 `ctx` 可以使用的常用接口

状态类不要直接发布 ROS topic。状态只设置电机目标，主循环统一发布。

常用数据：

```python
ctx.current_q
ctx.current_dq
ctx.current_quat_xyzw
ctx.current_quat_wxyz
ctx.current_omega
ctx.current_cmd_vel
ctx.loop_count
ctx.dt
```

常用方法：

```python
ctx.set_motor_target(qpos, kp, kd)
ctx.hold_last_motor_target()
ctx.request_state("zero_torque", trigger="safety")
ctx.request_state("normal", trigger="done", transition="soft_switch")
ctx.is_orientation_unsafe(ctx.current_quat_xyzw)
ctx.preheat_model(model, with_cmd_vel=True)
```

常用模型对象：

```python
ctx.normal
ctx.normal_run
ctx.amp_run
ctx.dance
ctx.recover
ctx.back_flip
ctx.forward_flip
ctx.noarm
```

### 3.3 新增状态类示例

假设新增一个 `wave` 状态。

先在 `robot_states.py` 添加状态类：

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

        ctx.dance.timestep += 1
        if ctx.dance.timestep > 300:
            ctx.request_state("normal", trigger="wave_finished", transition="soft_switch")
```

### 3.4 注册状态 ID

在 `bxi_example_demo.py` 的 `robotState` 增加一个 ID：

```python
class robotState:
    normal = 0
    zero_torque = 1
    pd_brake = 2
    initial_pos = 3
    dance = 4
    recover = 5
    amp_run = 6
    normal_run = 7
    back_flip = 8
    forward_flip = 9
    applause = 10
    wave = 11
```

然后在 `BxiExample.__init__` 的 `state_id_by_name` 增加：

```python
self.state_id_by_name = {
    ...
    "wave": robotState.wave,
}
```

### 3.5 注册状态类

在 `robot_states.py` 的 `build_robot_states()` 里增加：

```python
def build_robot_states(state_ids):
    return {
        ...
        "wave": WaveState("wave", state_ids["wave"]),
    }
```

### 3.6 配置状态转移

编辑：

```text
src/bxi_example_py_elf3/config/elf3_state_machine.yaml
```

新增事件映射。如果使用 `btn_10=4` 触发：

```yaml
remote_events:
  wave:
    slot: btn_10
    value: 4
```

新增状态：

```yaml
states:
  wave:
    behavior: WaveState
    transitions:
      on_event:
        normal:
          to: normal
          transition: soft_switch
        zero_torque: zero_torque
      after:
        - seconds: 5.0
          to: normal
          transition: soft_switch
```

给其他状态增加进入 `wave` 的转移：

```yaml
states:
  normal:
    transitions:
      on_event:
        wave:
          to: wave
          transition: soft_switch
```

支持的写法：

```yaml
# 简写：收到事件后立即切换
normal: normal

# 指定过渡配置
dance:
  to: dance
  transition: soft_switch

# 按键后延迟 0.5 秒再切换
zero_torque:
  to: zero_torque
  delay: 0.5
  transition: soft_switch

# 只执行 action，不切换状态
toggle_dance_pause:
  action: toggle_dance_pause

# 进入状态后 5 秒自动切换
after:
  - seconds: 5.0
    to: normal
    transition: soft_switch
```

### 3.7 配置过渡时间和过渡行为

过渡 profile 在 YAML 顶部定义：

```yaml
transition_profiles:
  soft_switch:
    duration: 0.2
    exit_behavior: hold_last_motor
    enter_behavior: hold_last_motor
```

当前内置行为：

- `hold_last_motor`：过渡期间保持最后一次电机目标。
- `none`：不做额外处理。

如果需要更复杂的过渡，比如关节位置插值，可以在状态类里重写：

```python
def on_enter_transition(self, ctx, from_state, progress, transition):
    qpos = ctx.pos_last_state + (self.target_pos - ctx.pos_last_state) * progress
    ctx.set_motor_target(qpos, self.kp, self.kd)
```

### 3.8 添加新 action

如果配置里写：

```yaml
toggle_wave_pause:
  action: toggle_wave_pause
```

优先在对应状态类里处理，让状态自己的私有变量留在状态内部：

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

`on_action()` 返回 `True` 表示这个 action 已被当前状态处理；返回 `False` 时状态机会继续查全局 `action_handlers`，仍没有处理器就报错。

## 4. 把新遥控器事件接到新状态

如果新增状态 `wave` 也想复用 `btn_10`，应该使用一个新的值，而不是覆盖已有的 `1/2/3`：

1. 遥控器配置输出 `btn_10=4`，并在释放动作时复位：

```yaml
bindings:
  - output: btn_10=4
    when: [button.north]
```

2. 状态机配置解释 `btn_10`：

```yaml
remote_events:
  wave:
    slot: btn_10
    value: 4
```

3. 状态转移表允许进入 `wave`：

```yaml
states:
  normal:
    transitions:
      on_event:
        wave:
          to: wave
          transition: soft_switch
```

这三层分开后，换遥控器只改 `remote_controller/config/*.yaml`；改业务状态只改 `robot_states.py` 和 `elf3_state_machine.yaml`。

## 5. 构建与验证

修改 Python 状态后：

```bash
python3 -m py_compile \
  src/bxi_example_py_elf3/bxi_example_py_elf3/state_machine.py \
  src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py \
  src/bxi_example_py_elf3/bxi_example_py_elf3/bxi_example_demo.py
```

修改遥控器 C++ 后：

```bash
colcon build --packages-select remote_controller --symlink-install --merge-install
```

修改状态机 Python 包后：

```bash
colcon build --packages-select bxi_example_py_elf3 --symlink-install --merge-install
```

一起构建：

```bash
colcon build --packages-select remote_controller bxi_example_py_elf3 --symlink-install --merge-install
```

如果工作区里保留了 `update/` 参考目录，里面也有同名 ROS 包，colcon 默认全目录扫描会报 duplicate package。此时只构建当前源码目录：

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

观察遥控输出：

```bash
ros2 topic echo /motion_commands
```

## 6. 常见问题

### 按组合键没有触发

检查 `bindings.when` 顺序。最后一个控件必须是触发键：

```yaml
# 正确
when: [shoulder.right, button.west]

# 不推荐
when: [button.west, shoulder.right]
```

### 按普通键时误触发组合键

确认组合键修饰键写进了 `modifiers`：

```yaml
modifiers:
  - shoulder.left
  - shoulder.right
```

没有写进 `modifiers` 时，输入映射层无法区分“普通键”和“组合键上下文”。

### 状态配置写了但没有生效

检查三处是否一致：

1. `robotState` 里有状态 ID。
2. `state_id_by_name` 里有状态名。
3. `build_robot_states()` 里注册了同名状态类。

### 状态运行了但没有电机输出

状态的 `on_update()` 或过渡钩子里必须调用：

```python
ctx.set_motor_target(qpos, kp, kd)
```

主循环只会发布 `ctx.motor_target`，状态类里不要直接调用 `send_to_motor()`。

### 自动切换没有触发

确认写在当前状态下面：

```yaml
states:
  recover:
    transitions:
      after:
        - seconds: 5.0
          to: normal
```

`after` 的计时从进入当前状态后开始。离开再进入会重新计时。

### yaml-cpp 找不到

`remote_controller` 现在依赖 `yaml-cpp`。Ubuntu 22.04 通常可以安装：

```bash
sudo apt install libyaml-cpp-dev
```

然后重新构建：

```bash
colcon build --packages-select remote_controller --symlink-install --merge-install
```
