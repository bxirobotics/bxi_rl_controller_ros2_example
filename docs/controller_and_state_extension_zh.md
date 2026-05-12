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

### 遥控器配置字段完整说明

本节逐项说明 `src/remote_controller/config/xbox_default.yaml` 当前使用到的所有字段。

`device`：

- `device.js`：Linux joystick 设备路径。默认是 `/dev/input/js0`，如果系统里手柄是 `js1/js2`，改这里即可。
- `device.vel_offset`：发布到 `MotionCommands.vel_des.x` 前额外叠加的速度偏置。通常保持 `0.0`。

`axes`：

- `axes.vx`：前进/后退速度轴配置，最终写到 `MotionCommands.vel_des.x`。
- `axes.vy`：左右平移速度轴配置，最终写到 `MotionCommands.vel_des.y`。
- `axes.yaw`：转向角速度轴配置，最终写到 `MotionCommands.yawdot_des`。
- `axes.<axis>.index`：物理 joystick 轴编号。
- `axes.<axis>.direction`：轴方向，`1` 表示保持设备原始方向，`-1` 表示反向。
- `axes.<axis>.deadzone`：死区阈值，原始轴范围按 `-32767..32767` 处理，小于死区的输入当作 0。
- `axes.<axis>.min`：归一化输入为负时的最小输出幅度。例如 `axes.vx.min: -1.0` 表示最大后退速度是 `-1.0`。
- `axes.<axis>.max`：归一化输入为正时的最大输出幅度。
- `axes.<axis>.alpha`：一阶低通滤波系数。越大响应越快，越小越平滑。

当前轴配置：

| 字段 | index | direction | deadzone | min | max | alpha | 输出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `axes.vx` | 3 | -1 | 1000.0 | -1.0 | 1.0 | 0.03 | `vel_des.x` |
| `axes.vy` | 0 | -1 | 1000.0 | -1.0 | 1.0 | 0.03 | `vel_des.y` |
| `axes.yaw` | 6 | -1 | 1000.0 | -1.0 | 1.0 | 0.05 | `yawdot_des` |

`buttons`：

- `buttons.<control>`：把物理按钮编号映射成标准控件名。后续 `bindings.when` 只使用标准控件名，不直接写数字。
- `buttons.system.stop`：停止机器人相关进程的按钮。
- `buttons.system.start`：启动机器人相关进程的按钮。
- `buttons.shoulder.left` / `buttons.shoulder.right`：左右肩键，默认对应 Xbox `LB/RB`。
- `buttons.button.south`：南向按钮，默认对应 Xbox `A`。
- `buttons.button.east`：东向按钮，默认对应 Xbox `B`。
- `buttons.button.west`：西向按钮，默认对应 Xbox `X`。
- `buttons.button.north`：北向按钮，默认对应 Xbox `Y`。

当前按钮配置：

| 控件名 | 物理编号 | 默认含义 |
| --- | ---: | --- |
| `system.stop` | 11 | 停止相关进程 |
| `system.start` | 14 | 启动相关进程 |
| `shoulder.left` | 6 | Xbox LB |
| `shoulder.right` | 7 | Xbox RB |
| `button.south` | 0 | Xbox A |
| `button.east` | 1 | Xbox B |
| `button.west` | 3 | Xbox X |
| `button.north` | 4 | Xbox Y |

`modifiers`：

- `modifiers`：修饰键列表。这里面的控件按下时，普通单键绑定会被屏蔽，只匹配带修饰键的组合键。
- 当前配置包含 `shoulder.left`、`shoulder.right`、`trigger.left`、`trigger.right`。

`axis_buttons`：

- `axis_buttons.<control>`：把某个轴包装成“按下/松开”的标准控件名，例如 `trigger.left`。
- `axis_buttons.<control>.index`：物理轴编号。
- `axis_buttons.<control>.direction`：轴方向，先乘这个方向再判断阈值。
- `axis_buttons.<control>.threshold`：按下阈值。归一化后的轴值大于等于该值时认为控件按下。
- `axis_buttons.<control>.release_outputs`：轴按钮从按下变为松开时自动输出的命令列表。当前用 `[btn_10=0]` 复位翻转/鼓掌命令槽。

当前轴按钮配置：

| 控件名 | index | direction | threshold | release_outputs |
| --- | ---: | ---: | ---: | --- |
| `trigger.left` | 5 | 1 | 0.85 | `[btn_10=0]` |
| `trigger.right` | 4 | 1 | 0.85 | `[btn_10=0]` |

`bindings`：

- `bindings`：手柄按钮/组合键到输出命令的列表。
- `bindings[].output`：触发后产生的输出命令。
- `bindings[].when`：触发条件。列表最后一个控件是触发键，前面的控件是必须已按住的修饰/组合控件。
- `output: system.stop`：执行 `system.stop_commands`。
- `output: system.start`：执行 `system.start_commands`。
- `output: btn_N`：切换 `MotionCommands.btn_N` 的 0/1 状态。
- `output: btn_N=value`：把 `MotionCommands.btn_N` 设置成指定整数值。当前 `btn_10=1/2/3` 分别表示后空翻、前空翻、鼓掌。

当前手柄绑定含义：

| output | when | 业务含义 |
| --- | --- | --- |
| `system.stop` | `[system.stop]` | 执行 `stop_commands` |
| `system.start` | `[system.start]` | 执行 `start_commands` |
| `btn_1` | `[shoulder.right, button.west]` | RB + X，进入 `normal` |
| `btn_2` | `[shoulder.right, button.south]` | RB + A，进入 `zero_torque` |
| `btn_3` | `[shoulder.right, button.east]` | RB + B，进入 `pd_brake` |
| `btn_4` | `[shoulder.right, button.north]` | RB + Y，进入 `initial_pos` |
| `btn_5` | `[shoulder.left, button.west]` | LB + X，进入 `dance` |
| `btn_6` | `[shoulder.left, button.south]` | LB + A，进入 `recover` |
| `btn_7` | `[shoulder.left, button.east]` | LB + B，进入 `normal_run` |
| `btn_8` | `[shoulder.left, button.north]` | LB + Y，进入 `amp_run` |
| `btn_9` | `[button.west]` | X，触发 `toggle_dance_pause` |
| `btn_10=1` | `[trigger.left, button.west]` | LT + X，进入 `back_flip` |
| `btn_10=2` | `[trigger.left, button.north]` | LT + Y，进入 `forward_flip` |
| `btn_10=3` | `[trigger.right, button.east]` | RT + B，进入 `applause` |
| `btn_10=0` | `trigger.left/right release_outputs` | 松开 LT/RT 时复位 `btn_10` |

`keyboard`：

- `keyboard.timeout_us`：键盘读取超时时间，单位微秒。超时没有收到移动键时，会把移动轴清零，用来模拟按键释放。
- `keyboard.movement`：键盘移动控制映射。
- `keyboard.movement.forward`：前进键，写入 `vx` 轴负满量程，再经过轴方向和速度缩放。
- `keyboard.movement.backward`：后退键。
- `keyboard.movement.yaw_left`：左转键。
- `keyboard.movement.yaw_right`：右转键。
- `keyboard.movement.strafe_left`：左平移键。
- `keyboard.movement.strafe_right`：右平移键。
- `keyboard.movement.stop`：停止移动键。`space` 表示空格。
- `keyboard.bindings`：键盘按键到输出命令的映射，输出格式和 `bindings[].output` 相同。

当前键盘移动键：

| 字段 | 当前按键 | 含义 |
| --- | --- | --- |
| `keyboard.movement.forward` | `w` | 前进 |
| `keyboard.movement.backward` | `s` | 后退 |
| `keyboard.movement.yaw_left` | `a` | 左转 |
| `keyboard.movement.yaw_right` | `d` | 右转 |
| `keyboard.movement.strafe_left` | `q` | 左平移 |
| `keyboard.movement.strafe_right` | `e` | 右平移 |
| `keyboard.movement.stop` | `space` | 清零移动命令 |

当前键盘绑定含义：

| keyboard.bindings key | output | 业务含义 |
| --- | --- | --- |
| `"1"` | `btn_1` | 进入 `normal` |
| `"2"` | `btn_6` | 进入 `recover` |
| `"3"` | `btn_5` | 进入 `dance` |
| `"4"` | `btn_8` | 进入 `amp_run` |
| `"5"` | `btn_7` | 进入 `normal_run` |
| `"6"` | `btn_10=1` | 进入 `back_flip` |
| `"7"` | `btn_10=2` | 进入 `forward_flip` |
| `"8"` | `btn_10=3` | 进入 `applause` |
| `"0"` | `btn_10=0` | 复位 `btn_10` |

`system`：

- `system.start_commands`：收到 `system.start` 输出后依次执行的 shell 命令。当前用于创建日志目录并后台启动硬件 demo 和 BMS。
- `system.stop_commands`：收到 `system.stop` 输出后依次执行的 shell 命令。当前用于向相关进程发送 `SIGINT`。
- 命令通过 `std::system()` 执行，按 YAML 顺序逐条运行；这里写的是原生 shell 命令字符串。

当前 `start_commands`：

```bash
mkdir -p /var/log/bxi_log
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py > /var/log/bxi_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log 2>&1 &
ros2 launch bxi_example_bms bms.launch.py > /var/log/bxi_log/bms_$(date +%Y-%m-%d_%H-%M-%S)_bms.log 2>&1 &
```

当前 `stop_commands`：

```bash
killall -SIGINT hardware_elf3
killall -SIGINT bxi_example_py_elf3
killall -SIGINT bxi_example_py_elf3_demo
killall -SIGINT bxi_bms
killall -SIGINT bxi_example_bms
```

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

### 状态机配置字段完整说明

本节逐项说明 `src/bxi_example_py_elf3/config/elf3_state_machine.yaml` 当前使用到的所有字段。

`initial_state`：

- `initial_state`：状态机初始化后进入的第一个状态名。必须是 `states` 下已定义的状态。当前是 `zero_torque`。

`remote_events`：

- `remote_events`：把 `MotionCommands.btn_1..btn_10` 转换成业务事件名。状态转移表只看业务事件名，不直接看 `btn_N`。
- `remote_events.<event>: btn_N`：简写形式。只要对应 `btn_N` 的值发生变化，就触发 `<event>`。
- `remote_events.<event>.slot`：完整形式的槽位名，例如 `btn_10`。
- `remote_events.<event>.value`：完整形式的期望值。只有 `slot` 变化到该值时才触发事件。

当前事件映射含义：

```text
normal               <- btn_1
zero_torque          <- btn_2
pd_brake             <- btn_3
initial_pos          <- btn_4
dance                <- btn_5
recover              <- btn_6
normal_run           <- btn_7
amp_run              <- btn_8
toggle_dance_pause   <- btn_9
back_flip            <- btn_10 == 1
forward_flip         <- btn_10 == 2
applause             <- btn_10 == 3
```

`transition_profiles`：

- `transition_profiles`：过渡配置表。状态切换规则里的 `transition` 字段引用这里的 profile 名。
- `transition_profiles.<profile>.duration`：过渡时长，单位秒。`0.0` 表示立即切换。
- `transition_profiles.<profile>.exit_behavior`：过渡期间旧状态的行为。
- `transition_profiles.<profile>.enter_behavior`：过渡期间新状态的行为。
- `hold_last_motor`：保持上一帧电机目标，避免过渡期间没有输出。
- `none`：不做额外处理。

当前 profile：

| profile | duration | exit_behavior | enter_behavior | 含义 |
| --- | ---: | --- | --- | --- |
| `instant` | `0.0` | `hold_last_motor` | `none` | 立即切换；离开旧状态时保留最后一帧电机目标，进入新状态不做额外过渡处理 |
| `soft_switch` | `0.02` | `hold_last_motor` | `hold_last_motor` | 用 0.02 秒软切换；退出侧和进入侧都保持上一帧电机目标，避免过渡期间输出断档 |

`speed_profiles`：

- `speed_profiles`：不同运动状态的速度缩放表。状态配置里的 `speed_profile` 引用这里的 profile 名。
- `speed_profiles.<profile>.vx_scale`：遥控器 `vel_des.x` 进入业务层后的倍率。
- `speed_profiles.<profile>.vx_min`：`vx` 缩放后的下限。
- `speed_profiles.<profile>.vx_max`：`vx` 缩放后的上限。
- `speed_profiles.<profile>.vy_scale`：遥控器 `vel_des.y` 的倍率。
- `speed_profiles.<profile>.yaw_scale`：遥控器 `yawdot_des` 的倍率。

当前 profile：

| profile | vx_scale | vx_min | vx_max | vy_scale | yaw_scale | 含义 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `normal` | `1.0` | `-1.0` | `1.0` | `0.5` | `1.5` | 普通站走速度 |
| `normal_run` | `2.0` | `-1.0` | `2.0` | `0.5` | `1.0` | 普通跑步速度 |
| `amp_run` | `4.0` | `-2.0` | `4.0` | `0.5` | `1.5` | 高速跑步速度 |

`states`：

- `states`：状态定义表。每个 key 是状态名，例如 `normal`、`dance`、`applause`。
- `states.<state>.behavior`：状态类名，必须在 `robot_states.py` 中存在，例如 `NormalState`。
- `states.<state>.id`：可选固定数字 ID。通常不要写；不写时按 `states` 顺序自动分配。
- `states.<state>.params`：可选构造参数，会作为关键字参数传给状态类构造函数。例如 `params: {start_frame: 100}`。
- `states.<state>.speed_profile`：可选速度 profile 名。当前 `normal`、`amp_run`、`normal_run`、`applause` 会读取速度命令；没有该字段的状态会忽略摇杆速度。
- `states.<state>.transitions`：该状态下允许的状态转移规则。
- `states.<state>.transitions.on_event`：按事件触发的规则表，key 是 `remote_events` 里定义的业务事件名。
- `states.<state>.transitions.on_event.<event>: <target>`：简写形式，收到事件后立即切换到目标状态，使用 `instant` 过渡。
- `states.<state>.transitions.on_event.<event>.to`：完整形式的目标状态名。
- `states.<state>.transitions.on_event.<event>.transition`：切换使用的过渡 profile 名，默认 `instant`。
- `states.<state>.transitions.on_event.<event>.delay`：收到事件后延迟多少秒再执行切换或 action，默认 `0.0`。
- `states.<state>.transitions.on_event.<event>.action`：只执行动作，不切换状态。当前 `toggle_dance_pause` 由 `DanceState.on_action()` 处理。
- `states.<state>.transitions.after`：可选自动转移规则列表，进入当前状态后计时触发。
- `states.<state>.transitions.after[].seconds`：进入当前状态多少秒后触发；也支持写成 `after`。
- `states.<state>.transitions.after[].to`：自动转移目标状态。
- `states.<state>.transitions.after[].transition`：自动转移使用的过渡 profile。
- `states.<state>.transitions.after[].action`：到时只执行 action，不切换状态。

当前状态说明：

```text
normal       -> NormalState，普通站走，可进入其他业务状态
zero_torque  -> ZeroTorqueState，零力矩
pd_brake     -> PdBrakeState，PD 刹车/保持
initial_pos  -> InitialPosState，回初始位置
dance        -> DanceState，舞蹈，可用 btn_9 暂停/继续
recover      -> RecoverState，倒地恢复
amp_run      -> AmpRunState，高速跑
normal_run   -> NormalRunState，普通跑
back_flip    -> BackFlipState，后空翻
forward_flip -> ForwardFlipState，前空翻
applause     -> ApplauseState，鼓掌
```

当前 `states` 明细：

| state | behavior | speed_profile | on_event 转移/action |
| --- | --- | --- | --- |
| `normal` | `NormalState` | `normal` | `zero_torque -> zero_torque`；`pd_brake -> pd_brake`；`initial_pos -> initial_pos`；`amp_run -> amp_run, soft_switch`；`normal_run -> normal_run, soft_switch`；`dance -> dance, soft_switch`；`recover -> recover, soft_switch`；`back_flip -> back_flip, soft_switch`；`forward_flip -> forward_flip, soft_switch`；`applause -> applause, soft_switch` |
| `zero_torque` | `ZeroTorqueState` | 未配置 | `normal -> normal, soft_switch`；`pd_brake -> pd_brake`；`initial_pos -> initial_pos`；`recover -> recover, soft_switch` |
| `pd_brake` | `PdBrakeState` | 未配置 | `normal -> normal, soft_switch`；`zero_torque -> zero_torque`；`initial_pos -> initial_pos`；`recover -> recover, soft_switch` |
| `initial_pos` | `InitialPosState` | 未配置 | `normal -> normal, soft_switch`；`pd_brake -> pd_brake`；`zero_torque -> zero_torque`；`recover -> recover, soft_switch` |
| `dance` | `DanceState` | 未配置 | `normal -> normal, soft_switch`；`toggle_dance_pause -> action: toggle_dance_pause` |
| `recover` | `RecoverState` | 未配置 | `normal -> normal, soft_switch`；`zero_torque -> zero_torque`；`pd_brake -> pd_brake`；`initial_pos -> initial_pos` |
| `amp_run` | `AmpRunState` | `amp_run` | `normal -> normal, soft_switch` |
| `normal_run` | `NormalRunState` | `normal_run` | `normal -> normal, soft_switch` |
| `back_flip` | `BackFlipState` | 未配置 | `normal -> normal, soft_switch`；`zero_torque -> zero_torque` |
| `forward_flip` | `ForwardFlipState` | 未配置 | `normal -> normal, soft_switch`；`zero_torque -> zero_torque` |
| `applause` | `ApplauseState` | `normal` | `normal -> normal, soft_switch`；`zero_torque -> zero_torque` |

表里没有写 `transition` 的简写转移会使用默认 `instant` profile。当前 YAML 没有使用 `id`、`params`、`delay`、`after`，这些字段是状态机支持的扩展字段，新增状态时可以按需要添加。

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

### 3.4 注册状态类

状态 ID 不需要手动写。`build_robot_states()` 会读取 `elf3_state_machine.yaml` 里的 `states:`，按配置顺序自动生成 ID，并根据 `behavior` 找到同名状态类。

新增状态时，只要把状态类放在 `robot_states.py` 里，然后在 YAML 里引用它：

```yaml
states:
  wave:
    behavior: WaveState
```

如果确实需要兼容某个固定数字 ID，可以在 YAML 里可选写 `id`：

```yaml
states:
  wave:
    id: 11
    behavior: WaveState
```

一般不要写 `id`，避免以后插入状态时又要维护数字。

`behavior` 必须等于 Python 类名。比如 `behavior: WaveState` 会实例化：

```python
class WaveState(RobotControlState):
    ...
```

### 3.5 配置状态转移

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

### 3.6 配置过渡时间和过渡行为

过渡 profile 在 YAML 顶部定义：

```yaml
transition_profiles:
  soft_switch:
    duration: 0.02
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

### 3.7 添加新 action

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

1. `robot_states.py` 里有同名状态类。
2. `elf3_state_machine.yaml` 的 `states.<name>.behavior` 写的是这个类名。
3. 允许进入该状态的 `transitions.on_event` 已经配置。

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
