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
- 输入驱动抽象：`src/remote_controller/include/remote_controller/input_driver.hpp`
- 内置输入驱动：`src/remote_controller/src/input_driver.cpp`
- 遥控器节点：`src/remote_controller/src/main.cpp`
- 状态机配置：`src/bxi_example_py_elf3/config/elf3_state_machine.yaml`
- 状态机引擎：`src/bxi_example_py_elf3/bxi_example_py_elf3/state_machine.py`
- 机器人状态基类：`src/bxi_example_py_elf3/bxi_example_py_elf3/robot_state_base.py`
- 机器人状态类：`src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py`

`remote_controller` 的核心解析、输入驱动和映射代码已经做成 `remote_controller_core` 库。以后写 CRSF、串口、UDP，可以新增 `InputDriver`，也可以单独写节点复用 `RemoteConfig` 和 `InputMapper`。

## 2. 遥控器配置

配置文件顺序建议保持：

```yaml
sources:
  ...

curves:
  ...

controls:
  ...

outputs:
  ...

system:
  ...
```

### 2.1 sources

`sources` 只声明语义 source 从哪里来，不写业务输出。

统一规则：

```text
左边：后续 YAML 要引用的语义 source
右边：这个 source 的来源描述，必须写 from
```

Linux joystick 示例：

```yaml
sources:
  gamepad:
    type: joystick
    device: /dev/input/js0
    signals:
      gamepad.left_y: {from: js.axis.3}
      gamepad.trigger_left: {from: js.axis.5}
      gamepad.rb: {from: js.button.7}
      gamepad.x: {from: js.button.3}
```

字段：

- `sources.<name>`：输入源分组名，例如 `gamepad`、`keyboard`、`crsf`。
- `sources.<name>.type`：输入源类型。内置支持 `joystick` 和 `keyboard`；其他类型按自定义 source 处理。
- `sources.<name>.device` / `sources.<name>.js`：joystick 设备路径。
- `sources.<name>.signals.<signal>`：声明一个语义 source。
- `sources.<name>.signals.<signal>.from`：这个语义 source 从哪里来，例如 `js.axis.3`、`js.button.7`、`keyboard.axis`、`keyboard.key`、`crsf.ch5`。
- `sources.<name>.signals.<signal>.hold_ms`：仅键盘 source 使用，覆盖该键或模拟轴的保持时间。
- `sources.<name>.signals.<signal>.timeout_ms`：source 超时，单位毫秒。CRSF、UDP、串口这类持续刷新输入按最后一次收到该 raw source 的时间判断；Linux joystick 这种事件型输入按 `/dev/input/js*` 驱动连接活跃时间判断，摇杆值保持不变不会触发超时。
- `sources.<name>.signals.<signal>.failsafe`：超时后写入该 raw source 的值，通常是 `0.0`。

这些语义 source 后续直接引用，例如 `gamepad.left_y`、`gamepad.x`。

键盘示例：

```yaml
sources:
  keyboard:
    type: keyboard
    poll_timeout_us: 20000
    hold_ms: 200
    stop: space
    signals:
      keyboard.vx: {from: keyboard.axis, negative: w, positive: s}
      keyboard.vy: {from: keyboard.axis, negative: q, positive: e}
      keyboard.yaw: {from: keyboard.axis, negative: a, positive: d}
      keyboard.normal: {from: keyboard.key, key: "1", hold_ms: 200}
      keyboard.back_flip: {from: keyboard.key, key: "6"}
```

键盘字段：

- `poll_timeout_us`：读取键盘的轮询超时，单位微秒。
- `hold_ms`：terminal 键盘没有真实释放事件，按一次键后对应 source 保持按下的毫秒数。
- `stop`：清空键盘模拟移动轴的按键。`space` 表示空格。
- `from: keyboard.axis`：声明一个由两个键模拟的轴。
- `negative`：让该轴变成负方向的键。
- `positive`：让该轴变成正方向的键。
- `from: keyboard.key`：声明一个动作键。
- `key`：动作键的键位。

例子 `keyboard.normal: {from: keyboard.key, key: "1"}` 的读法是：定义 `keyboard.normal`，它由 `1` 键产生。

CRSF 示例：

```yaml
sources:
  crsf:
    type: crsf
    signals:
      crsf.throttle: {from: crsf.ch1, timeout_ms: 200, failsafe: 0.0}
      crsf.mode: {from: crsf.ch5, timeout_ms: 200, failsafe: 0.0}
```

配置解析器只负责声明别名。真正读取 CRSF 协议的节点需要调用：

```cpp
mapper.set_signal("crsf.ch1", normalized_ch1);
mapper.set_signal("crsf.ch5", normalized_ch5);
```

### 2.2 curves

`curves` 是可复用的输入曲线。它可以挂在 `controls.<name>.curve`，也可以挂在 `controls.<name>.sources[].curve`。

示例：

```yaml
curves:
  stick:
    type: expo
    deadzone: 0.03
    expo: 0.2
    limit: [-1.0, 1.0]
    calibration:
      input: [-1.0, 0.0, 1.0]
      output: [-1.0, 0.0, 1.0]

  crsf_switch:
    type: piecewise
    points:
      - [172, -1.0]
      - [992, 0.0]
      - [1811, 1.0]
    limit: [-1.0, 1.0]
```

字段：

- `curves.<name>`：曲线名，后续用 `curve: <name>` 引用。
- `type`：曲线类型。`expo` 使用指数曲线；`piecewise` 使用分段线性插值。
- `deadzone`：曲线死区，绝对值小于等于该值时输出 `0`。
- `expo`：指数曲线强度，范围 `0.0..1.0`。越大，摇杆中心越柔和。
- `points`：`type: piecewise` 使用，按输入从小到大写 `[input, output]` 点。
- `calibration.input`：输入校准三点 `[min, center, max]`。
- `calibration.output`：输出校准三点 `[min, center, max]`。
- `calibration.clamp`：是否把输入夹在校准范围内，默认 `true`。
- `limit`：输出限幅，写成 `[min, max]`。
- `min` / `max`：也可以不用 `limit`，分别写下限和上限。

### 2.3 controls

`controls` 把 source 解释成统一控件。业务绑定只看 control，不关心它来自 Xbox、键盘还是 CRSF。

字段：

- `controls.<name>`：控件名，例如 `move.vx`、`button.west`、`mode.switch`。
- `type`：`analog`、`bool`、`enum`。
- `source`：单个 source 或 source 别名。
- `sources`：多个 source。
- `expr`：派生 bool control，由其他 control 的条件表达式计算出来。
- `mix`：多个 source 的混合策略。支持 `max_abs`、`sum`、`first_active`，默认 `max_abs`。
- `direction`：单个 source 的方向，默认 `1`，写 `-1` 可反向。
- `scale`：单个 source 的倍率，默认 `1.0`。
- `offset`：单个 source 的偏置，默认 `0.0`。
- `sources[].direction/scale/offset/deadzone/expo/curve`：单个 source 的局部变换。
- `invert`：control 层整体取反。
- `expo`：control 层曲线，`0.0..1.0`，越大中心越柔和。
- `curve`：control 层使用的命名曲线。
- `default`：control 没有匹配值时的默认字符串值，常用于 `enum`。

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

`expr` 条件写法和 `outputs.level[].when` 一样，支持 `all`、`any`、`pressed`、`released`、`equals`、`range`。派生 control 可以引用任意已定义 control，加载时会检查未知引用和循环依赖。

示例：

```yaml
controls:
  move.vx:
    type: analog
    mix: max_abs
    sources:
      - source: gamepad.left_y
        direction: -1
        curve: stick
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

  command.back_flip:
    type: bool
    expr:
      any:
        - [trigger.left, button.west]
        - [keyboard.back_flip]
```

### 2.4 outputs

`outputs` 是最终输出层，分三类：

- `outputs.conflict_policy`：多个 level binding 同时写同一个 `btn_N` 且值不同时的处理策略。
- `outputs.publish_on_change`：是否只在 `MotionCommands` payload 变化时发布。默认 `true`；设为 `false` 时按节点 timer 固定频率发布。
- `outputs.analog`：连续量，写到 `MotionCommands.vel_des` 或 `yawdot_des`。
- `outputs.level`：电平量，条件满足就保持，条件不满足就回 `0`。`btn_*` 推荐用这个。
- `outputs.edge`：边沿量，条件从不满足变成满足时触发一次。`system.*` 推荐用这个，`btn_*` 放这里会变成一次脉冲。

`conflict_policy` 支持：

- `first_wins`：先匹配到的 binding 生效，后面的冲突写入被忽略。
- `last_wins`：后匹配到的 binding 覆盖前面的值。
- `error`：运行时如果真实发生冲突，抛出错误；适合调试严格配置。

analog 示例：

```yaml
outputs:
  conflict_policy: first_wins
  publish_on_change: true

  analog:
    vx: move.vx
    vy: move.vy
    yaw: move.yaw
    height_des:
      control: body.height
      limit: [0.3, 1.2]
```

analog 字段：

- `outputs.analog.vx`：写到 `MotionCommands.vel_des.x`。
- `outputs.analog.vy`：写到 `MotionCommands.vel_des.y`。
- `outputs.analog.vz`：写到 `MotionCommands.vel_des.z`。
- `outputs.analog.yaw`：写到 `MotionCommands.yawdot_des`。
- `outputs.analog.height`：写到 `MotionCommands.height_des`。
- `outputs.analog.<field_path>`：也可以直接写 `MotionCommands` 字段路径，例如 `vel_des.z`、`height_des`。
- 值可以直接写 control，也可以写 `{control: move.vx, offset: 0.0}`。
- `controls`：多个 control。
- `mix`：多个 control 的混合策略，支持 `max_abs`、`sum`、`first_active`。
- `scale` / `offset`：输出层整体缩放和偏置。
- `limit` 或 `min` / `max`：输出限幅。

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
          - [keyboard.normal]
```

binding 字段：

- `output`：支持 `btn_N`、`btn_N=value`、`system.<action>`。
- `when`：触发条件。
- 放在 `outputs.level` 下就是 level 输出；放在 `outputs.edge` 下就是 edge 输出，不再单独写 `mode`。

输出目标和模式：

- `btn_*` 可以放在 `level` 或 `edge`。
- `system.*` 只能放在 `edge`，因为 shell 命令没有“条件不满足时撤销”的 level 语义。
- `edge` 下的 `btn_*` 会在条件上升沿输出一次，发布一帧后自动回 `0`。

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

### 2.5 btn_N 变化逻辑

`btn_N` 在 `outputs.level` 下是电平输出，不自动发布一帧后清零，也不需要 `release_outputs`。

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

如果把 `btn_N` 放在 `outputs.edge` 下，它就是脉冲输出：

```yaml
outputs:
  edge:
    - output: btn_1=1
      when:
        any:
          - [shoulder.right, button.west]
          - [keyboard.normal]
```

语义：

```text
条件 false -> true 的第一帧  -> btn_1 = 1
下一次发布                  -> btn_1 = 0
条件保持 true               -> 不重复触发
条件先回 false 再到 true     -> 再次触发一帧脉冲
```

`remote_controller` 发布 `/motion_commands` 时会填写 `header.stamp` 和 `header.frame_id=remote_controller`。`outputs.publish_on_change: true` 时，节点只在 `MotionCommands` 除 header 外的 payload 发生变化时发布；只有 header 时间戳变化不会触发发布。这样遥控器输入稳定时不会持续占用 `/motion_commands`，其他节点可以在这段时间发布自己的命令。`outputs.publish_on_change: false` 时关闭这个节流逻辑，节点恢复固定频率发布。

### 2.6 system

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

### 2.7 配置自检

遥控器节点加载 YAML 时会先做自检。

会直接报错并停止启动的情况：

- `sources`、`controls`、`outputs` 结构类型不对。
- source、curve、control 重名。
- control 引用了不存在的 source。
- 派生 control 的 `expr` 引用了不存在的 control 或形成循环依赖。
- binding 引用了不存在的 control。
- `outputs.analog` 写了不支持的 `MotionCommands` 字段。
- `system.*` 输出引用了不存在的 system action。
- `system_mutexes` 或 `system_reset_motion_after` 引用了不存在的 action。
- `alpha`、`expo`、`deadzone`、`timeout_ms`、curve calibration/points 等数值越界。
- `outputs.level` 里写了非 `btn_N` 输出，或 `outputs.edge` 里写了非 `btn_N` / `system.<action>` 输出。

会打印 warning/info 但继续启动的情况：

- 顶层出现未知字段。
- source、control 名字里有不推荐字符。
- enum 档位有重叠或空洞。
- 定义了没有使用的 source、curve、control。
- 同一个 `btn_N` 有多个可能值；最终按 `outputs.conflict_policy` 处理。

## 3. 添加非 joystick 输入源

有两种方式：

- 如果新输入源仍由 `remote_controller` 节点统一发布 `/motion_commands`，新增一个 `InputDriver`。
- 如果新输入源是独立 ROS2 节点，复用 `RemoteConfig` 和 `InputMapper`，自己发布 `/motion_commands`。

### 3.1 新增 InputDriver

驱动接口在：

```text
src/remote_controller/include/remote_controller/input_driver.hpp
```

新增驱动时做三件事：

1. 在 `src/remote_controller/src/input_driver.cpp` 里新增一个 `InputDriver` 子类。
2. 在驱动里读取协议，把通道值写成 raw source，例如 `mapper.set_signal("crsf.ch1", value)`。
3. 在 `create_input_driver()` 里注册新的 `driver_type`。

驱动内部需要遵守：

- 持有 `mapper_lock` 时只调用 `mapper.set_signal()` / `mapper.handle_*()`，拿到 edge outputs 后释放锁再执行回调。
- 周期逻辑不要自己写业务判断，业务判断留给 YAML 的 `controls` 和 `outputs`。
- `mapper.tick()` 由主节点定时器周期调用，驱动只负责输入事件。

启动时可用：

```bash
ros2 run remote_controller remote_controller \
  --config src/remote_controller/config/xbox_default.yaml \
  --driver joystick
```

键盘模式仍兼容：

```bash
ros2 run remote_controller remote_controller \
  --config src/remote_controller/config/xbox_default.yaml \
  --driver keyboard
```

`--keyboard` 也还可以用，等价于 `--driver keyboard`。

### 3.2 独立节点复用 InputMapper

独立节点原则：

1. 新节点负责读取协议。
2. 把协议通道归一化为 `-1.0..1.0`、`0.0/1.0`，或者直接交给 curve calibration 处理。
3. 复用 `remote_controller_core::InputMapper`。
4. 发布同一个 `/motion_commands`。

CRSF 三档开关示例：

```yaml
sources:
  crsf:
    type: crsf
    signals:
      crsf.throttle: {from: crsf.ch1, timeout_ms: 200, failsafe: 0.0}
      crsf.mode: {from: crsf.ch5, timeout_ms: 200, failsafe: 0.0}

curves:
  crsf_channel:
    type: expo
    calibration:
      input: [172, 992, 1811]
      output: [-1.0, 0.0, 1.0]
    deadzone: 0.02
    limit: [-1.0, 1.0]

controls:
  throttle:
    type: analog
    source: crsf.throttle
    curve: crsf_channel
    deadzone: 0.02
    min: -1.0
    max: 1.0
    alpha: 0.05

  mode.switch:
    type: enum
    source: crsf.mode
    curve: crsf_channel
    hysteresis: 0.05
    positions:
      low: [-1.0, -0.35]
      mid: [-0.35, 0.35]
      high: [0.35, 1.0]

outputs:
  analog:
    vx: throttle

  level:
    - output: btn_10=1
      when: [mode.switch=high]
```

节点伪代码：

```cpp
auto edge_outputs = mapper.tick();
dispatch_outputs(edge_outputs);

auto outputs = mapper.set_signal("crsf.ch1", normalize(channel_1));
dispatch_outputs(outputs);

outputs = mapper.set_signal("crsf.ch5", normalize(channel_5));
dispatch_outputs(outputs);

communication::msg::MotionCommands msg;
mapper.fill_message(msg);
if (payload_changed(msg)) {
    msg.header.stamp = node->now();
    msg.header.frame_id = "my_controller";
    publisher->publish(msg);
}
```

`tick()` 要周期调用，用来处理键盘保持时间、source 超时/failsafe、edge 条件更新。

如果多个节点都发布 `/motion_commands`，至少要做到“输出没变不重复发布”，否则高频发布的一方会持续覆盖其他节点。更稳的做法还是给上层做一个仲裁节点。

`MotionCommands` 的字段适配集中在 `remote_controller/motion_commands_adapter.*`。如果以后底层消息结构变化，优先改这个 adapter：支持哪些字段路径、字段路径怎么写入消息、`btn_N` 兼容槽位怎么落到新消息。发布节流使用 ROS2 生成消息自带的整包相等比较，并且保存的是 header 为空的 payload，所以新增消息字段后一般不需要再手写比较逻辑。

## 4. 当前能力边界

已经实现：

- source 别名声明。
- joystick source 自动归一化。
- keyboard 全局和单键 `hold_ms` 模拟释放。
- `analog`、`bool`、`enum` control。
- source 超时/failsafe。
- 命名曲线 `curves`，支持 deadzone、expo、limit、calibration、piecewise points。
- source 局部 `direction/scale/offset/deadzone/expo/curve`。
- control 层 `mix/invert/expo/default/expr`。
- 多 source 混合，支持 `max_abs`、`sum`、`first_active`。
- `outputs.analog`、`outputs.level`、`outputs.edge`。
- `outputs.conflict_policy`。
- `when` 支持 `all`、`any`、`pressed`、`released`、`equals`、`range`。
- `InputDriver` 抽象层，内置 joystick 和 keyboard driver。
- 加载配置时自检并通过 ROS log 报告诊断信息。

没有内置实现：

- CRSF 协议读取节点本身。
- 串口/UDP 协议读取节点本身。
- 查表曲线。
- control 级优先级锁定。

这些可以继续在 `remote_controller_core` 上扩展，不需要改业务状态机。

## 5. 添加新的机器人状态

一个机器人状态对应一个 Python 类。状态类放在：

```text
src/bxi_example_py_elf3/bxi_example_py_elf3/robot_states.py
```

机器人状态基类放在：

```text
src/bxi_example_py_elf3/bxi_example_py_elf3/robot_state_base.py
```

通常新增状态只改 `robot_states.py`。`robot_state_base.py` 只放所有状态共享的机器人控制逻辑，例如 `RobotControlState`、`get_first_frame(ctx)` 的默认行为、进入过渡时的 `first_frame_ramp_kp` 执行逻辑。

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

`graph`：

- `graph.validate`：是否启动时做状态图自检，默认 `true`。
- `graph.export.dot`：导出 Graphviz dot 文件路径。
- `graph.export.mermaid`：导出 Mermaid 状态图文件路径。

自检会检查：

- transition 目标状态是否存在。
- transition profile 是否存在。
- `on_event` 是否在 `remote_events` 中声明。
- 从 `initial_state` 出发是否有不可达状态。
- 是否有状态没有配置任何 outgoing transition。
- 是否存在只由 `after` 自动转移构成的循环。

示例：

```yaml
graph:
  validate: true
  export:
    dot: /tmp/elf3_state_machine.dot
    mermaid: /tmp/elf3_state_machine.mmd
```

`transition_profiles`：

- `transition_profiles.<profile>.duration`：过渡时长，单位秒。`0.0` 表示立即切换。只写 `duration` 时，`exit_duration` 和 `enter_duration` 都等于它。
- `transition_profiles.<profile>.exit_duration`：旧状态退出过渡时长。旧状态的 `on_exit_transition(..., progress, ...)` 按这个时长计算 `progress`。
- `transition_profiles.<profile>.enter_duration`：新状态进入过渡时长。新状态的 `on_enter_transition(..., progress, ...)` 按这个时长计算 `progress`。
- `transition_profiles.<profile>.exit_behavior`：过渡期间旧状态默认行为。
- `transition_profiles.<profile>.enter_behavior`：过渡期间新状态默认行为。
- `transition_profiles.<profile>.data`：过渡私有数据，必须是 map。状态机不理解里面的字段，只负责透传给 `on_exit_transition()` / `on_enter_transition()`。
- `hold_last_motor`：保持上一帧电机目标。
- `none`：不做额外处理。
- `first_frame_ramp_kp`：进入新状态时，目标角度固定为新状态 `get_first_frame(ctx)` 返回的第一帧角度，`kp` 按 `enter_progress` 从 `data.kp_start` 缓慢变化到第一帧 `kp`，`kd` 按 `data.kd_start` 变化到第一帧 `kd`。如果目标状态没有实现 `get_first_frame(ctx)`，会退回保持上一帧电机目标。
- `first_frame_ramp_kp` 的 `data.kp_start`：可选 `current`、`target`、`zero`，默认 `current`。
- `first_frame_ramp_kp` 的 `data.kd_start`：可选 `current`、`target`、`zero`，默认 `target`。

状态机实际过渡总时长是 `duration`、`exit_duration`、`enter_duration` 的最大值。这样可以让旧状态先完成退出，新状态继续进入，或者反过来。

从已有 profile 继承时，如果只覆盖 `exit_duration` 或 `enter_duration`，并且不写 `duration`，总过渡时长会用覆盖后的两侧时长重新计算。需要保留一个更长的总过渡时间时，显式写 `duration`。

`speed_profiles`：

- `speed_profiles.<profile>.vx_scale`：`vel_des.x` 倍率。
- `speed_profiles.<profile>.vx_min`：缩放后的 `vx` 下限。
- `speed_profiles.<profile>.vx_max`：缩放后的 `vx` 上限。
- `speed_profiles.<profile>.vy_scale`：`vel_des.y` 倍率。
- `speed_profiles.<profile>.yaw_scale`：`yawdot_des` 倍率。

速度 profile 只会通过 `states.<state>.speed_profile` 显式引用，不会按状态名自动匹配。状态没有写 `speed_profile` 时，遥控器速度输入会被清零；写了不存在的 profile 名时，也会清零速度，并且节点会输出一次 warning。

`states`：

- `states.<state>.behavior`：状态类名，必须在 `robot_states.py` 中存在。
- `states.<state>.id`：可选固定数字 ID。通常不要写，不写时按 YAML 顺序自动分配。
- `states.<state>.params`：可选构造参数，会作为关键字参数传给状态类。
- `states.<state>.speed_profile`：可选速度 profile 名，必须指向 `speed_profiles` 下的某个 key。没有配置时，该状态不接收遥控器速度控制，内部 `vx / vy / dyaw` 会被置为 `0`。
- `states.<state>.transitions.on_event`：事件触发转移表。
- `states.<state>.transitions.on_event.<event>: <target>`：简写，立即切换。
- `states.<state>.transitions.on_event.<event>.to`：目标状态。
- `states.<state>.transitions.on_event.<event>.transition`：过渡 profile，默认 `instant`。
- `states.<state>.transitions.on_event.<event>.duration`：内联过渡总时长，不需要先在 `transition_profiles` 里定义 profile。
- `states.<state>.transitions.on_event.<event>.exit_duration`：内联旧状态退出过渡时长。
- `states.<state>.transitions.on_event.<event>.enter_duration`：内联新状态进入过渡时长。
- `states.<state>.transitions.on_event.<event>.exit_behavior`：内联旧状态退出行为。
- `states.<state>.transitions.on_event.<event>.enter_behavior`：内联新状态进入行为。
- `states.<state>.transitions.on_event.<event>.data`：内联过渡私有数据，会合并到最终 `TransitionProfile.data`。
- `states.<state>.transitions.on_event.<event>.delay`：延迟多少秒后执行。
- `states.<state>.transitions.on_event.<event>.action`：只执行 action，不切换状态。
- `states.<state>.transitions.after`：进入该状态后自动触发的规则列表。
- `states.<state>.transitions.after[].seconds`：进入该状态多少秒后触发；也支持写成 `after`。
- `states.<state>.transitions.after[].to`：自动转移目标状态。
- `states.<state>.transitions.after[].transition`：自动转移使用的过渡 profile。
- `states.<state>.transitions.after[].duration / exit_duration / enter_duration / exit_behavior / enter_behavior / data`：和 `on_event` 下的内联过渡字段相同。
- `states.<state>.transitions.after[].action`：到时只执行 action，不切换状态。

常用过渡可以继续定义成 profile：

```yaml
transition_profiles:
  soft_switch:
    duration: 0.02
    exit_behavior: hold_last_motor
    enter_behavior: hold_last_motor

  first_frame_switch:
    exit_duration: 0.02
    enter_duration: 0.20
    exit_behavior: hold_last_motor
    enter_behavior: first_frame_ramp_kp
    data:
      kp_start: current
      kd_start: target

states:
  normal:
    transitions:
      on_event:
        dance:
          to: dance
          transition: soft_switch
```

个别状态可以直接内联写，不需要专门新增一个 profile：

```yaml
states:
  normal:
    transitions:
      on_event:
        hello:
          to: hello
          exit_duration: 0.02
          enter_duration: 0.12
          exit_behavior: hold_last_motor
          enter_behavior: first_frame_ramp_kp
          data:
            kp_start: current
            kd_start: target
```

也可以把 `transition` 写成 map，必要时从已有 profile 继承：

```yaml
states:
  normal:
    transitions:
      on_event:
        applause:
          to: applause
          transition:
            base: soft_switch
            enter_duration: 0.10
```

Python 代码主动切换状态时也可以传同样的内联 transition：

```python
ctx.request_state(
    "hello",
    trigger="code",
    transition={
        "exit_duration": 0.02,
        "enter_duration": 0.12,
        "exit_behavior": "hold_last_motor",
        "enter_behavior": "first_frame_ramp_kp",
        "data": {
            "kp_start": "current",
            "kd_start": "target",
        },
    },
)
```

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

    def get_first_frame(self, ctx):
        return None

    def on_action(self, ctx, action_name):
        return False
```

调用顺序：

```text
触发切换事件
  -> 旧状态.on_exit()
  -> 新状态.on_prepare_enter()
  -> 过渡期间每个控制周期:
       旧状态.on_exit_transition(exit_progress)
       新状态.on_enter_transition(enter_progress)
  -> 新状态.on_enter()
  -> 新状态.on_update()
```

`exit_progress` 按 `exit_duration` 计算，`enter_progress` 按 `enter_duration` 计算。如果某一侧 duration 为 `0.0`，对应 progress 直接是 `1.0`。如果总 `duration` 为 `0.0`，状态机会直接进入新状态，不跑逐帧过渡钩子。

`get_first_frame(ctx)` 是给 `first_frame_ramp_kp` 用的可选接口。需要这个进入过渡效果的状态，返回 `(qpos, kp, kd)`；不需要时返回 `None` 或不重写。状态内部可以自由决定“第一帧”是什么，例如动作数据的起始帧、预热模型后的第一帧输出，或者某个固定姿态。

自定义过渡行为不要把私有字段加到 `state_machine.py`。把行为需要的参数放进 YAML 的 `data`，然后在状态类里通过 `transition.data` 读取，例如 `transition.data.get("kp_start", "current")`。这样状态机库只负责调度和透传，不关心具体机器人控制策略。

如果状态重写了 `on_prepare_enter()` 或 `on_exit()`，并且还想使用 `RobotControlState` 默认的第一帧缓存清理逻辑，先调用对应的 `super()`。

### 5.3 状态机信息话题

`bxi_example_demo.py` 会把状态机运行信息发布成 JSON 字符串：

```bash
ros2 topic echo /simulation/state_machine_info
```

默认话题名是 `<topic_prefix>state_machine_info`，例如 launch 里 `/topic_prefix: simulation/` 时就是 `/simulation/state_machine_info`。可以用参数 `/state_machine_info_topic` 覆盖话题名，用 `/state_machine_info_hz` 调整发布频率；频率小于等于 `0` 时关闭发布。

消息类型是 `std_msgs/msg/String`，`data` 字段是 JSON。主要字段：

- `mode`：`state`、`pending` 或 `transition`。
- `current`：当前状态的 `name`、`id` 和已运行时间 `elapsed`。
- `pending`：延迟切换中的目标、触发源、延迟时间和进度；没有延迟切换时为 `null`。
- `transition`：正在执行的过渡，包括 `from`、`to`、`profile`、`trigger`、`elapsed`、`duration`、`progress`、`exit_duration`、`exit_progress`、`enter_duration`、`enter_progress`、`data`。
- `events`：本控制周期从遥控器解析到的状态机事件。
- `cmd_vel`：当前业务层使用的速度命令。
- `graph`：状态列表、过渡 profile、遥控事件名和状态转移边，方便外部工具画状态图或做调试 UI。

### 5.4 新增状态示例

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

    def get_first_frame(self, ctx):
        return (
            ctx.dance.target_dof_pos,
            ctx.dance.stiffness_array,
            ctx.dance.damping_array,
        )
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

### 5.5 action

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

直接指定 driver：

```bash
ros2 run remote_controller remote_controller \
  --config src/remote_controller/config/xbox_default.yaml \
  --driver keyboard
```

观察输出：

```bash
ros2 topic echo /motion_commands
```
