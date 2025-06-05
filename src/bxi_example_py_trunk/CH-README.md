# BXI运动控制功能模块

## 项目概述

本项目为BXI机器人提供了一系列运动控制功能，特别专注于手臂动作控制，包括挥手、握手和跑步时的协调手臂运动。所有动作实现了平滑的过渡和自然的协调性，提升了机器人的交互能力和运动表现。

## 主要功能

### 1. 手臂挥舞控制
- 实现自然流畅的手臂挥舞动作
- 平滑的动作启动和停止过渡
- 可配置的动作幅度和速度

### 2. 右手握手功能
- 自然的握手动作实现
- 精确的关节控制
- 状态管理和过渡优化

### 3. 奔跑协调手臂动作
- 基于腿部相位的手臂协调运动
- 符合生物力学原理的自然摆动
- 提升运动平衡性和视觉表现

## 文件结构

```
bxi_example_py_trunk/
├── bxi_example_py_trunk/
│   ├── __init__.py
│   ├── bxi_example.py            # 主控制逻辑模块
│   ├── arm_motion_controller.py  # 手臂挥舞控制器
│   ├── right_arm_handshake_controller.py  # 右手握手控制器
│   └── running_arm_controller.py # 奔跑手臂协调控制器
├── launch/
│   └── example_launch.py         # 启动配置文件
└── policy/
    └── model.onnx               # 更新的模型文件
```

## 技术特点

1. **平滑过渡机制**
   - 使用缓动函数实现动作的平滑启动和停止
   - 避免突变导致的机械冲击和不自然动作

2. **信号平滑和滤波**
   - 对腿部相位信号进行平滑处理
   - 低通滤波实现更自然的摆动效果

3. **配置性**
   - 参数配置选项
   - 支持动态调整动作特性

## 使用方法

### 手臂挥舞

```python
# 初始化控制器
arm_motion_controller = ArmMotionController(logger, joint_nominal_pos_ref)

# 启动挥手动作
arm_motion_controller.start_waving(current_sim_time)

# 计算挥手动作关节角度
joint_pos = arm_motion_controller.calculate_wave_motion(base_pos, current_sim_time)

# 停止挥手动作
arm_motion_controller.stop_waving(current_sim_time)
```

### 握手功能

```python
# 初始化控制器
handshake_controller = RightArmHandshakeController(logger, joint_nominal_pos_ref)

# 启动握手动作
handshake_controller.start_handshake(current_sim_time)

# 计算握手动作关节角度
joint_pos = handshake_controller.calculate_handshake_motion(base_pos, current_sim_time)

# 停止握手动作
handshake_controller.stop_handshake(current_sim_time)
```

### 奔跑协调手臂动作

```python
# 初始化控制器
running_arm_controller = RunningArmController(logger, joint_nominal_pos_ref)

# 启动奔跑手臂协调
running_arm_controller.start_running_motion(current_sim_time)

# 计算奔跑时的手臂动作
joint_pos = running_arm_controller.calculate_running_arm_motion(
    base_pos, time_in_seconds, leg_phase_left_signal, leg_phase_right_signal
)

# 停止奔跑手臂协调
running_arm_controller.stop_running_motion(current_sim_time)
```

## 参数配置

### 奔跑手臂控制器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| arm_startup_duration | 动作启动过渡时间(秒) | 3.0 |
| arm_shutdown_duration | 动作关闭过渡时间(秒) | 3.0 |
| arm_amplitude_y | 肩部Y轴摆动幅度 | 0.15 |
| arm_amplitude_z | 肩部Z轴摆动幅度 | 0.02 |
| elbow_coeff | 肘部运动系数 | 0.05 |
| smoothing_factor | 腿部相位信号平滑因子 | 0.8 |

### 手臂挥舞控制器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| wave_startup_duration | 挥手启动过渡时间(秒) | 2.0 |
| wave_shutdown_duration | 挥手关闭过渡时间(秒) | 2.0 |
| wave_frequency | 挥手频率(Hz) | 0.5 |
| wave_amplitude | 挥手幅度 | 0.2 |

### 握手控制器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| handshake_startup_duration | 握手启动过渡时间(秒) | 1.5 |
| handshake_shutdown_duration | 握手关闭过渡时间(秒) | 1.5 |


