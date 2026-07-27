# Elf3 吊挂跑步/线束测试

工作区：

```text
/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example
```

## 构建

```bash
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"
bash build.sh
source setup_env.sh
```

## MuJoCo 仿真

现在可以用一个 launch 同时启动仿真、轨迹控制和手柄：

```bash
ros2 launch bxi_example_py_elf3 example_launch_test_wire.py
```

如果系统已经单独运行了 `remote_controller`，避免重复发布
`/motion_commands`：

```bash
ros2 launch bxi_example_py_elf3 example_launch_test_wire.py \
  start_remote_controller:=false
```

等待出现：

```text
robot reset 2 acknowledged; initialization complete
```

手柄接入后，节点会先同步当前 X 键状态，防止节点重启时因残留状态意外运动。
之后每按一次 X，在“播放轨迹”和“暂停并保持名义姿态”之间切换。

## 不使用手柄时通过服务控制

开始或继续播放：

```bash
ros2 service call /run_trajectory_enable \
  std_srvs/srv/SetBool "{data: true}"
```

暂停并保持名义姿态：

```bash
ros2 service call /run_trajectory_enable \
  std_srvs/srv/SetBool "{data: false}"
```

## 实机吊挂测试

确认吊装、急停和周边安全后，以 root 启动：

```bash
sudo -s
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash

ros2 launch bxi_example_py_elf3 example_launch_test_wire_hw.py
```

如果实机已有独立手柄进程，同样添加：

```text
start_remote_controller:=false
```

## 当前保持不变的调试参数

```text
控制频率：50 Hz
初始化时间：2 秒
虚拟悬挂：保留
轨迹：data/data.txt，共 308 帧，循环播放
Kp：全部 29 关节为 10000
Kd：保持原 Elf3 配置
```

节点启动时会输出轨迹最大相邻帧变化和循环首尾变化。当前轨迹在 50 Hz 下的
最大相邻帧等效速度约为 `19.4 rad/s`，循环首尾跳变等效速度约为
`7.25 rad/s`。这些数值仅作为告警显示，没有修改现有调试参数。

## 安全与冲突保护

- reset 1 和 reset 2 必须收到驱动成功响应后才能进入下一阶段；
- 轨迹必须是有限的 29 维数据，并位于软件关节位置限制内；
- 实机播放前和播放中要求完整、实时的 29 关节反馈；
- `/hardware/actuators_cmds` 或 `/simulation/actuators_cmds` 发现多个发布者时停止发布；
- 实机控制命令间隔超过限制时停止发布；
- 指令或反馈出现 `NaN`/`Inf` 时进入故障锁存，必须排查后重启；
- 手柄停止键会把跑步和振动测试进程纳入停止范围。

不要同时启动跑步节点、振动节点或其他全身控制器。虽然节点会检测发布者冲突，
但正确的操作方式仍然是每次只启动一个控制模式。
