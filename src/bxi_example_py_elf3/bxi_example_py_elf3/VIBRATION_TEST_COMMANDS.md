# Elf3 全身振动测试启动命令

工作区：

```text
/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example
```

## 1. 源码修改后重新构建

```bash
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"
bash build.sh
```

## 2. 默认启动：全部 29 个关节振动 60 秒

```bash
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"

source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash

ros2 launch bxi_example_py_elf3 example_launch_vibration.launch.py
```

该 launch 默认同时启动 `remote_controller`。如果手柄节点已经由系统服务或另一个
终端启动，必须避免重复发布：

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration.launch.py \
  start_remote_controller:=false
```

默认参数：

```text
关节：全部 29 个关节
振幅：±0.23 rad（约 ±13.18°）
频率：10 Hz 线性扫频到 20 Hz
时间：60 秒
电机指令频率：200 Hz
异步 CSV 采样频率：100 Hz
正常停止回中时间：0.5 秒
振动前逐关节预检：开启
预检动作：29 个关节依次执行中心→正向→中心→反向→中心
预检振幅：±0.03 rad
预检时间：每关节 2 秒，全部约 58 秒
自动开始：关闭；预检和振动分别需要人工确认
虚拟悬挂：保留
CSV：/tmp/elf3_vibration_test.csv
```

等待 launch 终端出现
`robot reset 2 acknowledged; initialization complete`。另开一个已 source 的终端，
启动仿真逐关节预检：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example/install/setup.bash"

ros2 service call /joint_rotation_test_enable \
  std_srvs/srv/SetBool "{data: true}"
```

等待明确出现 `JOINT ROTATION PRECHECK PASSED` 后，再启动仿真振动：

```bash
ros2 service call /vibration_test_enable \
  std_srvs/srv/SetBool "{data: true}"
```

也可以全程使用手柄 X 键：节点启动后先同步当前按键状态；初始化完成后第一次按
X 启动 29 关节预检，预检通过并等待至少 1 秒后第二次按 X 启动振动，振动中再按
X 则停止激励并平滑回中。手柄入口与服务入口使用同一安全状态机，不会绕过预检。

## 3. 全身振动两分钟

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration.launch.py \
  joint_name:=all \
  amplitude_rad:=0.03 \
  start_frequency_hz:=1.0 \
  end_frequency_hz:=5.0 \
  duration_sec:=120.0
```

launch 启动后仍不会自动运动。等待初始化完成，再按第 2 节依次调用预检服务；
预检通过后再次调用振动启动服务。

## 4. 全身持续定频振动

`duration_sec:=0` 表示持续运行，直到按 `Ctrl+C`。

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration.launch.py \
  joint_name:=all \
  amplitude_rad:=0.02 \
  start_frequency_hz:=2.0 \
  end_frequency_hz:=2.0 \
  duration_sec:=0
```

launch 启动后仍需按第 2 节完成人工启动预检，并在预检通过后第二次人工确认
启动振动。

## 5. 常用参数

```text
joint_name             保持为 all，表示全部 29 个关节
amplitude_rad           位置振幅，单位 rad
start_frequency_hz      起始频率，单位 Hz
end_frequency_hz        结束频率，单位 Hz
duration_sec            持续时间；0 表示连续运行
control_rate_hz         指令发布频率，默认 200 Hz
stop_ramp_sec           正常停止后的平滑回中时间，默认 0.5 秒
motion_button_mode      C++ 手柄用 toggle；按下/松开型键盘源用 momentary
joint_test_required     是否强制先完成 29 关节转动预检
joint_test_amplitude_rad 预检正、反方向振幅，默认 0.03 rad
joint_test_move_sec     每段平滑转动时间，默认 0.4 秒
joint_test_hold_sec     每个目标点的反馈验证时间，默认 0.1 秒
joint_test_min_motion_rad 正、反方向最小实测位移，默认 0.015 rad
log_rate_hz             异步 CSV 采样频率，默认 100 Hz
release_suspension      是否释放 MuJoCo 虚拟悬挂
auto_start              预检开启时必须为 false，预检和振动均需人工确认
log_csv_path            CSV 输出路径
```

查看 launch 支持的全部参数：

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration.launch.py --show-args
```

## 6. 查看测试数据

```bash
ls -lh /tmp/elf3_vibration_test.csv
head -n 3 /tmp/elf3_vibration_test.csv
tail -n 3 /tmp/elf3_vibration_test.csv
```

CSV 中会分别记录每个活动关节的指令位置和实测位置。

自定义输出文件：

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration.launch.py \
  log_csv_path:=/tmp/my_vibration_test.csv
```

## 7. 停止测试

在运行 launch 的终端按：

```text
Ctrl+C
```

停止后机器人保持测试开始时的中心关节位置。

## 安全提示

- 全身振动建议保持 `release_suspension:=false`。
- 增大振幅或频率时应逐步调整，不要一次增加过多。
- 当前 launch 只启动 MuJoCo，不会启动真实硬件。
- 如果修改了 Python 或 launch 文件，必须重新执行 `bash build.sh` 并重新启动。

## 8. 实机部署与启动

实机 launch 默认让全部 29 个关节同步振动。当前参数是高动态测试参数，
不是保守参数：

```text
振幅：±0.23 rad（约 ±13.18°）
频率：10 Hz 扫频到 20 Hz
时间：60 秒
电机指令频率：200 Hz
异步 CSV 采样频率：100 Hz
初始化：10 秒
自动开始：关闭
29 关节转动预检：强制开启
预检振幅：±0.03 rad
预检总时间：约 58 秒
```

先登录实机并切换到 root：

```bash
sudo -s
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"

source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash

ros2 launch bxi_example_py_elf3 example_launch_vibration_hw.launch.py
```

实机已有独立 `remote_controller` 时使用：

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration_hw.launch.py \
  start_remote_controller:=false
```

等待终端出现
`robot reset 2 acknowledged; initialization complete`，并确认机器人周围无人、
急停可用。仅看到 `robot reset 2 sent` 还不表示驱动已经确认成功。

另开一个 root 终端，确认硬件命令话题只有一个发布者：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example/install/setup.bash"

ros2 topic info /hardware/actuators_cmds
```

应显示：

```text
Publisher count: 1
Subscription count: 1
```

先人工启动 29 关节逐个转动预检：

```bash
ros2 service call /joint_rotation_test_enable \
  std_srvs/srv/SetBool "{data: true}"
```

测试过程中一次只会转动一个关节，其余 28 个关节保持中心位置。每个关节执行：

```text
中心 → +0.03 rad → 中心 → -0.03 rad → 中心
```

每个目标点都会收集多条独立关节反馈，检查正向位移、反向位移、回中误差以及
其他关节是否发生异常串动。查询当前进度：

```bash
ros2 service call /joint_rotation_test_status std_srvs/srv/Trigger "{}"
```

如需正常取消预检并平滑回中：

```bash
ros2 service call /joint_rotation_test_enable \
  std_srvs/srv/SetBool "{data: false}"
```

等待终端明确出现：

```text
JOINT ROTATION PRECHECK PASSED: 29/29 joints responded
```

预检通过后振动仍然不会自动开始。等待至少 1 秒，再次确认机器人周围无人、
急停和支撑状态后，第二次人工确认启动全身振动：

```bash
ros2 service call /vibration_test_enable std_srvs/srv/SetBool "{data: true}"
```

预检通过许可在当前节点中保留 300 秒，并在开始一次振动时消耗；下一次振动前
必须重新完成预检。手柄操作时，第一次按 X 可启动预检，预检通过后再次按 X
才会启动振动；预检运行中按 X 会取消并平滑回中。

默认 `motion_button_mode:=toggle` 与本工程的 C++ `remote_controller` 一致。
如果 `btn_9` 来源是按下发 1、松开发 0 的键盘/通用手柄映射，启动 launch 时改用：

```bash
motion_button_mode:=momentary
```

正常停止振动并平滑返回中心姿态：

```bash
ros2 service call /vibration_test_enable std_srvs/srv/SetBool "{data: false}"
```

服务返回后振动激励立即停止，节点默认用 0.5 秒连续回到中心姿态。等待终端出现
`smooth return complete; holding center position` 后，再在 launch 终端按 `Ctrl+C`。

实机 CSV：

```text
/tmp/elf3_vibration_hardware.csv
```

实机版会拒绝以下情况：

- 不是 root 用户启动；
- 未收到完整、实时的 29 关节反馈；
- 29 关节逐个转动预检尚未通过或通过许可已过期；
- 预检关节未同时通过正向、反向和回中反馈检查；
- 预检时其他关节发生超过限制的异常串动；
- `/hardware/actuators_cmds` 存在多个发布者；
- 振幅、频率、峰值速度或峰值加速度超过软件安全上限；
- 控制定时器或关节反馈发生超时；
- 关节反馈或待发布指令含有 `NaN`/`Inf`；
- 指令可能超过软件关节位置限制。

发生 `SAFETY FAULT` 时，实机版会退出振动控制节点，并由 launch 联动停止
`hardware_elf3`。故障锁存后必须排查原因并重新启动，不能直接再次调用启动服务。
逐关节预检中的正向、反向、回中、反馈样本数或串轴检查任一失败，也会按
`SAFETY FAULT` 处理；这是故意的 fail-closed 行为，不会自动继续测试或振动。

当前实机 launch 中四项 `hardware_max_*` 动态上限被设置为 `100000`，等同于
基本关闭软件的振幅、频率、速度和加速度保护；硬件驱动自身限制仍然有效。
`0.23 rad @ 20 Hz` 的理论峰值速度约为 `28.90 rad/s`，峰值加速度约为
`3632 rad/s²`。首次实机测试前应由机械、电气和控制负责人确认这些数值。

启动时覆盖预检参数、振动振幅、CSV 采样率或命令超时参数：

```bash
ros2 launch bxi_example_py_elf3 example_launch_vibration_hw.launch.py \
  joint_test_amplitude_rad:=0.03 \
  joint_test_move_sec:=0.4 \
  joint_test_hold_sec:=0.1 \
  joint_test_min_motion_rad:=0.015 \
  amplitude_rad:=0.23 \
  log_rate_hz:=100.0 \
  max_command_gap_sec:=0.05
```
