# Elf3 吊挂跑步与振动组合测试

组合模式只运行一个控制节点和一个 `actuators_cmds` 发布者，启动后通过手柄选择：

```text
X：启动/停止跑步轨迹
Y：直接启动/停止振动
A：启动/停止碰撞余量全行程关节测试（手臂 → 躯干 → 腿部）
Start（按钮 11）：急停，只终止当前硬件和机器人测试程序
```

Start 急停不会再启动 demo、BMS 或其他控制程序。软件急停不能替代独立的实体
急停回路。

## 仿真启动

```bash
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"
source setup_env.sh

ros2 launch bxi_example_py_elf3 \
  example_launch_suspended_tests.launch.py
```

## 实机启动

确认机器人已经可靠吊装、急停可用且周围无人后：

```bash
sudo -s
cd "/home/bxi/BXI/robot_vibration testing/bxi_rl_controller_ros2_example"
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash

ros2 launch bxi_example_py_elf3 \
  example_launch_suspended_tests_hw.launch.py
```

程序启动后只完成初始化并保持姿态，不会自动开始测试。等待初始化完成后，通过
手柄 X/Y/A 选择测试；模型碰撞检查无法看到吊具、线缆和机器人周边物体。

两个 launch 默认启动手柄节点。如果系统已经有独立的 `remote_controller`，添加：

```text
start_remote_controller:=false
```

## A 键参数配置

A 键全行程测试参数集中在：

```text
src/bxi_example_py_elf3/config/suspended_tests.yaml
```

调速度主要修改：

```yaml
limb_test_range_speed_deg_s: 10.0
```

单位是度/秒，表示 minimum-jerk 轨迹的峰值速度。仿真调试允许范围为 `>0` 到
`1000.0`，实机允许范围为 `>0` 到 `180.0`。高速度会显著增加动态冲击，建议逐步
增加。修改后重新启动 launch 即可；从项目根目录启动时会直接读取源码配置文件。

也可以临时指定另一份配置：

```bash
ros2 launch bxi_example_py_elf3 \
  example_launch_suspended_tests.launch.py \
  controller_config_file:=/绝对路径/你的配置.yaml
```

配置文件还包含最短移动时间、目标保持时间、碰撞余量、机械限位余量、跟踪误差、
A/跑步中心启动姿态允许误差、振动启动包络微小余量和 STL 碰撞检查周期。调整
碰撞/机械余量会在启动时重新生成并扫描整条轨迹。

## X：跑步测试

等待出现：

```text
robot reset 2 acknowledged; initialization complete
```

第一次按 X：

1. 用现有 `stop_ramp_sec=0.5` 秒平滑进入原跑步中心姿态；
2. 保持中心姿态 2 秒，跑步全过程使用共享的29关节 `JOINT_KP` 数组，不再将
   所有关节增益升到10000；
3. 再用 0.5 秒从中心姿态平滑接入 `data.txt` 第 1 帧；
4. 从第 1 帧开始以 50 Hz 推进轨迹。

每次重新进入跑步模式都会把轨迹索引复位到第 1 帧，不会从上次被 Y 中断的
中间帧直接恢复，避免中心姿态到轨迹中间位置的跳变。

轨迹跑完一轮进入停顿段时，原数据最后一帧到第 1 帧存在约 `0.145 rad` 的右踝
跳变。组合节点会用现有 `stop_ramp_sec=0.5` 秒对此循环边界做最小冲击融合，再
进入原来的第 1～46 帧停顿段；原始轨迹文件和各帧动作值不变。

再次按 X：停止轨迹并平滑回到跑步中心姿态。

## Y：振动测试

组合模式已取消旧的 29 关节振动预检测。按 Y 直接使用原有振动参数启动；再次
按 Y 停止激励并平滑回中。跑步或 A 键关节测试运行时，Y 会被拒绝，先停止当前
模式再启动振动。

实机零位可能存在小于 `0.001 rad` 的包络误差。组合模式只在 Y 启动判定中允许
该微小余量，振幅配置仍是 `0.23 rad`；最终关节指令继续由原软件位置限位裁剪，
不会越过软件边界。

## A：碰撞余量全行程关节测试

按一次 A 后，机器人先按峰值不超过 `10°/s` 的速度进入全零碰撞扫描基准姿态，
确认反馈稳定后自动依次执行：

1. 手臂：腕旋转、腕俯仰、腕侧摆、肘、肩旋转、肩侧摆、肩俯仰；
2. 躯干：腰旋转、腰侧摆、腰俯仰；
3. 腿部：踝侧摆、踝俯仰、膝、髋旋转、髋侧摆、髋俯仰。

每组走安全下限、上限并返回中心。肩部测试前自动收肘，髋部测试前自动屈膝，
测试完成后展开回全零姿态。左右侧摆/旋转采用镜像方向，避免两侧肢体相撞。

这是全行程，不是 ±5° 往复：机械限位保留 `2°`，已扫描的自碰撞边界再保留
`10°`。每段使用 minimum-jerk 曲线，较大行程会自动延长时间，因此完整流程会
持续数分钟。运行中再次按 A 会限速回到全零姿态。

A 停止并安全回到全零姿态期间可以直接按一次 Y；系统会记住请求，先完成 A 的
全零回位，再移动到原振动中心姿态并等待反馈稳定，之后自动启动振动。即使 A 已
经回到全零姿态，按 Y 也会先进入振动中心，不会在膝关节0位直接施加 ±0.23rad。
过渡期间再次按 Y 会取消待启动请求。

启动前会检查到全零姿态的路径；运行中同时检查命令姿态、实测姿态、简化碰撞
体和 STL 外观网格。一旦发现碰撞会进入安全故障闭锁。跟踪误差超过 `2°` 会记录
警告和失败计数。

## ROS 服务备用入口

跑步：

```bash
ros2 service call /run_trajectory_enable \
  std_srvs/srv/SetBool "{data: true}"

ros2 service call /run_trajectory_enable \
  std_srvs/srv/SetBool "{data: false}"
```

振动服务：

```text
/vibration_test_enable
```

组合模式中的旧 `/joint_rotation_test_enable` 预检测入口会明确拒绝启动；独立振动
launch 仍保留原预检测能力。

全行程关节测试：

```bash
ros2 service call /whole_body_joint_test_enable \
  std_srvs/srv/SetBool "{data: true}"
```

## 冲突保护

- 跑步、振动和全行程关节测试状态互斥，不会同时生成指令；
- 模式切换经过平滑中心姿态过渡；
- 两个手柄发布者同时存在时忽略 X/Y/A 输入；
- 两个执行器指令发布者同时存在时进入故障闭锁；
- reset、反馈新鲜度、命令间隔、有限值和关节位置保护沿用振动安全状态机；
- 故障锁存后不能通过再次按 X/Y/A 恢复，必须排查并重启。
