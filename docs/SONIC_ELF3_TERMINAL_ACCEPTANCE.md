# ELF3 SONIC 平板前终端验收

目标是在不依赖平板启动控制框架的情况下，先从远程终端验证硬件主控、状态机、
SONIC fallback、自动 PICO runtime、5556/5557 数据流和退出清理。完成后再停止终端
进程，使用平板重复同一条状态链路。

## 安全前提

- 机器人可靠悬挂或处于经过批准的安全测试姿态，急停可用；
- 操作人员确认周围无人，先测试状态机和日志，再做连续动作；
- 不在硬件主控运行时覆盖 `/opt`；
- 不同时运行自动 supervisor 和 `run_sonic_pico_sources.sh`；
- 不因为看到 `8081` 被 gateway 占用就自动杀 gateway。

## 0. 静态与残留检查

先执行：

```bash
BUNDLE=/tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64

bash "$BUNDLE/source/script/audit_robot_sonic_host.sh"
bash "$BUNDLE/source/script/check_robot_sonic_runtime.sh"
```

记录当前进程和端口：

```bash
ps -ef | grep -E 'robot_gateway|hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager|pico_pose|RoboticsServiceProcess' | grep -v grep || true
sudo ss -lntup | grep -E ':(8081|60061|5556|5557)\b' || true
```

开始 T1 前不能已有另一份 `hardware_elf3`、controller 或 PICO manager。若硬件日志出现
`bxi pci busy`，先定位现有主控所有者，不要并行启动第二份硬件节点。

## 1. T1：硬件主控、状态机和自动 supervisor

root 终端执行：

```bash
BUNDLE=/tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64
export ROS_DOMAIN_ID="$(systemctl show ros_elf_launch.service -p Environment --value | tr ' ' '\n' | sed -n -E 's/^"?ROS_DOMAIN_ID=([0-9]+)"?$/\1/p' | head -n 1)"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
bash "$BUNDLE/source/script/run_robot_sonic_hw.sh"
```

必须看到：

```text
state graph loaded
robot reset 1!
robot reset 2!
watching hardware/state_machine_info for states=sonic_teleop,sonic_teleop_gripper
```

健康的约 50 Hz 控制稳态不再周期打印。只有控制频率低于 45 Hz、最大帧间隔超过
50 ms，或超过 25 ms 的帧占比高于 10% 时才打印 `[CONTROL RATE WARNING]`；
恢复后打印一次 `[CONTROL RATE RECOVERED]`。

若出现 `No module named zmq`、`can init failed`、`bxi pci busy` 或 controller exit code 1，
不要继续切状态。

## 2. T2：终端状态控制

另一个终端执行：

```bash
BUNDLE=/tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64
export ROS_DOMAIN_ID="$(systemctl show ros_elf_launch.service -p Environment --value | tr ' ' '\n' | sed -n -E 's/^"?ROS_DOMAIN_ID=([0-9]+)"?$/\1/p' | head -n 1)"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
bash "$BUNDLE/source/script/run_robot_sonic_controller.sh"
```

在确认机器人安全后，按既定顺序测试：

```text
! -> pd_brake
1 -> normal
6 -> sonic_teleop（默认，不带夹爪）
g -> sonic_teleop_gripper（带夹爪，仅空载确认后测试）
```

进入 `sonic_teleop` 后，T1 应继续以约 50 Hz 控制，并先报告
`idle_reference`。这证明无 live 数据时的安全 fallback 正常，但尚不能证明 PICO 链路。
两种模式不能直接互切；先回到 `normal`，再选择另一模式。

## 3. T3：进程和端口观察

第三个终端执行：

```bash
watch -n 1 "ps -ef | grep -E 'hardware_elf3|bxi_example_py_elf3_demo|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' | grep -v grep; echo PORTS; sudo ss -lntup | grep -E ':(8081|60061|5556|5557)\\b' || true"
```

进入 SONIC 后应只有一份 manager 和一份 bridge。`8081` 只记录，不作为单项失败条件。

## 4. T4：ROS 状态机观察

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /opt/bxi/bxi_rl_controller_ros2_example/setup.bash

ros2 topic echo /hardware/state_machine_info
```

依次确认 `pd_brake`、`normal`、`sonic_teleop`；空载夹爪验收时再确认
`sonic_teleop_gripper`。如果终端看不到 topic，先核对
`ROS_DOMAIN_ID`/RMW；不要仅凭这个现象否定正在运行的主控，可同时参考 T1 当前日志。

## 5. T5：PICO 与 live 数据

PICO 连接机器人 IPv4 并 send。成功顺序必须是：

```text
Body data available
ZMQ socket bound to port 5556
ABXY -> Calibration completed / OFF -> PLANNER
A+X -> PLANNER -> POSE
[pico->smpl_ref] stream ready ...
[SONIC] reference status: live_reference
```

健康的 PICO 约 50 Hz 稳态不再周期打印 FPS。低于 45 Hz 时打印
`PICO RATE WARNING`，恢复后打印一次 `PICO RATE RECOVERED`；POSE 输入超过
默认 0.2 秒未更新时打印 `PICO pose input stale` 并停止 live 输出。

带夹爪模式必须先松开左右 trigger。收到本次会话的新 trigger 数据后，应只看到一次：

```text
SONIC夹爪已解锁：左右电机进入motor mode
```

左、右 trigger 分别比例控制左、右夹爪。PICO trigger 短暂断流时，夹爪保持
最后有效目标，只在断流和恢复边沿各播报一次；不得自动把缺失输入当成 `0`。

然后使用部署报告第 3.7 节的采样命令确认：

- `5556 pose`：约 50 Hz、`mode=1`、`calib=true`、frame 持续增长；
- `5557 smpl_ref`：约 50 Hz、`ready=true`、`mode=1`、frame 持续增长；
- policy 日志：`reference status: live_reference`；
- 机器人连续跟随动作。

也可以直接用部署包内的一键检查完成两路采样：

```bash
BUNDLE=/tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
$PY "$BUNDLE/source/script/check_sonic_live_stream.py" --duration 5
```

命令必须以 `pose` 和 `smpl_ref` 两项 `[PASS]` 结束；否则不要继续扩大动作幅度。

若 PICO 报 socket error，先判断 `Body data available` 和 `5556` 是否出现，再结合当前
进程、当前日志和抓包定位；不要只根据 `8081` owner 下结论。

## 6. 退出与清理验收

从 SONIC 切回 `normal`，确认 manager、bridge 和其子进程退出，5556/5557 消失；
supervisor 本身仍由 T1 launch 管理。随后先停止 T2，再 Ctrl+C 停止 T1。

```bash
pgrep -af 'hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' || true
sudo ss -lntup | grep -E ':(60061|5556|5557)\b' || true
```

清理通过后再由平板启动控制框架，分别确认 `SONIC遥操`（`btn_10=7`）和
`SONIC遥操（夹爪）`（`btn_10=8`）两个入口。夹爪入口仅做空载测试；两种模式的
身体控制都必须保持 29 维，并以 5556/5557 持续流动和 `live_reference` 为最终标准。
