# SONIC 在 ELF3 上的部署与验收

本文记录 `com.bxi.sonic` 在当前 ELF3 框架中的实际部署方式和验收边界。算法实现、第三方
运行时裁剪依据及性能数据见同目录的 [README.md](README.md)，Mod 的状态、节点和参数定义
以 [mod.yaml](mod.yaml) 为唯一准则。

## 当前实现

SONIC 已作为标准 Mod 接入主控制器，不再使用独立 supervisor、额外的 T3 终端或
`pico_runtime.py`。它只注册一个状态：

```text
com.bxi.sonic/sonic_teleop
```

该状态执行同一套 29 关节 SONIC 策略，并在 PICO `POSE` 模式下追加两个具名头部
关节命令；`hardware_gripper: true` 时同时接管左右夹爪。不再提供单独的
`sonic_teleop_gripper` 状态。进入和退出 SONIC 使用 `soft_switch`，策略切换仍由
框架的两阶段准备和控制线程内切换机制完成。

控制数据路径为：

```text
PICO 头显/追踪设备
  -> RoboticsServiceProcess
  -> xrobotoolkit_sdk
  -> pico_manager（ZMQ pose）
  -> smpl_bridge（ZMQ smpl_ref）
  -> SonicTeleopPolicy
  -> 29 关节策略命令 + head_y_joint/head_z_joint
  -> ELF3 具名 MotorFrame
```

PICO 头部控制与 `com.bxi.pico_gmr_motion` 使用同一套语义：

```text
relative_head = inverse(Spine3) * Head
head_y_joint = -relative XYZ roll
head_z_joint =  relative XYZ pitch
```

`states.sonic_teleop.params.head_control_enabled` 默认为 `true`。关闭后状态的自然输出
从 31 关节恢复为策略原生的 29 个身体关节，不再创建或更新 PICO 头部命令层。

每次进入 `POSE` 都重置中心姿态。头部目标作为 `head_joint_pos[*,2]` 从 manager 经
bridge source chunk 与身体一起传给 policy 的十帧窗口；首次 live 前或显式状态重置时
目标回零，live 断流后则与身体一起保持最后完整窗口。策略模型输出仍为 29 关节，状态
才合成为 31 关节具名帧；
29 关节机器人会按名称裁掉不存在的头部关节，31 关节机器人则完整接收。

头部参数默认值与 PICO GMR 一致：

| 参数 | 默认值 |
| --- | ---: |
| `head_pitch_limit_rad` | `0.5` |
| `head_yaw_limit_rad` | `1.0` |
| `head_pitch_speed_rad_s` | `1.5` |
| `head_yaw_speed_rad_s` | `2.0` |
| `head_deadband_rad` | `0.015` |

头部 PD 增益为 `kp=16.747` / `kd=1.066`。

进入状态后还会并行启动独立的头部相机图传路径：

```text
simulation/hardware head camera Image
  -> head_camera_rtsp_node
  -> MediaMTX
  -> rtsp://<机器人IP>:2212/video
```

模型和固定参考数据随 Mod 安装：

```text
assets/sonic.onnx
assets/stream_reference.npz
```

策略通过框架统一推理接口打开 ONNX 模型，后端选择顺序为：

```text
RKNN -> OpenVINO -> ONNX Runtime
```

不再支持 `BXI_SONIC_MODEL_ONNX`、`BXI_SONIC_STREAM_REFERENCE_NPZ` 等算法环境变量；模型、
reference 和夹爪行为均由 Mod 资源与 `mod.yaml` 明确决定。

## 进程和生命周期边界

`mod.yaml` 声明四个 state-scoped 节点：

```text
ModNodeManager
├── mediamtx_server
│   └── runtime: command / execution: process
├── head_camera_rtsp
│   ├── runtime: executable / execution: process
│   └── depends_on: mediamtx_server
├── pico_manager
│   ├── runtime: command / execution: process
│   └── runtime_profile: pico_bootstrap
└── smpl_bridge
    ├── runtime: python / execution: in_process
    ├── runtime_profile: host_ros
    └── depends_on: pico_manager
```

- `pico_manager` 在独立选择并清理过环境变量的 Python 中运行，持有
  `xrobotoolkit_sdk`，并负责启动和回收它所使用的 `RoboticsServiceProcess`。
- `pico_bootstrap` 不预注入 Mod 内的厂商路径；manager 会先检查用户安装，失败后才启用
  当前平台的内置回退。
- `smpl_bridge` 是宿主 ROS executor 内的原生节点，不继承厂商 Python、SDK 或动态库环境。
- 四个节点均为 `lifecycle: state`，只在准备或运行 `sonic_teleop` 时存在；离开状态后由
  框架按依赖逆序回收。
- manager 普通运行故障最多重启 3 次；依赖或解释器配置错误使用退出码 `78`，框架将其
  视为确定性 fault，不进行无意义重启。
- manager 进程组关闭顺序为 `SIGINT`，3 秒后 `SIGTERM`，5 秒后 `SIGKILL`，其派生的
  RoboticsService 也在同一所有权边界内。

RoboticsService 保留独立进程是上游 SDK 的运行模型：Python binding 原本就通过本机
gRPC 使用该服务。把它做成 manager 持有的子进程不增加 SONIC 的观测—推理—输出数据层级，
也不要求用户预先启动 systemd 服务。

## Mod 内置依赖

当前目录同时携带 Linux x86_64 与 ARM64 产物：

```text
assets/
config/
pico/
runtime/
  mediamtx.yml
  linux-x86_64/mediamtx
  linux-aarch64/mediamtx
  linux-x86_64/roboticsservice/
  linux-aarch64/roboticsservice/
bin/
  linux-x86_64/head_camera_rtsp_node
  linux-aarch64/head_camera_rtsp_node
vendor/
  python/
    linux-x86_64-cpython-310/
    linux-aarch64-cpython-310/
  lib/
    linux-x86_64/libPXREARobotSDK.so
    linux-aarch64/libPXREARobotSDK.so
  licenses/
```

其中：

- `runtime/<platform>/roboticsservice` 是 XRoboToolkit PC Service 的最小可运行闭包，包含
  可执行文件、业务库、SDK 库及其实际需要的 Qt Core/Network/Core5Compat、ICU 等私有
  动态库。所谓 headless 包仍直接依赖这些 Qt 基础库，不能继续删除而保持 ABI 可用。
- `vendor/python/<platform>-cpython-310` 是 `xrobotoolkit_sdk==1.0.2` wheel 安装后的二进制
  扩展产物，目标机无需重新下载 wheel 或编译 binding。
- `vendor/lib/<platform>/libPXREARobotSDK.so` 是指向对应 service runtime 的相对符号链接，
  不复制大文件，普通安装和 `--symlink-install` 均保持有效。
- 用户环境可用时不会注入任何内置 binding；只有用户环境探测全部失败后，才会按当前
  OS、CPU 架构及 CPython ABI 注入对应回退，不会把另一架构的 `.so` 暴露给解释器。

内置回退 binding 当前要求 Linux 和 CPython 3.10。ARM64 上使用回退时必须选择 AArch64
Python 3.10；其他架构或 ABI 可以自行安装兼容的 `xrobotoolkit_sdk` 和 PC Service，无需
删除 Mod 内不匹配的平台文件。

## 通用 Python 依赖

manager 还需要 `requirements-pico.txt` 中的通用 Python 包：

```text
numpy>=1.26,<2
scipy>=1.10
pyzmq>=25
msgpack>=1.0
pin>=2.7
```

这些包体积较大，不作为跨平台 site-packages 直接提交，而由部署脚本安装到目标平台的
独立环境。脚本不会调用 apt、修改 systemd，也不会安装 Torch 或 CUDA。

在 Mod 源目录执行：

```bash
cd src/bxi_example_py_elf3/mods/com.bxi.sonic

# 只检查，不修改文件或环境
./deploy_dependencies.sh --check

```

当前仓库已经包含 x86_64 和 ARM64 runtime，正常目标机不需要重复此步骤。该命令只适用于
生成或更新尚不存在的平台目录，并要求 `ldd`、`readelf` 和 `patchelf`。

SONIC 只保留三个部署环境变量：

| 变量 | 用途 |
| --- | --- |
| `SONIC_PICO_PYTHON` | 固定 manager 使用的 Python 解释器 |
| `SONIC_XRT_SERVICE_DIR` | 显式指定用户 RoboticsService 根目录 |
| `SONIC_MEDIAMTX_BIN` | 可选；覆盖 Mod 内置 MediaMTX 路径 |

Service 查找顺序是：显式 `SONIC_XRT_SERVICE_DIR`、用户安装的
`/opt/apps/roboticsservice`、当前平台内置 runtime。用户路径一旦存在便具有权威性；若其
损坏或 ABI 不兼容，启动器会明确报错，不会悄悄换成内置版本。其他架构的外部 service
可通过其 `SDK/*/libPXREARobotSDK.so` 自动发现 SDK 目录。

## 构建

从工作区根目录执行：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg-main/setup.bash
colcon build --packages-select bxi_example_py_elf3 remote_controller
source install/setup.bash
```

若使用普通非 symlink 安装，应在构建前准备好需要随安装树部署的 Mod 本地 `.runtime`；
内置的 `assets/`、`runtime/`、`vendor/` 和相对符号链接会由包安装逻辑完整复制。

## 启动

仿真控制器：

```bash
ros2 launch bxi_example_py_elf3 example_demo.launch.py
```

硬件控制器：

```bash
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

遥控器可另开终端启动。正常输入设备自动选择：

```bash
ros2 launch remote_controller remote_controller.launch.py
```

只使用键盘调试：

```bash
ros2 launch remote_controller remote_controller_keyboard.launch.py
```

以上命令使用项目当前的 `remote_controller/config/xbox_default.yaml`。SONIC 映射为：

| 输入 | 操作 |
| --- | --- |
| 键盘 | `@` |
| Xbox/兼容映射 | `LB + RB + X` |
| 状态机事件 | `btn_10=9` |

应先进入 `normal`，确认机器人站立和周围安全，再请求 SONIC。离开时可请求 `normal`、
`pd_brake`、`recover` 或 `zero_torque`。无肩键和 trigger 修饰时按手柄 `X` 会发出
`btn_9=1`，在 SONIC 内对应“重置朝向对齐”。

## PICO 操作和 reference 行为

进入 SONIC 后，PICO manager 和 bridge 由状态生命周期自动启动：

1. 保持机器人和操作者处于预期的校准姿态，同时按下 PICO 控制器
   `A+B+X+Y` 请求校准。
2. 如果此时身体追踪数据还未到达，请求会被保留；第一帧新鲜 body 数据到达后自动完成，
   不需要反复按键。
3. 校准完成后同时按 `A+X`，从 PLANNER 切换到实时 POSE。
4. bridge 收到连续、有限、帧号前进且标记校准完成的 POSE 数据后，逐包转发完整
   source chunk；policy 经过 `source_blend_seconds` 平滑切入 live reference，并独占
   十帧窗口的播放游标。

当前 `require_live_reference: false`，因此 PICO 尚未就绪时 SONIC 仍使用
`assets/stream_reference.npz` 中固定的 idle window 运行推理，不会卡在非策略默认姿态。
首次 live 窗口到达后，policy 每次成功推理最多推进一帧；缺少新 chunk 时先消费剩余
缓冲，随后持续保持最后一个完整十帧窗口。它不会尾帧 clamp，也不会因短时断流退回 idle。
`live_reference_timeout_s` 只用于新鲜度状态和告警；重新进入或显式重置状态才清除上一
会话的 live packet，并重新使用自采站姿 reference。

`A+B+X+Y` 完成的是 PICO 三点追踪与 ELF3 FK 参考的对齐，同时作为操作者就绪握手；
`calibration_ready=true` 不应解释为对 SONIC 原始 SMPL tensor 的另一套数值标定。

## 当前状态参数

以下值来自当前 `mod.yaml`：

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `require_live_reference` | `false` | 无 live 输入时允许使用 idle reference |
| `yaw_bias_rad` | `1.57079632679` | PICO 朝向的 yaw 偏置 |
| `live_reference_timeout_s` | `0.5` | live reference 新鲜度阈值 |
| `idle_frame_start` | `3509` | 固定 idle window 起始帧 |
| `source_blend_seconds` | `0.4` | idle/live 切换混合时间 |
| `hardware_gripper` | `true` | 当前 SONIC 状态允许控制硬件夹爪 |
| `gripper_enable_interval_s` | `1.0` | 解锁后重新发送 `enter_motor_mode` 的周期 |
| `gripper_left_bus` | `5` | 左夹爪 CAN 总线 |
| `gripper_right_bus` | `6` | 右夹爪 CAN 总线 |
| `gripper_can_id` | `1` | 夹爪电机 CAN ID |
| `gripper_master_id` | `17` | 夹爪响应帧仲裁 ID |
| `gripper_kp` | `10.0` | 校准完成后的夹爪位置环 KP |
| `gripper_kd` | `0.5` | 校准完成后的夹爪位置环 KD |
| `gripper_calibration_speed_rad_s` | `0.2` | 每次进入状态时寻找限位的低速角度斜坡 |
| `gripper_calibration_kp` | `5.0` | 校准低位置增益 |
| `gripper_calibration_kd` | `0.5` | 校准速度增益 |
| `gripper_contact_torque` | `2.0` | 机械限位接触力矩阈值 |
| `gripper_abort_torque` | `8.0` | 校准立即中止力矩阈值 |
| `gripper_contact_confirm_s` | `0.25` | 限位条件连续确认时间 |
| `gripper_stopped_velocity_rad_s` | `0.1` | 接触时最大实测速度 |
| `gripper_tracking_error_rad` | `0.08` | 接触/回退判定角度误差 |
| `gripper_limit_margin_rad` | `0.15` | 从机械限位向内回退距离 |
| `gripper_minimum_span_rad` | `1.0` | 最小合法软限位行程 |
| `gripper_maximum_search_travel_rad` | `7.0` | 单方向最大校准行程 |
| `gripper_response_timeout_s` | `1.0` | 首个响应帧超时，超时即认为电机离线 |
| `gripper_feedback_timeout_s` | `0.3` | 运行中响应帧断流超时 |
| `gripper_phase_timeout_s` | `45.0` | 单个校准阶段超时 |
| `gripper_maximum_mos_temperature_c` | `80` | 驱动 MOS 温度上限 |
| `gripper_maximum_motor_temperature_c` | `80` | 电机线圈温度上限 |

硬件运行前必须在目标机器人上确认左右总线号、方向、力矩阈值和增益。每次进入状态时
立即对两侧发送 `enter_motor_mode`，订阅 `/canfd_packet/rx`，然后依次执行“寻找张开限位、
回退、寻找闭合限位、回退、返回张开位置”。没有合法响应帧即认为对应电机离线；任一侧
失败时左右夹爪都退出电机模式。校准完成前 trigger 不接管夹爪，完成后按各侧实测软限位
映射 trigger。

## 内部端口和诊断

Mod 内部默认只监听回环地址：

| 数据流 | 地址 | topic |
| --- | --- | --- |
| manager → bridge | `127.0.0.1:5556` | `pose` |
| bridge → policy | `127.0.0.1:5557` | `smpl_ref` |
| XR SDK → RoboticsService | `127.0.0.1:60061` | gRPC |

依赖检查：

```bash
cd src/bxi_example_py_elf3/mods/com.bxi.sonic
./deploy_dependencies.sh --check

# 检查指定环境
./deploy_dependencies.sh --check \
  --python /path/to/python \
  --service-dir /path/to/roboticsservice
```

运行时检查：

```bash
pgrep -af 'manager_launcher|RoboticsServiceProcess'
ss -lntup | grep -E ':(5556|5557|60061)\b'
ros2 topic echo --once /simulation/state_machine_info std_msgs/msg/String
```

硬件启动时状态话题使用 `/hardware/state_machine_info`。manager 会启动并持有自己的
RoboticsService，因此运行前不应另行启动 service 或占用 60061。离开 SONIC 后，manager、
RoboticsService 以及本次创建的 5556/5557/60061 监听都应被回收。

常见日志含义：

- `ABXY accepted; calibration requested`：组合键请求已被 manager 接受。
- `Waiting for body tracking data` / `no fresh body frame`：控制器按钮可读，但头显 body
  tracking 流尚未到达；应检查头显端 Body Tracking、目标 PC IP 和发送状态。
- manager 退出码 `78`：解释器、binding、SDK 或 service runtime 配置不完整。
- `idle_reference`：当前使用内置固定参考。
- `stale_hold`：live source 已过期，policy 正保持最后一个完整十帧窗口。

## 验收清单

1. `deploy_dependencies.sh --check` 通过，且报告当前平台 binding、必需 API、service
   executable 和动态库闭包均可用。
2. 构建后启动仿真；先进入 `normal`，再用 `@` 或 `LB+RB+X` 进入 SONIC。
3. 未连接 PICO 时确认策略继续使用 `idle_reference` 推理，控制器不因等待 live 数据阻塞。
4. 连接 PICO 后完成 `A+B+X+Y -> A+X`，确认切到 live reference，并观察动作方向和
   朝向是否正确。
5. 中断 PICO 数据，确认先消费剩余缓冲并保持最后完整窗口；恢复数据后继续顺序消费。
6. 在空载和安全条件下验证夹爪 trigger 松开门槛、左右映射、输入短暂中断和状态退出。
7. 离开 SONIC，确认 manager、bridge 和 RoboticsService 均按生命周期清理；重新进入后
   能重新创建且不发生端口冲突。
8. 关闭整个 launch，确认没有本次运行遗留的进程组或监听端口。

## 当前验证状态

截至 2026-07-31，当前实现已完成：

- 89 项框架自动化测试通过；
- 普通非 symlink `colcon build` 通过；
- 本机用户 Miniconda 环境中的 `xrobotoolkit_sdk` 优先选择验证通过，未注入内置 binding；
- x86_64 完整依赖检查、binding 导入和必需 API 校验通过；
- 安装树中的相对 SDK 符号链接验证有效；
- ARM64 binding 和 RoboticsService runtime 均确认是 AArch64 ELF；
- x86_64/ARM64 ELF 中的绝对构建机 RUNPATH 已规范化为 `$ORIGIN` 相对路径。

尚未闭环的项目是 ARM64 ELF3 真机上的完整导入、RoboticsService 启动、PICO body 数据链、
live/idle 切换和夹爪安全验收。在这些步骤完成前，只能认为双架构部署材料已经准备好，不能
把 ARM 真机链路标记为最终验收通过。
