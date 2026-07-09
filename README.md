# bxi_rl_controller_ros2_example

[English](./README.en.md)

## 项目概览

本仓库提供 BXI 机器人控制器开发示例框架，包含：

* 基于强化学习控制策略的示例控制程序；
* 基于 ROS2 的 Mujoco 仿真环境；
* 基于 ROS2 的 BXI 硬件环境；
* 读取手柄或键盘输入的 ROS2 遥控器节点。

## 包结构

ROS2 环境和 Mujoco 的二进制 ROS2 包位于 [`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg)。

在二进制 ROS2 包 `bxi_ros2_pkg/` 目录中：

1. `communication`：机器人通信包，包含自定义通信消息格式。
2. `description`：机器人描述文件，包括 URDF、XML 和 mesh 文件。
3. `mujoco`：基于 ROS2 的 Mujoco 仿真器。控制程序建议先在 Mujoco 中验证，再部署到机器人硬件。
4. `hardware`：机器人硬件包。该节点发布机器人所有传感器数据，并接收控制命令。
5. `hardware_arm`：机器人上半身版本的硬件控制包。该节点发布机器人上半身手臂信息，并接收控制命令。

[`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) 是仿真和硬件共用的统一框架：

ROS2 仿真结构：![ROS2 structure simulation](docs/ROS2_structure_simulation.png)

ROS2 硬件结构：![ROS2 structure hardware](docs/ROS2_structure_hardware.png)

`src/` 目录包含：

1. `src/bxi_example_py_elf3`：Elf3 学习控制策略示例。
2. `remote_controller`：读取手柄或键盘输入并发布控制命令，可用于机器人硬件和仿真环境。

## 描述文件（URDF）

1. `elf3_dof29`：Elf3，29 自由度。
2. `elf3_dof31`：Elf3，31 自由度（头部 2 自由度）。

USD 或 XML 格式请参考：[unofficial models](https://github.com/MelodyAI/TienKung-Lab-bxi/tree/main/legged_lab/assets/elf3_lite)。

## 使用说明

### 中文扩展文档 / Wiki

新增遥控器/输入控制器、添加机器人业务状态、配置状态转移、过渡行为、`on_bind(ctx)` 状态订阅和 `get_cmd_vel(ctx)` 速度处理，请参考项目 Wiki：

* GitHub Wiki：<https://github.com/bxirobotics/bxi_rl_controller_ros2_example/wiki>

### 硬件和仿真环境切换

1. `hw` 是 `hardware` 的缩写，所有带 `hw` 后缀的 `launch` 文件都会启动真实硬件，请谨慎使用。
2. 仿真环境和机器人硬件共用同一套控制程序。只需要使用不同的 launch 文件即可在仿真和硬件之间切换。仿真代码的话题带有 `simulation/` 前缀，硬件话题带有 `hardware/` 前缀。详情请参考 `src/bxi_example_py_elf3` 中的话题参数配置。
3. 仿真环境中的机器人启动时带有虚拟悬挂，启动后需要释放悬挂。机器人硬件运行时会忽略悬挂相关信号。
4. 仿真环境中存在全局里程计话题 `odm`，硬件环境中没有该话题。

### 系统环境配置

1. 使用 `Ubuntu 22.04` 和 ROS2 `humble`。`mujoco` 需要安装 `libglfw3-dev`。
2. 遥控器默认使用设备路径 `/dev/input/jsBattleDragon`。个人开发者或本地调试环境需要先配置 udev 规则，才能稳定生成该设备别名。
3. 安装 Battle Dragon 手柄链接脚本并添加执行权限：

```bash
sudo install -m 0755 ./script/bxi-battle-dragon-link /usr/local/bin/bxi-battle-dragon-link
```

4. 复制 udev 规则，并重新加载规则：

```bash
sudo cp ./script/bxi-dev.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新插拔 Battle Dragon 手柄后，应能看到 `/dev/input/jsBattleDragon`。如果只是本地调试，也可以临时将 `src/remote_controller/config/xbox_default.yaml` 中的 `sources.gamepad.device` 改成当前系统对应的 `/dev/input/js0`。

5. 如需设置遥控器开机自启动，先根据实际路径和 ROS 配置编辑 `./script/ros_elf_launch.service`，然后复制到 `/etc/systemd/system/` 并启用服务：

```bash
sudo cp ./script/ros_elf_launch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ros_elf_launch.service
sudo systemctl status ros_elf_launch.service
```

### 在仿真器中运行示例控制程序

1. 将 ROS2 二进制包 [`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) 拉取到 `/opt/bxi/bxi_ros2_pkg`：

```bash
mkdir -p /opt/bxi/
cd /opt/bxi/
git clone https://github.com/bxirobotics/bxi_ros2_pkg.git
```

然后激活环境（在机器人硬件上请使用 `root` 用户执行）：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
```

2. 在 `bxi_rl_controller_ros2_example` 目录中运行 `bash build.sh`，编译 `./src` 目录下的所有源码。编译完成后运行 `source ./install/setup.bash` 激活当前包环境。
3. 运行全身控制策略：

```bash
ros2 launch bxi_example_py_elf3 example_launch_demo.py
```

该命令会启动仿真环境和基于学习的控制程序。

4. 启动键盘控制节点：

```bash
ros2 launch remote_controller remote_controller_keyboard.launch.py
```

### 在硬件中运行示例控制程序

0. 使用 `root` 用户登录。
1. 将 ROS2 二进制包 [`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) 拉取到 `/opt/bxi/bxi_ros2_pkg`：

```bash
mkdir -p /opt/bxi/
cd /opt/bxi/
git clone https://github.com/bxirobotics/bxi_ros2_pkg.git
```

然后激活环境（在机器人硬件上请使用 `root` 用户执行）：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
```

2. 在 `bxi_rl_controller_ros2_example` 目录中运行 `bash build.sh`，编译 `./src` 目录下的所有源码。编译完成后运行 `source ./install/setup.bash` 激活当前包环境。
3. 运行全身控制策略：

```bash
ros2 launch bxi_example_py_elf3 example_launch_demo_hw.py
```

该命令会启动机器人硬件和控制策略。

### 控制程序运行提示

1. 话题中的控制命令必须按照指定关节顺序发送。关节顺序请参考 `src/bxi_example_py_elf3` 中的示例。
2. 仿真环境和硬件机器人都有失控保护。如果控制命令丢失超过 100 ms，保护会被触发，电机将失能，系统必须重新初始化后才能继续使用。
3. 遥控器默认设备为 `/dev/input/jsBattleDragon`。如果 udev 规则尚未配置，或调试时只想使用系统当前识别到的手柄设备，可将配置中的设备路径临时改为 `/dev/input/js0`。

### 启动流程

在仿真和硬件环境中，启动时电机都处于失能状态，所有参数不可控。启动流程分为两步：

1. 使能电机位置控制。电机可通过设置 `pos kp kd` 三个参数实现位置控制。以 Elf2 硬件为例，收到第 1 次 `reset` 命令后，关节会转到零位并保持 10 秒。
2. 使能所有控制参数。电机可设置 `pos vel tor kp kd`。以 Elf2 硬件为例，收到第 2 次 `reset` 命令后，关节开始接收控制输入。

启动示例请参考 `src/bxi_example_py_elf3`。

### 硬件保护

除通信超时保护外，硬件节点还包含力矩保护、超速保护和位置保护。

1. 硬件节点内置错误计数器。当错误计数达到 `1000` 时，电机会退出使能状态。
2. 错误计数逻辑：收到电机速度超限时错误计数增加 `50`；收到力矩超限时错误计数增加 `100`；收到正常电机消息时错误计数减少 `1`，最小值为 `0`。
3. 触发位置超限保护时，不会增加错误计数。超限方向的控制会被禁止，电机只能向相反方向旋转。
4. 详细超限值请联系我们获取。不建议在非必要情况下修改这些数值。

## 重要提示

大型机器人可能带来风险。操作前请仔细确认说明！

所有控制程序必须先经过仿真验证，再部署到机器人硬件。

如发生任何异常，请立即按下急停按钮！
