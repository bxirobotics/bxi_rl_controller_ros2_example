# ELF3 SONIC 离线部署手册

这套流程用于 Ubuntu 22.04、x86_64、Python 3.10 的 ELF3 主控。目标是把同一个已
验证的 Git commit、同一组固定版本依赖部署到每台机器人，并继续覆盖平板 App 实际
使用的 `/opt/bxi/bxi_rl_controller_ros2_example`。

## 部署包边界

GitHub 只存源码、部署脚本和检查脚本。PICO/XRoboToolkit vendor 文件与大体积 wheel
不提交到 GitHub；它们由工作站上的 `prepare_robot_sonic_bundle.sh` 合入离线包。

离线包只保留实机路径需要的版本：

- `numpy==1.26.4`，不包含 NumPy 2.x；
- `torch==2.6.0+cpu`，不启用 CUDA；
- `scipy==1.15.3`、`pyzmq==27.1.0`、`msgpack==1.1.2`；
- `pin==2.7.0`、`eigenpy==3.5.1`、`hpp-fcl==2.4.4` 及其 cmeel 依赖；
- `onnxruntime==1.23.2` 及控制节点所需依赖；
- 离线 `pip`/`setuptools` 启动 wheel，不依赖目标机 apt 或公网；
- RoboticsService deb 与 `xrobotoolkit_sdk` CPython 3.10 扩展。

安装脚本分别处理两套 Python 环境：

1. PICO venv：`/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop`；
2. App 启动的控制节点环境：`/opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages`。

第二套环境会显式安装 `pyzmq`，避免 PICO 检查通过但
`bxi_example_py_elf3_demo` 因缺 `zmq` 退出。

## 工作站生成部署包

要求源码仓库没有未提交修改。执行：

```bash
cd /path/to/bxi_rl_controller_ros2_example

bash script/prepare_robot_sonic_bundle.sh \
  /tmp/elf3_sonic_runtime_deps_20260716 \
  "$HOME/elf3_sonic_artifacts"
```

脚本会：

- 用 `git archive HEAD` 固定源码 commit，并只导出实机部署需要的两个 ROS 包、部署脚本和
  许可证/归因文件；顶层 CAD `resources/` 等仓库开发资产不进入机器人 bundle；
- 只选择唯一的固定版本 wheel；
- 拒绝缺文件、重复版本或混入 NumPy 2.x 的 payload；
- 生成逐文件 `MANIFEST.sha256` 和整个 tgz 的 SHA256。

## 机器人部署前盘点

先把小型审计脚本传到机器人并执行，或者解包后运行：

```bash
bash source/script/audit_robot_sonic_host.sh
```

必须满足：

- x86_64、Python 3.10；
- `/opt/ros/humble` 和 `/opt/bxi/bxi_ros2_pkg` 存在；
- `/tmp` 至少约 1800 MiB 可用；
- `hardware_elf3`、控制节点和 PICO runtime 已停止。

审计只读取状态，不会停止进程或改机器人。

## 上传与一键部署

在工作站先核对并上传。上传命令保持为单行，且 TGZ 与校验文件必须一起上传：

```bash
sha256sum -c elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256
scp elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256 bxi@<robot>:/tmp/
```

在机器人上执行：

```bash
cd /tmp
sha256sum -c elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256
tar -xzf elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz

cd elf3_sonic_deploy_<commit>_ubuntu22_amd64
bash source/script/deploy_robot_sonic_bundle.sh "$PWD"
```

这是两台机器人已经验证的 `4f87e2c` 流程。执行前仍需人工停止控制 service、运行审计并
确认 apt/dpkg 锁为空；该部署器完成 manifest 校验、构建、备份、覆盖、离线依赖安装和静态
健康检查，但不会启动硬件或停止共享 gateway。

部署完成分两级判断：部署器输出 `deployment complete` 表示静态安装完成；终端能够从新
`/opt` 启动硬件/状态机并通过 SONIC live 检查，表示机器人端功能部署完成。App manifest
与命令透传是随后独立的产品集成验收，App 端问题不能反向判定离线安装失败。

## 对接机器人已有 systemd/App 启动入口

覆盖 `/opt` 后必须检查机器人真正使用的 remote-controller service。部分机器人仍从旧
workspace 的 `install/setup.bash` 启动；这种情况下终端会使用新 `/opt`，但 App 重启后仍会
进入旧代码：

```bash
sudo systemctl show ros_elf_launch.service \
  -p ExecStart -p Environment --no-pager
sudo SYSTEMD_PAGER=cat systemctl cat --no-pager ros_elf_launch.service
```

对 `4f87e2c` 包，若 effective `ExecStart` 不是新 `/opt`，人工安装只覆盖启动命令的
drop-in：

```bash
sudo install -d /etc/systemd/system/ros_elf_launch.service.d
sudo tee /etc/systemd/system/ros_elf_launch.service.d/sonic-runtime.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/bin/bash -lc "source /opt/ros/humble/setup.bash && source /opt/bxi/bxi_ros2_pkg/setup.bash && source /opt/bxi/bxi_rl_controller_ros2_example/setup.bash && exec ros2 launch remote_controller remote_controller.launch.py"
EOF
sudo systemctl daemon-reload
```

不要在 drop-in 中硬编码另一台机器的 `ROS_DOMAIN_ID`、RMW 或 CycloneDDS 网卡配置；原
service 的 Environment 应继续生效。完成终端验收前保持该 service 停止，验收和进程清理
通过后再恢复 App。

### 下一版一入口安装器（本地候选，尚未进入 `4f87e2c`）

当前本地工作区新增候选入口：

```bash
bash source/script/install_robot_sonic_bundle.sh "$PWD"
```

按提示输入 `DEPLOY <当前主机名>` 后，它会校验包内 manifest、验证目标 unit 确为
remote-controller、只停止该控制 service、监护共享 gateway、拒绝残留进程/硬件锁/dpkg
锁，并在安装期间持有统一硬件锁；随后配置只覆盖 `ExecStart` 的 drop-in，并验证 effective
Environment 没有变化。`--terminal-only` 会跳过 App drop-in，`--service <name>` 仅接受实际
启动 `remote_controller.launch.py` 的 unit。

这一入口不会上传/解包，不会等待或强杀 apt/dpkg，不会修改现场 ROS Domain，也不会启动
service、上电硬件、执行动作或代替 PICO/App 验收。它已通过本地测试，但必须先从新 commit
重建 bundle 并完成真机“启动 -> 停止 -> 再启动”回归，才能替代上面的 `4f87e2c` 流程。

## 成功标准

静态部署至少需要全部通过：

- `ros2 pkg prefix bxi_example_py_elf3` 指向 `/opt/bxi/bxi_rl_controller_ros2_example`；
- `sonic_pico_runtime_supervisor` 已安装；
- PICO venv 的 NumPy、SciPy、torch CPU、pinocchio、XRT 全部可导入；
- 控制节点 Python 可导入 `numpy`、`onnxruntime`、`zmq` 和 SONIC policy；
- `RoboticsServiceProcess` 与 `SDK/x64/libPXREARobotSDK.so` 存在；
- 记录 `8081` owner 作为诊断基线，但不把 gateway 占用本身当成失败条件。

动态验证时还必须看到：

```text
Body data available
ZMQ socket bound to port 5556
Calibration completed
StreamMode switch: OFF -> PLANNER
StreamMode switch: PLANNER -> POSE
[pico->smpl_ref] stream ready ...
[SONIC] reference status: live_reference
```

并确认 `5556 pose`、`5557 smpl_ref` 均持续约 50 Hz，SONIC 状态为
`live_reference`。健康稳态不周期打印 FPS；低于 45 Hz 才打印
`PICO RATE WARNING`，恢复时打印一次 `PICO RATE RECOVERED`。一次姿势变化不等于实时遥操成功。

进入 POSE/live 后可直接运行离线包内的一键检查：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
$PY source/script/check_sonic_live_stream.py --duration 5
```

## 回滚

代码覆盖前会在这里生成带时间戳的备份：

```text
/opt/bxi/deploy_backups/bxi_rl_controller_ros2_example.before_sonic_*.tgz
```

如需回滚，先停止控制进程，再把目标备份解压回 `/opt/bxi`。不要在电机主控运行时
覆盖或回滚 `/opt`。
