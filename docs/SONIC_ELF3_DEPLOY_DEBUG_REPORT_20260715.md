# ELF3 SONIC/PICO 部署 Debug 报告

日期：2026-07-15
目标：机器人切到 `sonic_teleop` 后自动启动 PICO runtime，PICO 连接机器人，ABXY 完成校准，A+X 切到 POSE/live，最终由 PICO 实时驱动 SONIC 遥操。

## 1. 最终跑通状态

最终成功日志关键点如下：

```text
[Manager] Body data available after 12.99s
[Manager] ZMQ socket bound to port 5556
[PoseLoop] Robot model loaded for FK calibration
[PoseLoop] Calibration completed (zero-pose reference)
[Manager] StreamMode switch: OFF -> PLANNER
[Manager] StreamMode switch: PLANNER -> POSE
[PoseLoop] FPS: 50.25
```

最终 live 跟随还需要确认 `5556 pose` 和 `5557 smpl_ref` 都持续 50 Hz 左右输出：

```text
pose count ... mode 1 calib True frame_delta 1
smpl count ... ready True mode 1 frame_delta 1
```

这说明完整链路已打通：

```text
PICO
  -> RoboticsServiceProcess / xrobotoolkit_sdk
  -> pico_manager_legacy
  -> 5556 pose
  -> pico_pose_to_smpl_ref_bridge
  -> 5557 smpl_ref
  -> SonicTeleopPolicy live_reference
```

## 2. 主要问题和根因

### 2.1 缺 Python 依赖：torch

现象：

```text
ModuleNotFoundError: No module named 'torch'
```

处理：

机器人无 CUDA，安装 CPU 版 torch 即可。后续应使用 PICO 专用 venv，而不是系统 Python。

### 2.2 缺 XRoboToolkit SDK / RoboticsService

现象：

```text
ImportError: XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager.
```

或：

```text
ImportError: libPXREARobotSDK.so: cannot open shared object file: No such file or directory
```

处理：

补齐：

- `/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop`
- `xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so`
- `/opt/apps/roboticsservice/RoboticsServiceProcess`
- `/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so`

注意：第二种报错不是 Python binding 缺失，而是 binding 的 native 依赖库不在动态链接器搜索路径里。`sonic_pico_runtime_supervisor` 必须在启动 PICO manager/bridge 子进程前把 RoboticsService 库目录加入 `LD_LIBRARY_PATH`，至少包含：

```text
/opt/apps/roboticsservice/SDK/x64
/opt/apps/roboticsservice
/opt/apps/roboticsservice/lib
```

验证：

```bash
LD_LIBRARY_PATH=/opt/apps/roboticsservice/SDK/x64:/opt/apps/roboticsservice:${LD_LIBRARY_PATH:-} \
/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python - <<'PY'
import xrobotoolkit_sdk
print("xrt OK", xrobotoolkit_sdk)
PY

ls -lh /opt/apps/roboticsservice/RoboticsServiceProcess
ls -lh /opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so
```

### 2.3 ROS setup 在 `set -u` 下报 `AMENT_TRACE_SETUP_FILES: unbound variable`

现象：

```text
/opt/ros/humble/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable
```

处理：

部署脚本需要兼容 ROS setup 对未定义变量的使用。已在 helper scripts 中修正，不应在 `set -u` 状态下直接 source ROS setup。

### 2.4 PICO manager 默认带 `--cuda`

现象：

机器人端没有 CUDA 硬件，`--cuda` 启动可疑，也不符合部署预期。

处理：

`runtime_supervisor.py` 改为默认 CPU，仅在显式设置环境变量时启用 CUDA：

```python
if env_flag_enabled(env, "SONIC_PICO_USE_CUDA", default=False):
    manager.append("--cuda")
```

验证：

```bash
grep -n -- '--cuda\|SONIC_PICO_USE_CUDA' \
  /opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages/bxi_example_py_elf3/sonic_pico/runtime_supervisor.py
```

期望只看到条件追加 `--cuda`，不能固定带 `--cuda`。

### 2.5 控制节点环境缺 `zmq`

现象：

```text
[bxi_example_py_elf3_demo-2] ModuleNotFoundError: No module named 'zmq'
[ERROR] [bxi_example_py_elf3_demo-2]: process has died [exit code 1]
```

这时 `hardware_elf3` 和 `sonic_pico_runtime_supervisor` 可能还活着，但真正发布控制命令、驱动状态机的 `bxi_example_py_elf3_demo` 已经退出。表现为机器人控制框架没有接上，`/hardware/state_machine_info` 可能没有发布，或者状态切换无法进入完整 SONIC 控制链路。

原因：

`bxi_example_py_elf3_demo` 是 ROS install 里的 console script，通常由 `/usr/bin/python3` 加 `/opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages` 启动；它不是 PICO 专用 venv。即使 PICO venv 检查通过，也只能证明 PICO manager/bridge 的 venv 依赖完整，不能证明控制节点环境有 `zmq`。

修复：

优先从离线 wheelhouse 安装到 `/opt` install 的 site-packages：

```bash
PYVENV=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
TARGET=/opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages

ZMQ_WHL=$(find /tmp/elf3_sonic_runtime_deps_20260716/wheels -maxdepth 1 -type f -iname '*zmq*.whl' | head -1)
sudo "$PYVENV" -m pip install --no-index --no-deps --target "$TARGET" --upgrade "$ZMQ_WHL"
```

如果临时目录或 wheelhouse 已不存在，可从已验证的 PICO venv 复制：

```bash
PYVENV_SITE=$(/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

TARGET=/opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages

sudo cp -a "$PYVENV_SITE"/zmq "$TARGET"/
sudo cp -a "$PYVENV_SITE"/pyzmq-*.dist-info "$TARGET"/
sudo cp -a "$PYVENV_SITE"/pyzmq.libs "$TARGET"/ 2>/dev/null || true
```

验证控制节点环境，而不是 PICO venv：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /opt/bxi/bxi_rl_controller_ros2_example/setup.bash

python3 - <<'PY'
import importlib
for name in ["numpy", "onnxruntime", "zmq", "bxi_example_py_elf3.inference.sonic"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"), getattr(mod, "__file__", "unknown"))
PY
```

修复成功后，终端启动 `run_robot_sonic_hw.sh` 应看到：

```text
[bxi_example_py_elf3_demo]: state graph loaded
robot reset 1!
robot reset 2!
[CONTROL RATE] state=zero_torque, hz=50.0
```

### 2.6 PICO 能连接但一 send 就 socket error

这是本次最关键问题。

现象：

- 机器人可以切到 `sonic_teleop`
- SONIC idle 姿势正常
- PICO 显示能 connect
- 但 PICO 一点 send 就报 socket error
- ABXY / A+X 无效果
- manager / bridge 进程存在，但无 `5556 LISTEN`

关键排查：

```bash
sudo ss -lntup | grep -E ':(8081|60061|5556|5557)\b'
sudo ss -antup | grep -E '192\.168\.88\.210|8081|60061|5556|5557'
```

异常状态：

```text
0.0.0.0:8081 users:(api_server_node / robot_gateway / ...)
<tablet-ip>:8081 <-> <robot-ip> ESTAB
127.0.0.1:5557 LISTEN
[::ffff:127.0.0.1]:60061 LISTEN RoboticsService
no 5556 LISTEN
```

2026-07-16 复核结论：

当时抓包看到 PICO 设备与机器人 `8081` 通信，同时停止 gateway、重启并手动运行 manager 后出现了 `Body data available`，因此调试过程中曾把问题归因为 `8081` 端口冲突。但这个 A/B 实验同时改变了旧进程、runtime 状态和启动顺序，不能单独证明 `8081` owner 就是根因。后续实测已经证明：由平板启动控制框架、切到 `sonic_teleop` 后，PICO 可以连接同一个机器人 IP，并完成 ABXY、A+X 和连续遥操。

因此，`api_server_node/robot_gateway` 占用 `8081` 只作为网络拓扑观察信息，不能再据此判定部署失败，也不应在正常测试前自动停止 gateway。send socket error 应按实际数据链路重新定位。

正确理解：

- SONIC idle 正常，只说明 fallback `idle_reference` 正常；
- 它不证明 PICO live 数据进来了；
- 没有 `5556 LISTEN` 时，ABXY / A+X 必然无效。

当 PICO send 异常时的隔离验证方案：

```bash
sudo pkill -f 'socat.*60061' 2>/dev/null || true

sudo pkill -INT -f 'robot_gateway gateway.launch.py|api_server_node|bxi_example_bms|bxi_bms|hardware_elf3|bxi_example_py_elf3_demo|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' 2>/dev/null || true

sleep 3
sudo ss -lntup | grep -E ':(8081|60061|5556|5557)\b' || true
```

然后手动启动 manager：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /opt/bxi/bxi_rl_controller_ros2_example/setup.bash

PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

$PY -m bxi_example_py_elf3.sonic_pico.pico_manager_legacy \
  --manager \
  --num_frames_to_send 10 \
  --target_fps 50 \
  --port 5556
```

成功后可看到：

```text
device found
Body data available
ZMQ socket bound to port 5556
```

上述停止 gateway、手动启动 manager 的步骤只用于受控 A/B，不是正常部署流程。正常验收应保留实际平板/终端启动架构，并以下列事实判断链路：

- manager 输出 `Body data available`；
- `5556 pose` 持续输出；
- `5557 smpl_ref` 持续输出且 `source_ready=true`；
- SONIC policy 使用 `live_reference`；
- 机器人连续跟随，而不是只变化一次姿势。

### 2.7 缺 pinocchio

现象：

PICO body data 已进来，manager 已绑定 `5556`，但随后崩溃：

```text
ModuleNotFoundError: No module named 'pinocchio'
```

根因：

`ThreePointPose` 初始化 FK calibration 时会加载 `gear_sonic.data.robot_model...`，需要 `pinocchio`。

处理：

安装离线 wheel：

- `pin`
- `eigenpy`
- `hpp_fcl`
- `cmeel*`

### 2.8 NumPy 2.x ABI 与 pin/eigenpy 不兼容

现象：

安装 `pin/eigenpy/hpp_fcl` 后 import 报：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
Segmentation fault
```

根因：

当前 wheel 是按 NumPy 1.x ABI 编译的，不能和 NumPy 2.2.6 混用。

处理：

PICO venv 降级：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

$PY -m pip install --no-index --force-reinstall --no-deps \
  /tmp/sonic_numpy126_wheels/numpy-1.26.4-*.whl \
  /tmp/sonic_numpy126_wheels/scipy-1.15.3-*.whl
```

最终 PICO venv 推荐版本：

```text
numpy==1.26.4
scipy==1.15.3
pin==2.7.0
eigenpy==3.5.1
hpp-fcl==2.4.4
torch CPU
pyzmq
msgpack
xrobotoolkit_sdk
```

验证：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

$PY - <<'PY'
import sys
import numpy
import scipy
import eigenpy
import hppfcl
import pinocchio as pin
import torch
import zmq
import msgpack
import xrobotoolkit_sdk

print("python", sys.executable)
print("numpy", numpy.__version__, numpy.__file__)
print("scipy", scipy.__version__, scipy.__file__)
print("pinocchio", pin.__version__)
print("torch", torch.__version__)
print("all OK")
PY
```

### 2.9 进入 POSE 后机器人只换到另一个静止姿势

现象：

- PICO send 有反应；
- ABXY 后能校准；
- A+X 后机器人从 sonic idle 变到另一个姿势；
- 但机器人没有连续跟随 PICO 动作。

本次实测中，在确认 `5556 pose` 和 `5557 smpl_ref` 都持续输出后，重启机器人即可解决该现象。因此该问题不应优先怀疑 PICO 网络链路或 bridge 基本功能，而应先怀疑多轮调试后残留的 runtime/controller 状态、旧进程、端口占用或控制器内部状态没有完全复位。

推荐处理顺序：

```text
先确认 5556/5557 持续流动
再做一次干净重启/干净停止所有相关进程
重新进入 normal -> sonic_teleop -> ABXY -> A+X
最后再怀疑 bridge / policy 映射问题
```

这个现象不能再按 `8081` 或依赖问题处理，因为此时 PICO 入口、manager、ABXY、A+X 已经至少部分打通。正确排查顺序是逐层确认 live 数据是否持续：

```text
5556 pose 是否持续
5557 smpl_ref 是否持续
smpl_ref 内容是否变化
SONIC policy 是否持续使用 live_reference
```

先看进程和端口：

```bash
ps -ef | grep -E 'pico_manager|pico_pose|sonic_pico_runtime_supervisor|bxi_example_py_elf3_demo|RoboticsService' | grep -v grep

ss -lntup | grep -E ':(5556|5557|60061|8081)\b'
ss -antp | grep -E ':(5556|5557|60061|8081)\b'
```

已验证成功时应出现：

```text
pico_manager_legacy ... --port 5556
pico_pose_to_smpl_ref_bridge ... --pico-port 5556 --out-port 5557
RoboticsServiceProcess
0.0.0.0:5556 LISTEN
127.0.0.1:5557 LISTEN
127.0.0.1:5556 ESTAB
127.0.0.1:5557 ESTAB
```

然后采样 `5556 pose`：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

$PY - <<'PY'
import time, json, zmq, numpy as np

HEADER_SIZE = 1280

def decode(msg, topic):
    prefix = topic.encode()
    if not msg.startswith(prefix):
        return None
    off = len(prefix)
    header = json.loads(msg[off:off+HEADER_SIZE].split(b"\0", 1)[0].decode())
    off += HEADER_SIZE
    out = {}
    for f in header["fields"]:
        dt = {"f32":np.float32,"f64":np.float64,"i32":np.int32,"i64":np.int64,"u8":np.uint8,"bool":np.bool_}[f["dtype"]]
        shape = tuple(f["shape"])
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        out[f["name"]] = np.frombuffer(msg[off:off+n], dtype=dt).reshape(shape).copy()
        off += n
    return out

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt_string(zmq.SUBSCRIBE, "pose")
s.connect("tcp://127.0.0.1:5556")

last_frame = None
count = 0
t0 = time.time()

while time.time() - t0 < 5:
    try:
        msg = s.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        time.sleep(0.005)
        continue
    d = decode(msg, "pose")
    if d is None:
        continue
    count += 1
    frame = int(np.asarray(d.get("frame_index", [-1])).reshape(-1)[-1])
    mode = int(np.asarray(d.get("stream_mode", [-1])).reshape(-1)[-1])
    calib = bool(np.asarray(d.get("calibration_ready", [False])).reshape(-1)[-1])
    if count % 20 == 0:
        print("pose count", count, "frame", frame, "mode", mode, "calib", calib, "frame_delta", None if last_frame is None else frame-last_frame)
    last_frame = frame

print("pose total", count)
PY
```

成功参考：

```text
pose count 20 frame 996 mode 1 calib True frame_delta 1
...
pose total 248
```

再采样 `5557 smpl_ref`：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

$PY - <<'PY'
import time, json, zmq, numpy as np

HEADER_SIZE = 1280

def decode(msg, topic):
    prefix = topic.encode()
    if not msg.startswith(prefix):
        return None
    off = len(prefix)
    header = json.loads(msg[off:off+HEADER_SIZE].split(b"\0", 1)[0].decode())
    off += HEADER_SIZE
    out = {}
    for f in header["fields"]:
        dt = {"f32":np.float32,"f64":np.float64,"i32":np.int32,"i64":np.int64,"u8":np.uint8,"bool":np.bool_}[f["dtype"]]
        shape = tuple(f["shape"])
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        out[f["name"]] = np.frombuffer(msg[off:off+n], dtype=dt).reshape(shape).copy()
        off += n
    return out

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt_string(zmq.SUBSCRIBE, "smpl_ref")
s.connect("tcp://127.0.0.1:5557")

last_frame = None
count = 0
t0 = time.time()

while time.time() - t0 < 5:
    try:
        msg = s.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        time.sleep(0.005)
        continue
    d = decode(msg, "smpl_ref")
    if d is None:
        continue
    count += 1
    frame = int(np.asarray(d.get("frame_index", [-1])).reshape(-1)[-1])
    ready = bool(np.asarray(d.get("source_ready", [False])).reshape(-1)[-1])
    mode = int(np.asarray(d.get("source_stream_mode", [-1])).reshape(-1)[-1])
    if count % 20 == 0:
        print("smpl count", count, "frame", frame, "ready", ready, "mode", mode, "frame_delta", None if last_frame is None else frame-last_frame)
    last_frame = frame

print("smpl total", count)
PY
```

成功参考：

```text
smpl count 20 frame 1612 ready True mode 1 frame_delta 1
...
smpl total 250
```

注意：`smpl_ref` 没有 `root_pos` 字段，不能用 `root_delta` 判断是否静止。SONIC policy 实际使用的是：

```text
term1_local
root_quat
wrist
```

如果怀疑内容静止，应比较这三个字段的变化量，而不是比较不存在的 `root_pos`。

## 3. 新机器人部署 Checklist

### 3.1 拉取并部署 example 包

```bash
cd ~

if [ ! -d "$HOME/bxi_rl_controller_ros2_example/.git" ]; then
  git clone -b feature/sonic-elf3-runtime-gripper-clean --single-branch \
    https://github.com/bxirobotics/bxi_rl_controller_ros2_example.git \
    "$HOME/bxi_rl_controller_ros2_example"
else
  cd "$HOME/bxi_rl_controller_ros2_example"
  git remote add sonic https://github.com/bxirobotics/bxi_rl_controller_ros2_example.git 2>/dev/null || \
    git remote set-url sonic https://github.com/bxirobotics/bxi_rl_controller_ros2_example.git
  git fetch sonic feature/sonic-elf3-runtime-gripper-clean
  git checkout -B feature/sonic-elf3-runtime-gripper-clean sonic/feature/sonic-elf3-runtime-gripper-clean
fi

bash "$HOME/bxi_rl_controller_ros2_example/script/deploy_robot_sonic_example.sh"
```

如果 GitHub 网络不稳定，可使用离线 patch/bundle，不要在机器人上长时间等待公网 pip/git。

### 3.2 安装/确认 PICO venv 和 RoboticsService

必须存在：

```text
/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
/opt/apps/roboticsservice/RoboticsServiceProcess
/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so
```

验证：

```bash
bash ~/bxi_rl_controller_ros2_example/script/check_sonic_pico_python.sh \
  /home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

bash ~/bxi_rl_controller_ros2_example/script/check_robot_sonic_runtime.sh
```

注意：检查脚本已覆盖 `pinocchio/eigenpy/hppfcl`、NumPy 版本、XRT native library path 和 controller Python 依赖检查。

### 3.3 验证控制节点 Python 环境

`bxi_example_py_elf3_demo` 不使用 PICO venv。每台新机器人覆盖 `/opt` 包后，都必须验证 ROS 控制节点环境能 import SONIC policy 需要的依赖：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /opt/bxi/bxi_rl_controller_ros2_example/setup.bash

python3 - <<'PY'
import importlib
for name in ["numpy", "onnxruntime", "zmq", "bxi_example_py_elf3.inference.sonic"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"), getattr(mod, "__file__", "unknown"))
PY
```

如果缺 `zmq`，`bxi_example_py_elf3_demo` 会 exit code 1，控制框架不会接入机器人。

### 3.4 记录 8081 端口状态

部署和排障时记录 `8081` owner 与连接即可：

```bash
sudo ss -lntup | grep ':8081' || true
```

如果看到：

```text
0.0.0.0:8081 users:(api_server_node/robot_gateway/...)
```

不能仅凭这一项判定冲突或停止 gateway。只有当 `Body data available`、`5556`、`5557` 没有建立时，才把端口和抓包结果与当前进程、当前日志一起分析。

### 3.5 验证 PICO body data 是否真正进 manager

启动 manager 后观察：

```text
Body data available
ZMQ socket bound to port 5556
```

同时检查：

```bash
ss -lntup | grep -E ':(5556|5557|60061|8081)\b'
```

关键判断：

- 有 `5556 LISTEN`：PICO pose manager 已起来；
- 没有 `5556 LISTEN`：ABXY/A+X 不可能有效；
- 只有 `5557 LISTEN`：bridge 在等 pose，但 manager 没出数据；
- SONIC idle 姿势正常只代表 fallback `idle_reference` 正常，不代表 live PICO 正常。

### 3.6 ABXY / A+X 成功判断

成功日志：

```text
[PoseLoop] Calibration completed
[Manager] StreamMode switch: OFF -> PLANNER
[Manager] StreamMode switch: PLANNER -> POSE
[PoseLoop] FPS: 50.xx
```

桥接成功后还应看到：

```text
[pico->smpl_ref] sent ...
```

SONIC policy 成功切 live 后应从 `idle_reference` 变为 `live_reference`。

### 3.7 Live 跟随成功判断

只看到机器人从 sonic idle 变到另一个姿势还不够，必须确认 live 数据持续：

- `5556 pose`：`mode=1`、`calib=True`、`frame_delta=1`；
- `5557 smpl_ref`：`ready=True`、`mode=1`、`frame_delta=1`；
- 机器人连续跟随 PICO 动作，而不是只切换到一个静止姿势。

若 `5556/5557` 都持续，而机器人仍不跟随，应继续检查 `term1_local/root_quat/wrist` 的内容变化，以及 SONIC policy 日志中的：

```text
[SONIC] reference status: live_reference
```

如果以上数据都正常，但机器人仍只停在一个新姿势，先重启机器人或彻底清理相关进程再测；本次问题即通过重启解决。

## 4. 建议固化到 GitHub / 部署包的内容

### 4.1 必须补进依赖检查脚本

`script/check_sonic_pico_python.sh` 应检查：

- `numpy`，且建议要求 `<2`
- `scipy`
- `zmq`
- `msgpack`
- `torch`
- `xrobotoolkit_sdk`
- `pinocchio`
- `eigenpy`
- `hppfcl`
- `/opt/apps/roboticsservice/RoboticsServiceProcess`
- `/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so`
- `LD_LIBRARY_PATH` 中可发现 RoboticsService native libs，否则 `xrobotoolkit_sdk` 会因 `libPXREARobotSDK.so` 加载失败

`script/check_robot_sonic_runtime.sh` 还必须检查 ROS 控制节点环境，即 source `/opt` install 后的 `python3` 能导入：

- `numpy`
- `onnxruntime`
- `zmq`
- `bxi_example_py_elf3.inference.sonic`

这是为了避免 `check_sonic_pico_python.sh` 通过，但 `bxi_example_py_elf3_demo` 仍因缺 `zmq` 直接退出。

### 4.2 必须补进离线 wheelhouse

机器人公网慢，不应依赖现场 pip 下载。部署包应包含：

```text
numpy-1.26.4
scipy-1.15.3
pin-2.7.0
eigenpy-3.5.1
hpp_fcl-2.4.4
cmeel*
torch CPU
pyzmq
msgpack
onnxruntime
```

### 4.3 保留 8081 观察信息

部署/运行脚本应在启动 PICO runtime 前检查：

```bash
ss -lntup | grep ':8081'
```

脚本应打印 owner 和连接作为诊断基线，但不能因为 `api_server_node` 或 `robot_gateway` 占用 `8081` 就让部署失败，也不能自动杀进程。最终以 5556/5557 数据流和 `live_reference` 为准。

### 4.4 CPU-only 默认必须保留

机器人无 CUDA，`runtime_supervisor.py` 默认不能传 `--cuda`。只允许通过：

```bash
SONIC_PICO_USE_CUDA=1
```

显式开启。

### 4.5 文档必须说明 fallback reference

部署文档里要明确：

- 机器人切到 sonic idle 不代表 PICO live 数据已通；
- 没有 `5556 LISTEN` 时，ABXY/A+X 必然无效；
- 只有一次姿势变化也不代表 live 跟随成功；
- 如果 5556/5557 都正常但机器人不连续跟随，优先做干净重启/清理旧进程；
- `idle_reference` 是安全 fallback；
- `live_reference` 需要 PICO body data + ABXY + A+X + bridge `source_ready=true`。

## 5. 推荐的一键排雷命令

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python

echo "===== processes ====="
ps -ef | grep -E 'robot_gateway|api_server_node|RoboticsServiceProcess|pico_manager|pico_pose|sonic_pico_runtime_supervisor|bxi_example_py_elf3_demo|hardware_elf3' | grep -v grep || true

echo "===== ports ====="
sudo ss -lntup | grep -E ':(8081|60061|5556|5557)\b' || true
sudo ss -antup | grep -E ':(8081|60061|5556|5557)\b' || true

echo "===== controller python deps ====="
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /opt/bxi/bxi_rl_controller_ros2_example/setup.bash
python3 - <<'PY'
import importlib
mods = ["numpy", "onnxruntime", "zmq", "bxi_example_py_elf3.inference.sonic"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc!r}")
PY

echo "===== pico venv python deps ====="
$PY - <<'PY'
import sys
print("python", sys.executable)

mods = [
    "numpy",
    "scipy",
    "zmq",
    "msgpack",
    "torch",
    "xrobotoolkit_sdk",
    "eigenpy",
    "hppfcl",
    "pinocchio",
]

for name in mods:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc!r}")
PY

echo "===== roboticsservice ====="
ls -lh /opt/apps/roboticsservice/RoboticsServiceProcess || true
```

如需确认 live stream 是否持续，可追加：

```bash
echo "===== live stream ports ====="
ss -antp | grep -E ':(5556|5557|60061|8081)\b' || true
```

并用第 3.7 节的 `5556 pose` / `5557 smpl_ref` 采样脚本检查帧号是否持续增长。

## 6. 最短结论

本次不是 ROS2 domain 问题，也不是 SONIC policy 不支持 live input。真正踩坑点是：

1. `8081` owner 曾与 send socket error 同时出现，但后续平板 + PICO 实测证明它不是充分的失败条件，只保留为诊断信息；
2. PICO venv 缺 `pinocchio/eigenpy/hpp_fcl`；
3. `pin/eigenpy` 与 NumPy 2.x ABI 不兼容，需要 NumPy 1.26；
4. 机器人无 CUDA，manager 默认必须 CPU-only；
5. 新机器人部署必须使用离线 wheelhouse，不能依赖现场公网下载；
6. “机器人动了一下”不等于遥操成功，必须确认 `5556 pose` 和 `5557 smpl_ref` 都持续输出且 SONIC 使用 `live_reference`；
7. 若 live 数据链路正常但机器人仍像卡静止姿势，先重启机器人清理残留状态，本次该问题即通过重启解决；
8. 新机器人 `/opt` 控制节点环境容易缺 `pyzmq`，即使 PICO venv 正常，`bxi_example_py_elf3_demo` 仍会因 `ModuleNotFoundError: No module named 'zmq'` 退出，必须单独检查 controller Python deps。

## 7. 后续固化结果（2026-07-16）

上述实机问题已进一步固化为可重复的离线部署流程：

- `script/audit_robot_sonic_host.sh`：部署前只读盘点系统、磁盘、基础 ROS、进程和端口；
- `script/prepare_robot_sonic_bundle.sh`：从干净 Git commit 生成版本唯一、带 SHA256 的精简离线包；
- `script/deploy_robot_sonic_bundle.sh`：校验 payload 后构建、备份、覆盖 `/opt` 并执行完整检查；
- `script/install_robot_sonic_runtime_offline.sh`：分别安装 PICO venv 和 `/opt` 控制节点依赖；
- `docs/SONIC_ELF3_OFFLINE_DEPLOY.md`：记录上传、部署、成功标准与回滚流程。

离线包不再包含 NumPy 2.x，也不包含 headless 实机路径不需要的 VTK/PyVista。部署脚本会
拒绝同一依赖存在多个候选 wheel，从源头避免 pip 选错版本；源码 commit 也会在 build 前
核对，避免机器人上的代码、报告和 GitHub 分支互不一致。

## 8. 新机器人部署复盘（2026-07-17）

本次在新机器人 `<robot-id>` 上部署离线包：

```text
elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz
sha256: b2d4aff4a5b4a65070d72b44ec35f9f2c170cbe9fbc5098bba9de0f5dc2d7331
commit: <deployment-commit>
```

最终结果：

- ROS 包已覆盖到 `/opt/bxi/bxi_rl_controller_ros2_example`；
- 旧版本已自动备份到 `/opt/bxi/deploy_backups/bxi_rl_controller_ros2_example.before_sonic_20260717_095536.tgz`；
- PICO venv 和主控 `/opt` Python 环境均安装 `pyzmq==27.1.0`；
- `check_robot_sonic_runtime.sh` 最终通过；
- 平板启动控制框架后，PICO 可连接机器人 IP，`Body data available`、`5556 pose`、`5557 smpl_ref` 均打通。

### 8.1 上传命令必须单行执行

本次曾因多行命令被拆开执行，导致 `rsync` 误同步了用户 home 下大量目录，并出现：

```text
elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz: 未找到命令
权限不够
rsync error: some files/attrs were not transferred (code 23)
```

这不是部署包损坏，而是 shell 把续行内容当成独立命令。新机器人部署时推荐只给单行命令：

```bash
rsync -avP --partial --append-verify -e 'ssh -p <ssh-port>' "$HOME/elf3_sonic_artifacts/elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz" "$HOME/elf3_sonic_artifacts/elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256" <robot-user>@<bastion-host>:/tmp/
```

上传后必须先校验 SHA256，再解包部署。

### 8.2 部署前要停旧硬件主控和旧 controller

首次运行 `deploy_robot_sonic_bundle.sh` 被阻塞：

```text
[bundle-deploy] ERROR: robot control/PICO processes are still active:
2714 /opt/bxi/bxi_ros2_pkg/lib/hardware_elf3/hardware_elf3 ...
2716 /usr/bin/python3 /opt/bxi/bxi_rl_controller_ros2_example/lib/bxi_example_py_elf3/bxi_example_py_elf3_demo ...
```

普通 `kill` 报：

```text
Operation not permitted
```

原因是旧进程由 root/systemd/平板框架拉起，不能用普通 `bxi` 权限停止。正确处理：

```bash
sudo systemctl stop ros_elf_launch.service 2>/dev/null || true
sudo pkill -INT -f 'hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess'
sleep 3
pgrep -af 'hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' || true
```

如仍残留，再用 `sudo pkill -TERM ...`。不要直接 `kill -9`，先看父进程是否会自动拉起。
正常远程部署不要停止 `bxi_rc_ros2.service`、`robot_gateway` 或 `api_server_node`；远程
维护隧道可能属于同一 gateway 进程树，停止后会直接断开 SSH。应先在平板停止当前控制
会话，再只停止上述硬件主控、controller、PICO manager/bridge 和 RoboticsService。

### 8.3 `dpkg` 锁可能来自 unattended-upgrade

ROS 包部署完成后，安装离线 RoboticsService 时遇到：

```text
dpkg: error: dpkg frontend lock was locked by another process with pid 4602
```

定位后发现：

```text
/usr/bin/python3 /usr/bin/unattended-upgrade --download-only
/usr/lib/apt/apt.systemd.daily update
```

处理原则：

- 不删除 `/var/lib/dpkg/lock*`；
- 先等自动升级结束；
- 若长时间无进展，再停止 apt daily 相关服务并温和终止进程；
- 最后执行 `sudo dpkg --configure -a`。

可用命令：

```bash
sudo systemctl stop apt-daily.service apt-daily-upgrade.service apt-daily.timer apt-daily-upgrade.timer packagekit.service 2>/dev/null || true
sudo pkill -TERM -f 'unattended-upgrade|apt.systemd.daily|/usr/lib/apt/methods/http'
sleep 10
pgrep -af 'apt|apt-get|dpkg|unattended|packagekit' || true
sudo dpkg --configure -a
```

### 8.4 `/tmp` 解包目录不可靠

部署和调试过程中，`/tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64` 曾消失，导致：

```text
cd: /tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64: No such file or directory
bash: source/script/...: No such file or directory
```

因此：

- 部署阶段可从 `/tmp/*.tgz` 重新解包；
- 部署完成后的长期调试不要依赖 `/tmp/source/script`；
- 已安装内容应优先从 `/opt/bxi/bxi_rl_controller_ros2_example` 启动和检查。

### 8.5 手动启动 PICO manager 要带 XRT native 路径

部署健康检查已证明 `xrobotoolkit_sdk` 存在，但手动启动 manager 时若只设置 ROS install
的 `PYTHONPATH`，仍会报：

```text
ImportError: XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager.
```

原因是手动命令缺少 RoboticsService native 库路径。正确的手动 manager 启动命令：

```bash
cd /opt/bxi/bxi_rl_controller_ros2_example

export PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
export PYTHONPATH=/opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/apps/roboticsservice/SDK/x64:/opt/apps/roboticsservice:/opt/apps/roboticsservice/lib:${LD_LIBRARY_PATH:-}

$PY -m bxi_example_py_elf3.sonic_pico.pico_manager_legacy \
  --manager --num_frames_to_send 10 --target_fps 50 --port 5556
```

成功时日志：

```text
[Manager] Body data available after 141.43s
[Manager] ZMQ socket bound to port 5556
[PoseLoop] Calibration completed (zero-pose reference)
[Manager] StreamMode switch: OFF -> PLANNER
[Manager] StreamMode switch: PLANNER -> POSE
```

### 8.6 手动启动 bridge 验证 5557

若只想验证 PICO 输入链路，可以在另一个终端手动启动 bridge：

```bash
cd /opt/bxi/bxi_rl_controller_ros2_example

export PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
export PYTHONPATH=/opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/apps/roboticsservice/SDK/x64:/opt/apps/roboticsservice:/opt/apps/roboticsservice/lib:${LD_LIBRARY_PATH:-}

$PY -m bxi_example_py_elf3.sonic_pico.pico_pose_to_smpl_ref_bridge \
  --pico-host 127.0.0.1 \
  --pico-port 5556 \
  --pico-topic pose \
  --out-host 127.0.0.1 \
  --out-port 5557 \
  --out-topic smpl_ref \
  --wrist-source pico_g1_legacy \
  --log-every 1 \
  --disable-ros-pico-topics
```

成功时应持续输出：

```text
[pico->smpl_ref] sent 302 received=293 skipped=0 frame=4202 input_age_ms=3 local_current=7 T=10 window_start=4195 catchups=1 term1=(10, 72) root=(10, 4) wrist=(10, 6)
```

这说明链路至少已打通到：

```text
PICO -> manager -> 5556 pose -> bridge -> 5557 smpl_ref
```

若此时机器人仍不跟随，下一层才检查 `bxi_example_py_elf3_demo` 是否切到
`[SONIC] reference status: live_reference`，以及硬件执行层是否存在大量
`motor_timeout` / `imu data error`。

### 8.7 不要并行启动两套 PICO runtime

平板/主控进入 `sonic_teleop` 后，`sonic_pico_runtime_supervisor` 会自动拉起 manager 和
bridge。手动执行 `pico_manager_legacy` / `pico_pose_to_smpl_ref_bridge` 只适合作为诊断，
启动前必须先确认没有已有进程和端口：

```bash
pgrep -af 'sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' || true
sudo ss -lntup | grep -E ':(5556|5557|60061)\b' || true
```

若已有 `5556 LISTEN` 或 `5557 LISTEN`，不要再启动第二套，否则会混淆端口、日志和状态机判断。

进入 POSE/live 后优先使用部署包内的一键流检查代替临时复制长 Python 片段：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
$PY source/script/check_sonic_live_stream.py --duration 5
```

脚本同时订阅 `5556 pose` 和 `5557 smpl_ref`，检查帧号持续前进、POSE/calibration、
`source_ready`、字段 shape、有限值和实测速率，并打印 `term1_local/root_quat/wrist`
相对首帧的变化量。只有两路均显示 `[PASS]` 才算 live 输入验收通过。

### 8.8 本次新增经验结论

1. 部署包正确不代表部署能立刻覆盖 `/opt`，旧 root 控制进程必须先停干净。
2. `dpkg` 锁优先查 `unattended-upgrade`，绝不手动删除 lock 文件。
3. `/tmp` 只适合临时上传和部署，不适合作为部署后调试入口。
4. 手动跑 PICO manager 时，必须同时带 PICO venv、ROS install `PYTHONPATH` 和 RoboticsService `LD_LIBRARY_PATH`。
5. PICO 显示 `working` 只证明设备侧状态，不等价于机器人会跟随；最终仍以 `5556 pose`、`5557 smpl_ref` 和 `live_reference` 为准。
6. `8081` gateway 仍然只作为诊断信息，不作为 PICO 链路失败的单项证据。

## 9. 新机器人部署与 App 联调复盘（2026-07-27 至 2026-07-28）

本轮使用离线包：

```text
elf3_sonic_deploy_4f87e2c1582f_ubuntu22_amd64.tgz
commit=4f87e2c1582f4251561c13d19ef6aa5572343945
branch=feature/sonic-elf3-runtime-clean
```

上面是当时产物中保留的历史字段；当前部署统一使用
`feature/sonic-elf3-runtime-gripper-clean`，不再依赖该旧分支。

部署目标在公开报告中统一记为 `<robot-id>`，不记录机器人实际编号、IP、账号或现场网络信息。

### 9.1 部署完成时间点必须和 App 验收分开

本轮有三个不同的完成边界：

1. **静态安装完成**：2026-07-27 约 19:51。离线部署器完成源码构建、旧 `/opt` 备份、
   新 `/opt` 覆盖、RoboticsService/PICO 依赖安装和完整健康检查。日志明确结束于：

   ```text
   [OK] robot SONIC runtime checks passed
   [bundle-deploy] deployment complete: 4f87e2c1582f4251561c13d19ef6aa5572343945
   ```

   同时生成了旧安装备份：

   ```text
   /opt/bxi/deploy_backups/bxi_rl_controller_ros2_example.before_sonic_20260727_195113.tgz
   ```

2. **机器人端功能部署完成**：2026-07-27 约 20:21。终端能够从新 `/opt` 启动
   `hardware_elf3`、`bxi_example_py_elf3_demo` 和 supervisor，状态机保持约 50 Hz，键盘能够
   正确进入 `normal`、SONIC、挥手、鼓掌等状态；随后 SONIC/PICO 实测也成功。到这一时点，
   机器人端 SONIC 部署已经完成。
3. **App 兼容验收完成**：2026-07-28。后续出现的“App 点击 SONIC 却进入其他动作”是
   App 控制命令处理问题，经 App 端更新后解决。它不应倒推为离线包、SONIC policy 或
   机器人 `/opt` 部署失败。

以后报告进度时应分别使用“安装完成”“终端验收完成”“App 联调完成”，不能把三者混为
一个完成条件。对于机器人端发布，终端硬件链、SONIC 状态和 PICO live 链全部通过，即可
判定部署成功；App 是产品集成的下一层验收。

### 9.2 App 限幅/透传问题及机器人端排除证据

旧 App 的现象是：界面已经读取并显示“SONIC 遥操”，但点击后机器人进入挥手/鼓掌等
其他动作。App 同事最终确认与 App 端的“限幅/透传”处理有关，并通过更新 App 解决。

机器人端排查证明三处 SONIC 定义一致：

```text
键盘输入链：key 6 -> keyboard.sonic_teleop_event -> btn_10=7
状态机入口：sonic_teleop_event -> slot=btn_10, value=7
App manifest：sonic_teleop -> sonic_teleop_event -> btn_10=7
```

`btn_10` 是整数动作选择槽，不是单一布尔按钮：

```text
btn_10=2  forward_flip
btn_10=5  applause
btn_10=6  hello
btn_10=7  sonic_teleop
```

因此“多个动作都使用 `btn_10`”本身不是冲突；接收方必须透传并按完整的
`(slot, value)` 识别。联调时曾在 gateway 日志中观察到旧 App 对 SONIC 点击发出的值不是
预期的 `7`，而机器人端对收到的值进行了正确执行。不能为了迁就旧 App 把机器人端 SONIC
改成 `5` 或 `6`，否则会与鼓掌/挥手形成真实冲突。

今后的 App 验收必须同时检查：

```text
manifest: id=sonic_teleop, event=sonic_teleop_event, slot=btn_10, value=7
gateway:  event(b10)=7
state:    current.name=sonic_teleop
```

只看到 App 上出现 SONIC 按钮不等于命令透传正确；只看到机器人执行某个动作也不能用于
反推 App 实际发送的值。

当前 `xbox_default.yaml` 仍有一个手柄 `btn_10=4` 输出，但 ELF3 状态机和 App manifest
都没有对应事件。这是遗留的无效映射，后续应单独清理或明确用途，不影响本轮 SONIC
`btn_10=7` 的部署结论。

### 9.3 `robot.startup.cmd_failed` 是另一类生命周期问题

联调期间还出现过：App 停止控制后，硬件和主 controller 已退出，但 `ros2 launch` 与
`sonic_pico_runtime_supervisor` 残留并继续持有：

```text
/tmp/bxi_example_hw.lock
```

下一次启动因此报告：

```text
bxi_example_hw is already running
robot.startup.cmd_failed
```

这与 App 动作值的限幅/透传问题互相独立。现场通过终止实际持锁的 `ros2 launch` 进程恢复。
本地工作区已经准备了“关键硬件/controller 退出时关闭整个 launch”的生命周期修复及测试，
但该修改不属于已部署的 `4f87e2c` 离线包；在提交、重新打包并做连续两次 App
“启动 -> 停止 -> 再启动”验收之前，不能宣称这一项已在发布包中永久修复。

不要杀共享 gateway 的进程组。再次遇到时先检查：

```bash
pgrep -af 'ros2 launch bxi_example_py_elf3|hardware_elf3|bxi_example_py_elf3_demo|sonic_pico_runtime_supervisor' || true
sudo fuser -v /tmp/bxi_example_hw.lock 2>/dev/null || true
```

### 9.4 本轮实际采用的最短部署流程

工作站准备阶段：

```bash
cd "$HOME/elf3_sonic_artifacts"
sha256sum -c elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256
scp elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256 <robot>:/tmp/
```

机器人安装阶段：

```bash
cd /tmp
sha256sum -c elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz.sha256
tar -xzf elf3_sonic_deploy_<commit>_ubuntu22_amd64.tgz
cd elf3_sonic_deploy_<commit>_ubuntu22_amd64

# 先在平板停止控制会话；只停止需要被覆盖的旧控制框架，不停止共享 gateway。
sudo systemctl stop ros_elf_launch.service 2>/dev/null || true
bash source/script/audit_robot_sonic_host.sh

sudo -v
sudo dpkg --configure -a
set -o pipefail
bash source/script/deploy_robot_sonic_bundle.sh "$PWD" 2>&1 | tee /tmp/elf3_sonic_deploy_<commit>.log
```

只有在审计明确显示 apt/dpkg 锁被占用时，才按第 8.3 节处理自动升级；不应把停止 apt 服务
作为每台机器人无条件执行的默认步骤。

终端验收阶段使用部署包内脚本：

```bash
BUNDLE=/tmp/elf3_sonic_deploy_<commit>_ubuntu22_amd64

# T1：硬件、状态机、自动 PICO supervisor
bash "$BUNDLE/source/script/run_robot_sonic_hw.sh"

# T2：另一个终端运行键盘控制
bash "$BUNDLE/source/script/run_robot_sonic_controller.sh"
```

安全地按 `! -> 1 -> 6` 验证 `pd_brake -> normal -> sonic_teleop`，随后连接 PICO，完成
ABXY、A+X，并执行：

```bash
PY=/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
$PY "$BUNDLE/source/script/check_sonic_live_stream.py" --duration 5
```

`pose`、`smpl_ref` 两项都为 `[PASS]` 且 policy 使用 `live_reference`，才算 SONIC live
动态验收通过。App 应在终端验收完成、进程清理干净后再单独验收。

### 9.5 当前是不是跨机器人健壮的一键部署包

准确结论是：**已经有“固定 ELF3 基线上的解包后一键安装器”，还没有覆盖任意机器人环境的
端到端一键部署器。**

当前包已经具备的重要能力：

- 固定 Git commit、分支和平台；
- TGZ 外层 SHA256 与包内逐文件 SHA256；
- 固定且唯一的 CPU-only wheel 版本；
- 同时安装 PICO venv 和 `/opt` controller Python 依赖；
- 部署前拒绝覆盖运行中的控制/PICO 进程；
- 离线 colcon build；
- 覆盖前自动备份旧 `/opt`；
- 安装后检查 ROS prefix、可执行文件、Python import、XRT 和端口；
- 独立的终端硬件、键盘和 live-stream 验收脚本。

在两台机器人均通过的 `4f87e2c` 流程基础上，本地下一版还增加了：

- `install_robot_sonic_bundle.sh` 单一安全入口；
- 自动读取 effective systemd `ExecStart` 和 `Environment`；
- 只停止现场控制 service，同时监护共享 gateway 不得消失；
- 阻塞残留进程、硬件锁、apt/dpkg 锁，不用无条件 `pkill`；
- `configure_robot_sonic_service.sh` drop-in，只改 `/opt` 启动路径并逐字保留现场
  Environment；
- 安装失败后保持 service 停止并报告最近备份；
- hardware/controller 退出即关闭整个 launch，并给 PICO supervisor 留足有界清理时间；
- bundle 源码改为 allowlist，不再携带顶层 CAD `resources/` 和无关 ROS 包。

这些是下一 commit 的候选实现，已通过本地脚本语法和单元测试；不能写成已经包含在
`4f87e2c` 或已经完成真机 App 启停回归。

因此在满足下列基线时，它已经可重复使用：Ubuntu 22.04、x86_64、Python 3.10、ROS 2
Humble、已有 `/opt/bxi/bxi_ros2_pkg`、用户/路径布局与当前 ELF3 镜像一致。

仍然需要人工处理或尚未覆盖的部分：

1. 上传、解包和选择部署包路径；
2. 非默认控制 service 仍需显式传 `--service`；
3. apt/dpkg 锁只安全阻塞，仍需操作员等待或处理；
4. 当前固定的 `/home/bxi/...`、`/opt/bxi/...` 路径与用户名；
5. `/opt` 仍是备份后 overlay copy，并非原子目录切换；PICO venv/deb 也没有自动回滚；
6. App 最低兼容版本及命令透传仍需产品侧验收；
7. 生命周期修复和新安装入口尚未进入正式 bundle，也未做连续两次 App 启停真机回归；
8. PICO 配对、ABXY、A+X 和真实人体动作不能由安装器无条件自动完成。

所以本地下一版 `install_robot_sonic_bundle.sh` 的“一键”边界是：**文件已上传并解包后，
用一个安全入口完成现场确认、停止控制 service、静态安装、service 启动路径配置和检查；
最后仍由操作员执行硬件、PICO 和 App 验收。**

### 9.6 下一版一键入口的实现状态

本地已经实现候选入口：

```bash
bash source/script/install_robot_sonic_bundle.sh "$PWD"
```

实现保持安全分阶段，而不是无条件杀进程或自动驱动机器人：

1. `preflight`：校验包、平台、磁盘、基础 ROS、用户路径、service、端口和包锁；
2. `stop`：只停止识别出的控制 service，保留 gateway/SSH，并打印将要停止的对象；
3. `install`：构建到 staging、备份旧版本并 overlay `/opt`；
4. `failure`：失败时保持 service 停止并给出备份，不声称已自动完整回滚；
5. `configure`：用 drop-in 写入 `/opt` setup，保留而不改现场 ROS domain；
6. `static-check`：运行当前完整 health check；
7. `terminal-acceptance`：由操作员确认安全后再启动硬件，不在安装阶段自动上电运动；
8. `app-check`：当前仍由操作员人工核对 App manifest、实际下发值和最低兼容版本；安装器只
   静态检查机器人端状态机中 SONIC 为 `btn_10=7`，后续可另行增加只读 App 联调工具；
9. `result`：输出 commit、备份、安装路径、检查结果和下一条唯一验收命令。

在生命周期修复、新入口完成真机回归，且 staging/回滚边界进一步收紧前，它应称为
“标准 ELF3 基线上的一入口安全静态安装”，不应称为“任意 ELF3 全自动一键部署”。

## 10. 第二台标准 ELF3 快速部署复盘（2026-07-28）

第二台机器人继续使用同一个已验证离线包和 commit：

```text
elf3_sonic_deploy_4f87e2c1582f_ubuntu22_amd64.tgz
commit=4f87e2c1582f4251561c13d19ef6aa5572343945
```

外层 SHA256、主机审计、包内逐文件校验、离线构建、运行时安装和完整健康检查均通过；
终端硬件与 SONIC 验收随后通过。部署目标在公开文档中记为 `<robot-id-B>`，不记录实际
机器人编号、IP、ROS Domain 或现场账号。

这次部署明显更快，说明前一台机器上的依赖、包锁、旧进程、PICO/XRT 和 App 问题已经被
拆成了可复用的检查步骤。没有重新修改 policy、reference、状态机或 PICO 数据链代码。

### 10.1 本轮审计发现和处理

部署前环境满足 Ubuntu 22.04、x86_64、Python 3.10、ROS 2 Humble 和磁盘要求，但有四个
现场差异需要处理：

1. 旧 `remote_controller`、`hardware_elf3` 和 controller 正在运行，且旧 launch 持有
   `/tmp/bxi_example_hw.lock`；先通过机器人自身的 `ros_elf_launch.service` 温和停止完整
   control group，再确认控制进程和锁都为空。
2. `bxi_rc_ros2.service`、gateway、API server 和 `8081` 保持运行；它们不属于 `/opt`
   example 覆盖对象，也不能为了清理控制进程而误停。
3. 旧 `ros_elf_launch.service` 的 `ExecStart` 指向
   `/home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash`。如果只覆盖 `/opt` 而
   不修正该服务，终端测试会使用新包，但 App 重启后仍会使用旧包。
4. 该机器使用自己的非默认 ROS Domain。部署时保留原 service 的 Environment，只用
   systemd drop-in 覆盖 `ExecStart` 到
   `/opt/bxi/bxi_rl_controller_ros2_example/setup.bash`，不能照搬其他机器的 Domain。

PICO venv、RoboticsService 和 XRT native 库在部署前不存在，但均由离线 payload 正常安装；
这正是完整离线包应覆盖的正常新机情况，不再需要现场联网补依赖。

### 10.2 新增的部署成功判据

第二台机器人验证后，成功判据补充为：

```text
1. deploy_robot_sonic_bundle.sh -> deployment complete
2. ros2 pkg prefix -> /opt/bxi/bxi_rl_controller_ros2_example
3. ros_elf_launch.service effective ExecStart -> source /opt/bxi/.../setup.bash
4. ros_elf_launch.service 保留该机器人原 ROS_DOMAIN_ID/RMW/CycloneDDS 环境
5. 终端 T1/T2 在同一 Domain 下启动并通过 SONIC 验收
6. 最后才恢复 service/App 验收
```

前两项只能证明安装内容正确；第三、四项证明产品启动入口确实会使用新包；第五项证明机器人
端功能可运行；第六项属于 App 集成验收。

### 10.3 本部署流程涉及的仓库文件

生成和安装离线包：

- `script/prepare_robot_sonic_bundle.sh`：从干净 commit 生成源码快照、固定 wheel/XRT payload、
  包内 manifest、TGZ 和外层 SHA256；
- `script/audit_robot_sonic_host.sh`：只读检查平台、磁盘、基础 ROS、旧安装、控制进程和端口；
- `script/install_robot_sonic_bundle.sh`：下一版唯一安全入口，校验目标、停止控制 service、
  阻塞锁/残留进程、串联安装和 service 集成，但不启动硬件；
- `script/configure_robot_sonic_service.sh`：检查或安装只覆盖 `ExecStart` 的 systemd drop-in，
  并验证现场 Environment 没有变化；
- `script/deploy_robot_sonic_bundle.sh`：离线包总入口，串联校验、ROS 包部署、runtime 安装和
  健康检查；
- `script/deploy_robot_sonic_example.sh`：构建两个 ROS 包、备份旧 `/opt` 并覆盖安装；
- `script/install_robot_sonic_runtime_offline.sh`：安装 RoboticsService、PICO venv 和 controller
  Python 依赖；
- `script/sonic_runtime_requirements.txt`：记录固定运行时依赖集合。

静态和动态验收：

- `script/check_robot_sonic_runtime.sh`：检查 ROS prefix、可执行文件、controller/PICO imports、
  XRT、进程和端口；
- `script/check_sonic_pico_python.sh`：检查 PICO Python ABI 和依赖版本；
- `script/run_robot_sonic_hw.sh`：终端 T1，启动硬件、状态机和自动 supervisor；
- `script/run_robot_sonic_controller.sh`：终端 T2，启动键盘控制；
- `script/check_sonic_live_stream.py`：同时验收 `5556 pose` 和 `5557 smpl_ref`；
- `docs/SONIC_ELF3_TERMINAL_ACCEPTANCE.md`：记录安全终端验收顺序。

真正进入机器人运行链的配置和代码：

- `src/remote_controller/config/xbox_default.yaml`：键盘/手柄到 `MotionCommands` 的映射以及
  App 启停命令；
- `src/bxi_example_py_elf3/config/elf3_state_machine.yaml`：manifest、`btn_10` 多值事件和状态转换；
- `src/bxi_example_py_elf3/launch/example_demo_hw.launch.py`：硬件、controller 和 PICO supervisor
  的统一 launch；
- `src/bxi_example_py_elf3/bxi_example_py_elf3/sonic_pico/runtime_supervisor.py`：进入/退出
  `sonic_teleop` 时管理 PICO manager/bridge；
- `src/bxi_example_py_elf3/bxi_example_py_elf3/inference/sonic.py`：SONIC policy 与 reference 输入；
- `script/ros_elf_launch.service`：仓库服务模板，但现场应优先检查机器人已有 service，并通过
  drop-in 保留其 Domain/RMW 环境。

部署文档：

- `docs/SONIC_ELF3_OFFLINE_DEPLOY.md`；
- `docs/SONIC_ELF3_TERMINAL_ACCEPTANCE.md`；
- `docs/SONIC_ELF3_DEPLOY_DEBUG_REPORT_20260715.md`。

不进入 GitHub 的文件：生成后的 `*.tgz`、`*.tgz.sha256`、离线 wheelhouse、RoboticsService
deb、XRT 二进制、机器人日志、现场 systemd drop-in、IP、机器人编号和 ROS Domain。

### 10.4 GitHub 更新结论

有必要更新 GitHub，但应分两组提交：

1. **部署文档更新**：提交本节、App 限幅/透传复盘、离线部署手册中的 systemd 集成步骤。
   这些是第二台机器人已经验证的事实。
2. **生命周期代码更新**：本地已有关键 hardware/controller 退出时关闭整个 launch、释放
   单实例锁的修改及测试。它解决 `robot.startup.cmd_failed` 的残留 launch 问题，但尚未进入
   `4f87e2c` 包；应先做连续两次“启动 -> 停止 -> 再启动”回归，再单独 commit。

上述 effective `ExecStart` / Domain 检查及保留 Environment 的 drop-in 脚本已经在本地补齐，
并有部署脚本测试覆盖。下一步是把文档、生命周期修复、部署自动化和 bundle allowlist 分开
commit；随后从新 commit 重建 artifact，在安全条件下做连续两次 App
“启动 -> 停止 -> 再启动”及新入口真机回归。

不需要因为第二台机器人部署成功而重新提交 ONNX、reference、PICO vendor、wheel 或 XRT；
同一个 `4f87e2c` 包已经证明这些内容在两台同基线 ELF3 上可用。文档/生命周期修改提交后，
未来新部署应从新 commit 重建新 bundle，不应覆盖或伪装旧 `4f87e2c` artifact。

### 10.5 下一版候选的提交前加固与验证

部署复盘文档本身不需要重新部署到已经验收完成的机器人；只有运行代码、launch、配置或
安装脚本变化后，才需要从新 commit 生成新 bundle。当前本地候选在提交前进一步完成：

- `--service` 先验证 effective `ExecStart` 确实启动
  `remote_controller.launch.py`，并显式拒绝 SSH、gateway 等关键 unit；
- service 存在时，`--terminal-only` 仍可为部署安全而停止旧控制，但跳过 drop-in 和 App
  integration 强制检查；
- 安装窗口全程持有与 `example_demo_hw.launch.py` 相同的硬件 flock，避免 App/终端在
  colcon、copy、pip/dpkg 中途竞态启动；
- hardware launch 使用 `O_NOFOLLOW` 打开 `/tmp/bxi_example_hw.lock`，只接受 regular file，
  并通过 fd 修改权限，避免跟随伪造 symlink；
- installer 捕获 `INT`、`TERM`、`HUP`，中断后保持控制 service 停止并打印可用备份；
- systemd drop-in 修改前留下持久 effective-state/旧 drop-in 备份，失败时尝试恢复并在恢复
  失败时保留人工恢复材料；
- audit 对 Ubuntu 22.04、ELF3 hardware package、`fuser`/`flock` 等安全依赖做阻断检查；
- supervisor launch 的 SIGINT grace 延长到 10 s，覆盖其最多约 7.5 s 的 manager/bridge
  有界清理路径；
- 固定 `/tmp` 诊断输出改为 `mktemp`，部署路径在 `rm -rf` 前规范化并限制到专用前缀。

本地验证结果：

```text
bash -n（8 个相关脚本）                    PASS
git diff --check                           PASS
真实 LaunchDescription 构造                PASS
硬件锁 symlink 不改写目标文件              PASS
fake-systemd unit 拒绝/drop-in/回滚测试     PASS
src/bxi_example_py_elf3/test                54 passed
```

bundle allowlist 仍保留两个实机 ROS 包、全部动作/SONIC 数据、部署脚本、三份操作文档及
许可证/归因材料，只排除不参与实机构建的顶层 CAD `resources/`、无关 ROS 包和开发工具。
实测源码压缩快照约从 123.8 MiB 降至 87.3 MiB，预计完整 529.1 MiB 离线包降至约
492.7 MiB（约 6.9%）；runtime wheel/XRT 约 415 MiB，未经独立依赖试验不能继续删除。

这些结果足以进入代码评审，但不替代真机验收。生命周期修复/新安装入口提交后，应生成新
commit 对应的新 artifact，并至少在一台安全悬挂的 ELF3 上执行两轮
“启动 -> 停止 -> 确认 launch/supervisor/lock 全清 -> 再启动”，再升级为推荐部署入口。
