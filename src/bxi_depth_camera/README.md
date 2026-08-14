# bxi_depth_camera

独立的 C++ ROS 2 深度相机发布包。它在运行期间发现、打开和重连所有受支持的
Intel RealSense 与 Orbbec Gemini 335，不包含任何策略专属裁剪。节点使用
`rclcpp`、librealsense2 C++ API 和 OrbbecSDK_v2 C++ API，不依赖 Python、NumPy
或厂商 Python wheel。

## 获取源码与构建

两套 SDK 均使用包内按架构存放的预编译 C++ bundle：OrbbecSDK_v2 固定为 `2.9.3`，
librealsense 固定为 `2.57.7`。普通构建不会下载或编译 SDK，也不会查找系统安装的
Orbbec/librealsense 开发包。目前同时内置 `linux-x86_64` 和 `linux-aarch64`，CMake
根据目标机的 `CMAKE_SYSTEM_PROCESSOR` 自动选择，禁止跨架构误用。

Ubuntu 22.04 / ROS 2 Humble 的基础构建依赖：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake libopencv-dev libusb-1.0-0 libudev1 \
  ros-humble-ament-cmake \
  ros-humble-rclcpp \
  ros-humble-rcl-interfaces \
  ros-humble-sensor-msgs
```

不需要安装 `ros-humble-librealsense2`、`librealsense2-dev` 或系统版 Orbbec SDK。
`libusb`、`libudev`、glibc 和 libstdc++ 属于操作系统基础库，不随包复制。

首次部署时，用包内脚本一键安装 Orbbec 和 RealSense 的 udev 规则：

```bash
source install/setup.bash
ros2 run bxi_depth_camera install-udev-rules
```

也可以在构建前直接从源码树运行；脚本路径不依赖仓库所在位置：

```bash
src/bxi_depth_camera/tools/install_udev_rules.sh
```

脚本会按当前 CPU 架构寻找随包内置的规则，自动使用 `sudo` 安装、重新加载 udev，并
重新触发已连接的 USB、IIO、HID 和 Video4Linux 设备。执行成功后仍建议重新插拔两种
相机。以下是无需脚本时的手动安装方式：

```bash
case "$(uname -m)" in
  x86_64 | amd64) PLATFORM=linux-x86_64 ;;
  aarch64 | arm64) PLATFORM=linux-aarch64 ;;
  *) echo "unsupported architecture: $(uname -m)"; exit 1 ;;
esac
VENDOR=src/bxi_depth_camera/vendor/cpp/${PLATFORM}
sudo install -m 0644 \
  "${VENDOR}/orbbec_sdk_v2/shared/99-obsensor-libusb.rules" \
  /etc/udev/rules.d/99-obsensor-libusb.rules
sudo install -m 0644 \
  "${VENDOR}/realsense2/shared/60-librealsense2-udev-rules.rules" \
  /etc/udev/rules.d/60-librealsense2-udev-rules.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

构建并加载环境：

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-select bxi_depth_camera
source install/setup.bash
```

构建会把 `libOrbbecSDK.so`、`librealsense2.so`、Orbbec 扩展库和 SDK 配置资源一起
安装到工作空间。两个节点都带相对 RPATH；部署机复制完整 ROS 2 安装空间并 source
即可，不需要额外的 SDK、`LD_LIBRARY_PATH` 或 Python 路径。

```bash
ros2 launch bxi_depth_camera cameras.launch.py
```

## 在 ARM64 上生成 SDK bundle

这一步只需要维护者在 ARM64 Ubuntu 22.04 机器上执行一次。它会从两个官方仓库拉取
固定版本、在临时目录中原生编译，并生成
`vendor/cpp/linux-aarch64`；后续正常构建仍只链接产物。

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config \
  libusb-1.0-0-dev libudev-dev libssl-dev

cd ~/bxi_rl_controller_ros2_example
src/bxi_depth_camera/tools/build_sdk_bundle.sh
```

成功时最后一行应为：

```text
Created .../src/bxi_depth_camera/vendor/cpp/linux-aarch64
```

把整个 `src/bxi_depth_camera/vendor/cpp/linux-aarch64/` 目录交付回来即可。不要只复制
两个主 `.so`，Orbbec 的 `extensions/`、XML、头文件、SONAME 符号链接、udev 规则、
许可证和 `MANIFEST.md` 都是 bundle 的一部分。脚本若发现目标目录已经存在会直接
停止，避免覆盖已有产物。

## 相机参数与序列号探测

安装并 source 工作空间后，使用 `cameras-inspect` 快速读取当前相机的硬件序列号和
RGB/Depth 声明支持的全部 profile。它只枚举，不启动任何流；报告包含每项 profile 的
分辨率、帧率、像素格式、默认项、FOV、内参、畸变模型与系数：

```bash
ros2 run bxi_depth_camera cameras-inspect
```

快速枚举会立即排序并输出完整 profile 表，每项使用稳定的 `Pxxx` 编号；重复的内参、
FOV 和畸变数据整理成独立的 `Kxxx` 标定记录供 profile 引用。相机正被 ROS 节点占用时，
只要 SDK 仍允许枚举，`cameras-inspect` 也可以正常使用。

需要真正验证所有配置时，单独运行 `cameras-validate`：

```bash
ros2 run bxi_depth_camera cameras-validate
```

验证程序会先输出与 `cameras-inspect` 相同的完整清单，再对每个 RGB/Depth profile 分别
执行一次 `open/start`，等待并读取一帧，核对实际帧的分辨率、FPS 和格式，最后执行
`stop/close`。它还会从成功的深度帧报告原始深度值到米/毫米的转换比例。

直接在终端运行时，验证阶段使用无额外依赖的 ANSI TUI，实时显示总进度、PASS/FAIL、
当前 profile 和最近失败项；完成或按 Ctrl-C 后会恢复原来的终端内容，再输出验证汇总和
全部失败项。输出被重定向到文件或管道时会自动退化成普通的逐项文本进度，不会写入
ANSI 控制字符。这里没有引入架构相关的 TUI 二进制，因此 x86_64 与 aarch64 使用同一份
实现，无需再维护两套预编译库。

只有所有 profile 都成功读帧时，`overall verification` 才为 `PASS`；非监听模式下，只要
一项失败，进程就返回非零退出码。因此完整检查可能需要数分钟，并且运行期间相机不能
被其他节点占用。这里验证的是每个单独的 RGB 或 Depth profile，不会枚举 RGB 与 Depth
的笛卡尔组合。

`cameras-validate` 单项默认最多等待 3000 ms；低帧率设备或 USB 链路启动较慢时可以
增大超时：

```bash
ros2 run bxi_depth_camera cameras-validate --frame-timeout-ms 5000
```

`cameras-validate` 只证明能够取得一帧以及帧的 profile 元数据与请求一致，不测量持续
吞吐。需要验证真实帧率时，使用第三个独立程序：

```bash
ros2 run bxi_depth_camera cameras-fps-test
```

它默认对每个 profile 预热 3 帧，再连续测量 3 秒，同时统计：

- 主机实际收帧 FPS；
- 设备时间戳 FPS；
- 主机帧间隔的 P95 和最大值；
- 采样帧数以及根据帧号检测到的丢帧数。

非 TTY 逐项结果、交互式 TUI 当前结果和最终汇总都会显示每个 profile 的真实采样
帧数（`frames`）；即使启动失败且一帧未收到，也会明确显示 `frames=0`。

主机 FPS 和设备 FPS 都不得低于标称 FPS 的 95%，且测量窗口内不得丢帧，否则该
profile 为 `FAIL`。可以调整测量时长、预热帧数和允许误差：

```bash
ros2 run bxi_depth_camera cameras-fps-test \
  --measure-seconds 5 \
  --warmup-frames 5 \
  --fps-tolerance-percent 5 \
  --frame-timeout-ms 5000
```

该程序会串行测试全部 profile，完整运行时间约为“profile 数量 ×（预热时间 + 测量
时间）”；数百个 profile 可能需要几十分钟。可以用 `--serial` 只选择一台相机，Ctrl-C
会在当前 profile 收尾后停止并输出已完成部分的报告。

需要通过拔插确认某一台相机的序列号时，使用持续监听模式；程序会在每次接入和
拔出时打印相机厂商、型号和稳定的硬件序列号，新接入时还会读取完整参数：

```bash
ros2 run bxi_depth_camera cameras-inspect --watch
```

也可以只等待或检查已知序列号，并调整轮询间隔：

```bash
ros2 run bxi_depth_camera cameras-inspect --watch \
  --serial 349422070502 --interval 0.5
```

序列号枚举不需要启动 ROS 相机节点。如果相机已被其他进程独占，`cameras-inspect` 仍会
显示序列号和 SDK 声明的 profile，但 `cameras-validate` 的逐项开流会显示 `FAIL`。要取得
全部 `PASS` 的验证报告，应先停止占用该相机的节点。

真机话题按 ROS 相机惯例组织在 `hardware` 命名空间下，并使用机器人部署配置中的逻辑相机名称：

```text
/hardware/<camera_name>/color/image_raw
/hardware/<camera_name>/color/camera_info
/hardware/<camera_name>/depth/image_rect_raw
/hardware/<camera_name>/depth/camera_info
/hardware/<camera_name>/aligned_depth_to_color/image_raw
/hardware/<camera_name>/aligned_depth_to_color/camera_info
/hardware/<camera_name>/depth/color/points
/hardware/<camera_name>/infra1/image_rect_raw
/hardware/<camera_name>/infra1/camera_info
/hardware/<camera_name>/infra2/image_rect_raw
/hardware/<camera_name>/infra2/camera_info
/hardware/<camera_name>/gyro/sample
/hardware/<camera_name>/accel/sample
```

例如 MuJoCo 中 `<camera name="head_depth_camera">` 对应真机
`/hardware/head_depth_camera/depth/image_rect_raw`。序列号只用于打开物理设备，不进入策略
话题名称。

未配置序列号映射且只发现一台相机时，默认自动使用逻辑名称
`head_depth_camera`，因此单相机机器人不需要配置序列号：

```text
/hardware/head_depth_camera/color/image_raw
/hardware/head_depth_camera/depth/image_rect_raw
```

该名称由 `single_camera_name` 参数控制。发现两台或更多未映射相机时，为避免连接
顺序改变头部相机身份，自动回退为 `SN_<serial>`；此时应通过
`cameras.<logical_name>.serial_no` 明确映射。显式序列号映射始终优先。运行中插拔使
设备数量在一台和多台之间变化时，相机 worker 会重启并切换到对应话题名称。

图像主话题贴近 `realsense2_camera`：彩色使用 `color/image_raw`，深度与红外使用
`image_rect_raw`，每个流同时发布 `camera_info`。每个流只发布一个主图像话题，
对应去畸参数控制发布前是否执行软件去畸，不会额外发布一套重复图像。IMU 使用
RealSense ROS 驱动也采用的 `gyro/sample` 和 `accel/sample` 话题结构。

## 深度对齐到彩色图

设置与 `realsense-ros` 同名的 `align_depth.enable` 后，会额外发布一张投影到彩色
相机成像平面的 `16UC1` 毫米深度图：

```bash
ros2 launch bxi_depth_camera cameras.launch.py \
  align_depth.enable:=true
```

```text
/hardware/<camera_name>/aligned_depth_to_color/image_raw
/hardware/<camera_name>/aligned_depth_to_color/camera_info
```

对齐图的宽高、`frame_id` 和 `CameraInfo` 都使用彩色相机坐标系。原始的
`depth/image_rect_raw` 会继续发布，不会改变依赖原深度尺寸的策略输入。对齐要求
同时开启深度和彩色流；缺少任一流时参数更新会被拒绝。

参数也支持按逻辑相机覆盖：

```yaml
/depth_camera_manager:
  ros__parameters:
    cameras:
      head_depth_camera:
        serial_no: "349422070502"
        align_depth:
          enable: true
```

RealSense 使用 librealsense 的 Depth-to-Color 标定与 `align` 处理块；Orbbec 使用
SDK 的 `AlignFilter`，并在开启时启用帧同步。若彩色流开启软件去畸，对齐深度会
应用同一套彩色去畸映射，并使用最近邻插值，保证它仍与发布的彩色图逐像素对应。

对齐输出分辨率等于彩色流分辨率。例如彩色配置为 `1920x1080x30` 时，一路
`16UC1` 对齐深度自身约产生 119 MiB/s 的未压缩像素数据，并增加 SDK 对齐和 ROS
发布开销；不需要高清彩色图时，建议选择较低且设备支持的彩色 profile。

## 点云

点云默认关闭，开启方式与 `realsense-ros` 一致：

```bash
ros2 launch bxi_depth_camera cameras.launch.py \
  pointcloud.enable:=true \
  pointcloud.max_fps:=10.0
```

发布话题为：

```text
/hardware/<camera_name>/depth/color/points
```

消息类型是 `sensor_msgs/msg/PointCloud2`。点坐标 `x/y/z` 使用 `float32` 米；彩色
流可用时还包含 PCL/ROS 惯例的 packed `rgb` 字段。RGB 只是从彩色图映射到深度
点上的纹理，XYZ 仍在深度相机坐标系中，因此消息的 `frame_id` 是
`<camera_name>_depth_optical_frame`。

可用参数：

```yaml
pointcloud.enable: false
pointcloud.ordered_pc: false
pointcloud.allow_no_texture_points: false
pointcloud.max_fps: 10.0
```

- `ordered_pc=false`：删除无效点，输出 `height=1` 的紧凑点云；
- `ordered_pc=true`：保持深度图宽高，无效点写为 NaN；
- `allow_no_texture_points=true`：RealSense 彩色视场之外的有效深度点也会保留，
  颜色填零；
- `max_fps`：独立限制点云频率，不改变图像流帧率。

点云使用深度流分辨率生成，再通过相机外参采样彩色纹理，不会因为开启
`align_depth` 而扩展到 1920x1080。节点仅在点云话题存在订阅者且达到限频周期时
调用 SDK；无人订阅时不会计算和序列化点云。点云在独立 latest-only 后台线程中
生成，处理落后时覆盖尚未开始的旧帧，不阻塞图像发布。当前默认深度 `848x480`
最多约 40.7 万点，满 XYZRGB 负载约 6.2 MiB/帧，因此高帧率或跨机 DDS 传输时
仍建议配合 decimation 或降低 `pointcloud.max_fps`。

按相机单独配置示例：

```yaml
/depth_camera_manager:
  ros__parameters:
    cameras:
      head_depth_camera:
        serial_no: "CP0F4630000L"
        pointcloud:
          enable: true
          ordered_pc: false
          max_fps: 5.0
```

## 按订阅者惰性处理

相机 SDK pipeline 和设备 watchdog 始终运行，但发布侧的高成本处理会根据 ROS
订阅数按需执行：

- 深度图或其 `camera_info` 无订阅者时，跳过深度滤波、去畸和消息拷贝；
- 彩色图无订阅者时，跳过 BGR 转换、MJPEG 解码、去畸和消息拷贝；
- 对齐深度图和其 `camera_info` 均无订阅者时，跳过 Depth-to-Color 对齐；
- 红外图无订阅者时，跳过数组转换、去畸和消息拷贝；
- IMU 话题无订阅者时，不构造对应的 `Imu` 消息；
- 点云额外使用订阅门控、频率限制和 latest-only 后台队列。

因此可以一直开启相机能力，RViz、录包或策略节点开始订阅后的下一帧会自动恢复
相应处理，不需要重启 pipeline。Orbbec 在配置了深度对齐时仍保持帧同步和完整帧
聚合，但无人订阅时不会调用每帧 `AlignFilter` 或构造对齐图消息。

SDK 回调也采用同一门控：无人订阅时只刷新设备存活时间，不搬运帧、不唤醒 ROS
executor。已排队的点云若遇到订阅者退出会直接丢弃，避免完成一次已经无人消费的
后台计算。

## 按流去畸

深度、RGB 和两路红外可以独立开启去畸。开启后会在发布主图像话题前完成校正，
话题名称保持稳定：

```text
/hardware/<camera_name>/depth/image_rect_raw
/hardware/<camera_name>/color/image_raw
/hardware/<camera_name>/infra1/image_rect_raw
/hardware/<camera_name>/infra2/image_rect_raw
```

全局开启示例：

```bash
ros2 launch bxi_depth_camera cameras.launch.py \
  depth_module.rectification.enable:=true \
  rgb_camera.rectification.enable:=true \
  infra1.rectification.enable:=false \
  infra2.rectification.enable:=false
```

也可以只为某个逻辑相机开启：

```yaml
/depth_camera_manager:
  ros__parameters:
    cameras:
      head_depth_camera:
        serial_no: "349422070502"
        depth_module:
          rectification:
            enable: true
        rgb_camera:
          rectification:
            enable: false
        infra1:
          rectification:
            enable: true
        infra2:
          rectification:
            enable: false
```

校正映射表按相机内参和分辨率缓存。深度使用最近邻插值，避免生成虚假的中间深度；
RGB 和红外使用线性插值。遇到 SDK 不支持的畸变模型时，节点会发布 SDK 原始帧并
输出警告。

launch 会自动加载包内的 `config/default.yaml`；profile `0,0,0` 表示使用设备
默认值。也可以显式覆盖：

```bash
ros2 launch bxi_depth_camera cameras.launch.py \
  depth_module.depth_profile:=480x270x60 \
  enable_color:=true
```

包内提供完整默认配置 `config/default.yaml`，默认开启深度和彩色流，并为彩色流
开启软件去畸；正常启动时会自动加载：

```bash
ros2 launch bxi_depth_camera cameras.launch.py
```

部署时建议先复制该文件，再添加本机的逻辑相机名与序列号映射，避免直接修改安装
目录中的模板。
参数优先级为包内默认配置、`config_file`、命令行显式参数；后者可以覆盖前者。

默认打开全部设备；需要只打开一台时可使用与 `realsense-ros` 同名的参数：

```bash
ros2 launch bxi_depth_camera cameras.launch.py serial_no:=349422070502
```

全局参数是所有相机的默认值。每台机器人需要把稳定的逻辑名称映射到实际序列号：

```bash
ros2 param set /depth_camera_manager \
  cameras.head_depth_camera.serial_no '"349422070502"'
```

部署时更适合保存为每台机器自己的 YAML：

```yaml
/depth_camera_manager:
  ros__parameters:
    cameras:
      head_depth_camera:
        serial_no: "349422070502"
        depth_module:
          depth_profile: "640x480x30"
```

启动时加载，映射会在设备首次打开前生效：

```bash
ros2 launch bxi_depth_camera cameras.launch.py \
  config_file:=/etc/bxi/cameras.yaml
```

然后可以使用逻辑名称单独设置 profile、流开关和滤波参数；只有目标相机的
pipeline 会重建：

```bash
ros2 param set /depth_camera_manager \
  cameras.head_depth_camera.depth_module.depth_profile "640x480x30"

ros2 param set /depth_camera_manager \
  cameras.rear_cam.depth_module.depth_profile "848x480x30"
```

未设置的单机参数继承对应的全局参数。删除单机覆盖后也会恢复继承并重建目标
pipeline：

```bash
ros2 param delete /depth_camera_manager \
  cameras.head_depth_camera.depth_module.depth_profile
```

全局相机参数也支持运行时修改；修改后会重建所有已连接相机：

```bash
ros2 param set /depth_camera_manager depth_module.depth_profile "640x480x30"
```

参数会先检查名称、类型和取值范围。具体 profile 是否受设备支持由 SDK 在重建
pipeline 时确认；不支持时节点会记录错误并按 `retry_interval_sec` 重试。

运行节点和探测工具均为原生 C++ 可执行文件：

```bash
ros2 run bxi_depth_camera cameras
ros2 run bxi_depth_camera cameras-inspect --watch
ros2 run bxi_depth_camera cameras-validate --frame-timeout-ms 5000
ros2 run bxi_depth_camera cameras-fps-test --measure-seconds 5
```

相机处理采用 SDK callback 的 latest-only 队列；点云在独立 latest-only 工作线程中
生成。慢订阅者不会让采集线程积压历史帧。
