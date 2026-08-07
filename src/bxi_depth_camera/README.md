# bxi_depth_camera

独立的 ROS 2 深度相机发布包。它在运行期间发现、打开和重连所有受支持的
Intel RealSense 与 Orbbec Gemini 335，不包含任何策略专属裁剪。

```bash
ros2 launch bxi_depth_camera cameras.launch.py
```

## 相机参数与序列号探测

安装并 source 工作空间后，可以直接读取当前相机的硬件序列号、默认输出流尺寸、
帧率、FOV、相机内参以及原始深度值到米/毫米的转换比例：

```bash
ros2 run bxi_depth_camera cameras-inspect
```

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

序列号枚举不需要启动 ROS 相机节点。读取输出流参数时探测程序会短暂打开设备；
如果相机已经被其他进程独占，程序仍会显示序列号，并提示流参数暂时无法读取。
要取得完整参数，应先停止占用该相机的节点。

真机话题按 ROS 相机惯例组织在 `hardware` 命名空间下，并使用机器人部署配置中的逻辑相机名称：

```text
/hardware/<camera_name>/color/image_raw
/hardware/<camera_name>/color/camera_info
/hardware/<camera_name>/depth/image_rect_raw
/hardware/<camera_name>/depth/camera_info
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
RealSense ROS 驱动也采用的 `gyro/sample` 和 `accel/sample` 结构。

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

节点优先使用系统中可导入的 `pyrealsense2` 和 `pyorbbecsdk`。只有系统导入失败
时，启动器才为相应 SDK 启用 `vendor/python/<platform>-<python-abi>` 中的内置
runtime。内置 runtime 不会加入其他 ROS 2 进程的 `PYTHONPATH`。
