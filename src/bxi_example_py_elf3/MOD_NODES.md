# Mod 节点清单

Mod API 1.2 支持三种节点运行时，同时复用相同的生命周期、参数、重启和展示字段：

- `python`：Mod 内的 Python 工厂，返回一个 `rclpy.Node`。
- `executable`：Mod 内随包分发的原生可执行文件，适用于 C++ ROS 2 节点。
- `ros`：通过 ament index 查找已安装 ROS 2 包中的可执行文件，行为等价于
  `ros2 run`。

使用 `executable` 或 `ros` 的 Mod 应声明 `api: ">=1.2,<2"`。

## 公共格式

```yaml
nodes:
  detector:
    runtime: executable
    entrypoint: detector_node
    execution: process
    lifecycle: state
    states: [detect]
    arguments: [--device, "0"]
    namespace: /vision
    remappings:
      image: /camera/image_raw
    params:
      threshold: 0.5
      enabled: true
    manifest:
      label: C++ 检测节点
    runtime_requirements:
      python: []
      ros:
        - package: sensor_msgs
      system: []
    restart:
      max_attempts: 3
      delay: 1.0
```

字段说明：

- `runtime`：`python`、`executable` 或 `ros`。为兼容旧 Mod，省略时默认为
  `python`。
- `entrypoint`：其格式由 `runtime` 决定，具体见下文。
- `execution`：Python 节点支持 `in_process` 和 `process`；原生节点必须是
  `process`。原生节点省略该字段时默认使用 `process`。
- `lifecycle`：`mod` 表示随 Mod 常驻，`state` 表示仅在 `states` 指定的状态
  活跃或预加载时运行。
- `arguments`：放在可执行文件之后、`--ros-args` 之前的普通程序参数。
- `namespace`：空字符串或以 `/` 开头的绝对 ROS namespace。
- `remappings`：ROS 名称重映射表。
- `params`：节点参数。原生节点会收到自动生成的标准 ROS 2 参数文件。
- `restart`：仅适用于进程节点，沿用现有退出监控和重启策略。

所有原生节点都会收到框架生成的唯一节点名。实际命令形如：

```text
<executable> <arguments...> --ros-args \
  -r __node:=com_example_detector_detector \
  -r __ns:=/vision \
  -r image:=/camera/image_raw \
  --params-file <temporary-file>
```

临时参数文件由节点管理器持有，在管理器关闭时删除。

## Python 节点

```yaml
nodes:
  camera:
    runtime: python
    entrypoint: camera_node:create_node
    execution: process
    lifecycle: mod
    arguments: []
    namespace: ""
    remappings: {}
    params:
      fps: 30
    manifest:
      label: Python 相机节点
```

Python 工厂通过 `NodeBuildContext` 获得 `node_name`、`params`、`arguments`、
`namespace` 和 `remappings`。Python 工厂负责在构造 `rclpy.Node` 时使用这些值。

## Mod 内原生节点

一个 Mod 可以同时携带多个平台的同名二进制：

```text
com.example.detector/
├── mod.yaml
├── bin/
│   ├── linux-x86_64/
│   │   └── detector_node
│   └── linux-aarch64/
│       └── detector_node
└── vendor/
    └── lib/
        ├── linux-x86_64/
        └── linux-aarch64/
```

对应清单只写平台无关的文件名：

```yaml
runtime: executable
entrypoint: detector_node
```

运行时使用与 `runtime_platform_tag()` 相同的平台标签，自动解析
`bin/<platform>/<entrypoint>`。不会回退到其他平台。文件缺失或没有执行权限时，
节点状态为 `unavailable`，其他可用节点仍可继续加载。

`entrypoint` 必须是相对于平台 `bin` 目录的安全路径；绝对路径、`..` 和解析后
逃出该目录的符号链接都会被拒绝。Mod 私有动态库放在
`vendor/lib/<platform>/`，该目录会被优先加入子进程的 `LD_LIBRARY_PATH`。

## 已安装 ROS 包节点

```yaml
nodes:
  detector:
    runtime: ros
    entrypoint: detector_package:detector_node
    execution: process
    lifecycle: mod
    arguments: []
    namespace: ""
    remappings: {}
    params: {}
    manifest:
      label: 系统检测节点
```

运行时通过 ament index 获取包前缀，并直接执行
`<prefix>/lib/detector_package/detector_node`。直接执行避免了监管 `ros2 run`
包装进程，停止和异常重启仍由同一个 Mod 节点管理器负责。
