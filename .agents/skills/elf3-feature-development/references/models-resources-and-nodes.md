# 模型、Resource 与 Mod 子节点

## 目录

- [什么应该成为 Resource](#什么应该成为-resource)
- [startup 与 on_demand](#startup-与-on_demand)
- [模型契约](#模型契约)
- [何时用 nodes](#何时用-nodes)
- [节点清单示例](#节点清单示例)
- [Host、Vendor 与 Portable](#hostvendor-与-portable)
- [依赖声明](#依赖声明)

## 什么应该成为 Resource

模型 session、动作数据、配置数据集、长期连接、硬件句柄和其他昂贵或需要统一关闭的对象都由 Resource 拥有。不要在模块 import、状态构造器或 `on_update()` 中加载它们。

```python
from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ResourceKey,
    ResourceLoadContext,
)


POLICY = ResourceKey[MotionPolicy]("com.example.motion/policy")


def _load_policy(context: ResourceLoadContext) -> MotionPolicy:
    return MotionPolicy(str(context.asset("assets/policy.onnx")))


def create_mod(context) -> ModDefinition:
    context.register_resource(POLICY, _load_policy, policy="on_demand")
    policy = context.resource(POLICY)
    return ModDefinition(
        state_factories={
            "motion": lambda state: MotionState(
                state.name,
                state.state_id,
                policy,
            )
        }
    )
```

规则：

- `ResourceKey` 使用完整全局命名空间；按状态独立资源可用 `f"{state.name}/policy"`。
- 资产位于所属 Mod 的 `assets/`，通过 `context.asset("assets/...")` 访问；不要硬编码 source/install 绝对路径。
- 状态构造时声明 `resources=(handle,)`，或使用自动声明依赖的 `PolicyState`/`MotionReplayState`。
- `ResourceHandle.get()` 只返回已经 ready 的对象，不触发加载、不阻塞等待。
- 实例有 `close()` 时，资源管理器在关闭时逆序调用。
- 模型 loader 应立刻验证输入输出张量、metadata、具名关节布局和静态配置，使错误发生在准备阶段。

## startup 与 on_demand

| 策略 | 时机 | 使用场景 | 失败语义 |
| --- | --- | --- | --- |
| `startup` | 控制循环启动前 | 初始状态依赖、启动时必须验证的核心资源 | 阻止框架启动 |
| `on_demand` | 首次请求相关状态时由后台线程准备 | 大型、不常进入的动作或可选能力 | 保持当前状态，拒绝/终止该次切换 |

初始状态的资源必须使用 `startup`，因为此时没有来源状态可以在 preparing 期间继续控制机器人。资源加载策略写在注册代码中，不写进 `mod.yaml`。

多个状态共享对象时使用同一个 handle。多个 Mod 共享时考虑拆出无状态 Resource Mod，并通过 `requires` 建立依赖。

## 模型契约

Sim2Real 前固定并校验：

- observation 与 action 的真实关节名称和顺序；
- 四元数顺序、角速度和速度命令坐标系；
- observation 归一化、history 长度、控制频率；
- action scale、默认姿态、kp/kd 和输出含义；
- motion fps、start/end frame 和 trim；
- 模型输出布局与 Policy/State 最终输出布局之间的确定性合成。

不要从数组长度、机器人消息顺序或类名猜模型布局。固定参数用带布局的参数集，不维护四组脱离名称的裸数组。

## 何时用 `nodes`

相机发布器、感知预处理、通信桥、设备 manager 和 SDK 服务属于可独立运行的后台能力，声明在清单顶层 `nodes`。轻量、只属于一个状态对象的订阅/service/timer 才放在状态 `on_bind/on_unbind`。

支持的 runtime：

| runtime | 入口 | 用途 |
| --- | --- | --- |
| `python` | `module:create_node` | 返回 `rclpy.Node` 的 Mod 内 Python 工厂 |
| `executable` | Mod `bin/<platform>/` 中的程序 | 随 Mod 分发的原生 ROS 节点 |
| `ros` | `package:executable` | 已安装 ament 包节点 |
| `command` | Mod 内脚本/命令 | 不自动追加 ROS 参数的普通后台程序 |

不要从 State 里随意 `subprocess.Popen`；那会绕过状态准备、依赖排序、重启、日志和关闭管理。

## 节点清单示例

```yaml
nodes:
  manager:
    runtime: command
    entrypoint: scripts/manager.py
    interpreter: python3
    execution: process
    lifecycle: state
    states: [teleop]
    scheduling:
      cpu_affinity: background
    arguments: [--port, "5556"]
    environment:
      PYTHONUNBUFFERED: "1"
    manifest:
      label: 设备管理器
    runtime_requirements:
      python: []
      ros: []
      system: []
    restart:
      max_attempts: 3
      delay: 2.0
      non_retryable_exit_codes: [78]
    shutdown:
      signal: SIGINT
      terminate_after: 3.0
      kill_after: 5.0

  bridge:
    runtime: python
    entrypoint: bridge:create_node
    execution: in_process
    lifecycle: state
    states: [teleop]
    depends_on: [manager]
    params:
      output_topic: /example/reference
    manifest:
      label: 参考桥
```

关键语义：

- `lifecycle: mod` 随启用 Mod 常驻；`lifecycle: state` 必须列出非空 `states`，在目标 prepare 前启动，离开最后一个关联状态后停止。
- `depends_on` 按拓扑顺序启动、逆序停止；局部名称自动限定到当前 Mod。
- `execution: process` 支持有限重启和进程级 CPU affinity；`in_process` 共享 Executor，不能声明 scheduling。
- 普通异常保留重启机会；只有确定性配置错误才使用不可重试退出码（通常 78）。
- shutdown 最终会升级到 SIGTERM/SIGKILL，信号发向整个进程组。
- 子进程 stdout/stderr 会被持续排空并添加 Mod/node 来源，不能让输出管道反向堵塞程序。

## Host、Vendor 与 Portable

默认 `host` 使用宿主环境。只有需要随 Mod 携带依赖时才声明：

- `vendor`：宿主 Python 加当前平台匹配的 `vendor/python/<平台-Python ABI>`、`vendor/python/common` 和 `vendor/lib/<平台>`；
- `portable`：Mod 内完整运行根，只允许独立进程，可携带专用 Python/C++/SDK 环境。

不要把 aarch64 与 x86_64 原生依赖混用。Portable fallback 只在 candidate 根完全不存在时继续；已经存在但损坏时应明确失败。

`command` 的 `entrypoint`、`cwd` 和 Portable 路径必须留在 Mod 内。环境变量只做 `$NAME`、`${NAME}`、`${NAME:-default}` 的受限展开，不经过 Shell，不执行 `$(...)`、反引号、管道或重定向。

## 依赖声明

顶层 `runtime_requirements` 决定整个 Mod 是否 available；节点级 requirements 只影响该节点。缺失依赖不会自动安装：

```yaml
runtime_requirements:
  python:
    - import: zmq
  ros:
    - package: sensor_msgs
  system:
    - library: realsense2
```

顶层缺失且存在依赖链时可阻止启动；节点依赖缺失时节点为 `unavailable`，不会导入模块或进入重启循环。运行后异常退出则是 `faulted`，要区分配置不可用与运行故障。
