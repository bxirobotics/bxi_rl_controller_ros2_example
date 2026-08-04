# Mod 目录、清单与状态图

## 目录

- [目录选择](#目录选择)
- [必填描述头](#必填描述头)
- [简单状态与显式插件](#简单状态与显式插件)
- [强类型参数](#强类型参数)
- [名称限定](#名称限定)
- [Event、route 与 action](#eventroute-与-action)
- [Speed profile](#speed-profile)
- [安全占位符](#安全占位符)

## 目录选择

最小动作 Mod：

```text
com.example.wave/
├── mod.yaml
└── state.py
```

含模型、共享对象或自定义扩展：

```text
com.example.motion/
├── mod.yaml
├── plugin.py
├── state.py
└── assets/
    ├── motion.npz
    └── policy.onnx
```

不要创建无用途的 README、脚本或资产目录。不要提交 `__pycache__`、`.pyc`、运行日志、临时模型缓存或机器专属生成文件。

仓库内可参考：

- `com.bxi.back_flip`：最薄的模型回放 Mod；
- `com.bxi.basic_actions`：一个 Mod 注册多个基础状态和共享资源；
- `com.bxi.any_motion`：强类型动作参数、相对资产校验和按状态资源键；
- `com.bxi.normal_depth`：传感器状态与标准 ROS 话题；
- `com.bxi.sonic`：状态级辅助进程、节点依赖和硬件控制。

## 必填描述头

清单 schema 1 要求以下 12 个描述字段全部显式存在，即使内容为空：

```yaml
schema: 1
id: com.example.wave
name: 挥手示例
version: 1.0.0
api: ">=4,<5"
enable: true
entrypoint: null
visibility: public
requires:
  - id: com.bxi.basic_actions
    version: ">=1,<2"
conflicts: []
python_exports: []
runtime_requirements:
  python: []
  ros: []
  system: []
```

规则：

- `enable` 必须是 YAML 布尔值。
- `id` 全局唯一；目录一般与 id 一致。
- 当前公共 API 是 4.x，项目 Mod 使用 `api: ">=4,<5"`。
- 简单约定式状态写 `entrypoint: null`；高级入口写 `plugin:create_mod`，框架不会猜测。
- `visibility` 为 `public` 或 `protected`。高危或不应公开交付的完整 Mod 使用 `protected`。
- 跨 Mod 状态、事件、profile 或资源引用必须配套 `requires`。
- `conflicts` 单向声明、双向生效。
- Python import、ROS package 和系统库属于 `runtime_requirements`，不是 `requires`；框架只检查，不自动安装。
- 不添加未知顶层字段。

## 简单状态与显式插件

约定式加载：

```yaml
entrypoint: null

states:
  wave:
    factory: state:WaveState
    manifest:
      label: 挥手
      priority: 100
      group: Customer
      icon: waving_hand
```

显式插件：

```python
from bxi_example_py_elf3.framework.mod_api import ModDefinition, ModLoadContext

from .state import WaveState


def create_mod(context: ModLoadContext) -> ModDefinition:
    return ModDefinition(
        state_factories={
            "wave": lambda state: WaveState(state.name, state.state_id),
        }
    )
```

`states` 与 `state_factories` 的本地键必须完全一致。工厂返回后框架会检查参数是否全部消费。

## 强类型参数

简单状态优先在类上声明 dataclass `Params`：

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WaveParams:
    amplitude: float = 0.25
    frequency: float = 0.5
    joint: str = "r_elbow_y_joint"


class WaveState(RobotControlState):
    Params = WaveParams
```

约定工厂会构造 `WaveState(name, state_id, params)`。显式插件使用 `state.dataclass_params(WaveParams)`，或逐项使用 `int_param/float_param/string_param/bool_param/param`。dataclass 可在 `__post_init__()` 验证范围和组合约束。

未知参数、错误类型和未消费参数必须在加载阶段失败；不要静默忽略以求兼容。

## 名称限定

- 不含 `/` 的 state、event、node state reference、speed profile 和 Mod transition profile 会自动加当前 Mod id。
- 含 `/` 的值是完整名称。
- `ResourceKey` 必须自行写完整全局命名空间，例如 `com.example.wave/policy`。
- 跨 Mod 引用既要完整名称，也要在 `requires` 中声明依赖。
- 普通状态用 `manifest.priority` 排序；相同 priority 按完整状态名排序。除外部协议强制要求外，不设置 `manifest.index`。

## Event、route 与 action

```yaml
events:
  activate:
    slot: btn_10
    value: 11
  toggle_pause:
    slot: btn_9
    value: 1

routes:
  - from: com.bxi.basic_actions/normal
    event: activate
    to: wave
    transition: first_frame_switch
  - from: wave
    event: com.bxi.basic_actions/normal
    to: com.bxi.basic_actions/normal
    transition:
      profile: dual_running_blend
      duration: 0.6

actions:
  - from: wave
    event: toggle_pause
    action: toggle_pause
    manifest:
      label: 暂停/继续
```

约束：

- event 必须至少被一条 route 或 action 使用。
- route 必须包含 `to`，不得包含 `action`。
- action 不切换状态，必须包含非空 `manifest.label`，不得包含 `to/transition/delay`。
- 同一源状态、同一事件只能有一个 route 或 action。
- `value` 通常显式填写；部署前检查整棵启用 Mod 树和遥控输出，避免可同时触发的冲突。
- 安全退出 route 必须覆盖所有实际可达的运行状态。

## Speed profile

```yaml
speed_profiles:
  walk:
    vx_scale: 1.0
    vx_min: -1.0
    vx_max: 1.0
    vy_scale: 0.5
    yaw_scale: 1.5

states:
  wave:
    speed_profile: walk
```

状态通过 `self.get_cmd_vel(ctx)` 读取，不直接识别键盘、手柄或 CRSF。

## 安全占位符

清单字符串只支持框架固定的 `${bxi.…}` 预设，如 `${bxi.system.ip}`、`${bxi.system.machine}`、`${bxi.python.version}`。未知 BXI 占位符会阻止加载。

这不是 Shell：`$(uname)`、反引号、管道和 `${HOME}` 不执行。不要为了“灵活配置”引入 `eval`、`shell=True` 或命令替换，尤其程序可能以 root 运行。命令节点的环境变量展开也只做受限的字符串替换。
