# 功能封装、按键与 App 规划

## 目录

- [默认封装原则](#默认封装原则)
- [编码前的功能契约表](#编码前的功能契约表)
- [btn 槽位和值规划](#btn-槽位和值规划)
- [route 与 action 规划](#route-与-action-规划)
- [App 展示规划](#app-展示规划)
- [单 Mod 移植验收](#单-mod-移植验收)

## 默认封装原则

收到“实现某个功能、状态、动作或控制模式”时，先把它当成一个可独立安装的功能包规划，再决定内部类和文件。

默认新建一个额外 Mod，避免直接扩张 `com.bxi.basic_actions` 或中央控制类。一个功能所需的以下内容尽量全部属于同一 Mod：

- 一个或多个紧密协作的状态；
- events、routes、actions 和 speed profiles；
- 状态工厂、策略、Resource 和资产；
- 感知、转换、通信或设备辅助 nodes；
- Python/ROS/system 运行依赖及平台运行环境；
- App 所需的 state/action manifest。

即使功能需要 manager、bridge、converter 等多个进程，也先把它们声明成同一 Mod 的 `nodes`，用 `depends_on` 和 lifecycle 管理，不据此拆成多个业务 Mod。

以下情况可以不新建业务 Mod：

- 修复已有功能本身的缺陷，外部契约不变；
- 增加所有 Mod 都需要的框架公共 API 或通用 Transition；
- 接入机器人平台、电机协议或新的全局 Input Driver；
- 多个独立 Mod 确实共享同一个昂贵资源，并需要单独版本和生命周期；
- 用户明确要求扩展某个现有 Mod。

选择例外时记录理由。不要以“代码少”为由污染基础 Mod，也不要以“解耦”为由让一个功能必须同步安装多个包才能工作。

## 编码前的功能契约表

先写清以下决策；能从仓库发现的内容不要反问用户：

| 项目 | 要确定的内容 |
| --- | --- |
| 功能边界 | Mod id、用户目标、包含哪些状态、哪些内容明确不属于它 |
| 进入方式 | 从哪些状态可进入、使用哪个 event、btn 槽位和值 |
| 物理输入 | Xbox/键盘/CRSF 等哪些组合产生该 btn output |
| 运行内操作 | 暂停、复位、模式切换等使用哪些 action |
| 退出方式 | normal 返回、动作完成、取消、pd_brake、zero_torque 和故障退回 |
| App 展示 | state/action label、priority、group、icon、confirm 与提示语 |
| 控制输出 | 关节布局、MIT 五字段、多个命令来源及所有权 |
| 外部输入 | 话题、消息、QoS、新鲜度、超时和恢复行为 |
| 运行能力 | Resource、nodes、依赖、启动策略和关闭顺序 |
| 移植检查 | 复制一个 Mod 目录后还缺哪些配置或平台能力 |

只有物理按键映射、平台固定 Defaults、系统级配置等天然属于 Mod 外部的内容才允许成为移植前置条件，并在交付说明中列出。

## btn 槽位和值规划

完整输入链是：

```text
物理按键/设备信号
  -> remote controller controls
  -> outputs.level 或 outputs.edge
  -> MotionCommands.btn_N=value
  -> Mod events.<event>.slot/value
  -> route 或 action
```

在选择槽位前扫描当前占用：

```bash
rg -n 'slot: btn_|value:' src/bxi_example_py_elf3/mods/*/mod.yaml
rg -n 'output: btn_|shoulder\.|trigger\.|button\.' \
  src/remote_controller/config/xbox_default.yaml
```

判断冲突不能只看全局是否重复，还要看同一个来源状态是否可能同时触发多个目标。优先复用项目已经规划好的空闲 `btn_*` 值；不要仅凭数字顺眼自行占用。

Mod 只声明抽象事件：

```yaml
events:
  activate:
    slot: btn_10
    value: 11
  toggle_pause:
    slot: btn_9
    value: 1
```

物理组合键属于 `src/remote_controller/config/xbox_default.yaml`。新增组合时明确要求未使用的辅助键处于 released，避免一个按键同时命中较短组合。例如组合结构应完整表达按下与释放条件：

```yaml
- output: btn_10=11
  when:
    - shoulder.left_event
    - shoulder.right_event
    - button.a_event
    - released: trigger.left_event
    - released: trigger.right_event
```

当前配置中，`when` 直接给列表表示全部条件同时满足；需要手柄和键盘等多套替代组合时使用 `when.any`，其中每个子列表是一组完整条件。修改前仍要核对当前代码和 YAML schema。若一个物理绑定要支持多种设备，让不同设备 control 产生同一个抽象 event，不把设备判断写进状态类。

`xbox_default.yaml` 已预留四个辅助键 LB/RB/LT/RT 的全部 16 种状态与 A/B/X/Y 的组合。规划新功能时先从下表选择尚未被启用 Mod 占用的槽位和值，只需在 Mod 中声明对应 event；不要重复修改 remote controller 配置。表中 LB/RB 分别对应左右肩键，LT/RT 分别对应左右扳机：

| 辅助键 | A | B | X | Y |
| --- | --- | --- | --- | --- |
| 无 | `btn_10=11` | `btn_1=2` | `btn_9=1` | `btn_10=12` |
| LB | `btn_6=1` | `btn_7=1` | `btn_5=1` | `btn_8=1` |
| RB | `btn_2=1` | `btn_3=1` | `btn_1=1` | `btn_4=1` |
| LB+RB | `btn_10=13` | `btn_10=14` | `btn_10=9` | `btn_10=10` |
| LT | `btn_10=3` | `btn_10=4` | `btn_10=1` | `btn_10=2` |
| LB+LT | `btn_10=15` | `btn_10=16` | `btn_10=17` | `btn_10=18` |
| RB+LT | `btn_10=23` | `btn_10=24` | `btn_10=25` | `btn_10=26` |
| LB+RB+LT | `btn_10=35` | `btn_10=36` | `btn_10=37` | `btn_10=38` |
| RT | `btn_10=6` | `btn_10=5` | `btn_10=7` | `btn_10=8` |
| LB+RT | `btn_10=19` | `btn_10=20` | `btn_10=21` | `btn_10=22` |
| RB+RT | `btn_10=27` | `btn_10=28` | `btn_10=29` | `btn_10=30` |
| LB+RB+RT | `btn_10=39` | `btn_10=40` | `btn_10=41` | `btn_10=42` |
| LT+RT | `btn_10=31` | `btn_10=32` | `btn_10=33` | `btn_10=34` |
| LB+LT+RT | `btn_10=43` | `btn_10=44` | `btn_10=45` | `btn_10=46` |
| RB+LT+RT | `btn_10=47` | `btn_10=48` | `btn_10=49` | `btn_10=50` |
| LB+RB+LT+RT | `btn_10=51` | `btn_10=52` | `btn_10=53` | `btn_10=54` |

预留物理组合不代表槽位永久空闲。每次仍要扫描所有启用 Mod 的 events，确认所选 `slot/value` 没有语义冲突。只有需要多个主按键同时按下、第五个辅助键、edge/system 输出或其他预设范围外输入时，才扩展 `xbox_default.yaml`。

## route 与 action 规划

用 route 表达状态切换：

- 基础状态进入功能；
- 功能状态之间的切换；
- 返回 normal；
- 动作完成后的目标；
- pd_brake/zero_torque 等安全退出；
- 传感器或外部参考失效后的退回。

用 action 表达不切换状态的操作：

- 暂停/继续；
- 重置对齐；
- 在同一状态内部切换手动/话题控制；
- 清除或重新捕获参考。

每个 action 都要在状态 `on_action()` 中有对应 handler，并在清单中提供 App label。不要为一个布尔开关额外制造两个几乎相同的状态；也不要把真实状态切换藏在 action 中而使状态图不可见。

## App 展示规划

App 中的状态入口来自 state manifest：

```yaml
states:
  feature:
    manifest:
      label: 功能名称
      priority: 820
      group: Advanced
      icon: extension
      confirm: true
      confirm_message: 请确认设备在线且机器人周围安全
```

规则：

- `label` 面向用户，不直接显示内部类名；
- 普通状态使用 `priority` 排序，相同 priority 再按完整名称排序；
- `group` 与现有 App 分组一致；
- state manifest 的 `icon` 和 action manifest 的 `ui` 都从 Google Fonts Icons（<https://fonts.google.com/icons>）选择语义匹配的 Material Symbols 图标，填写页面给出的准确图标名称；同时核对 App 使用的图标集/版本能够渲染该名称，不自行发明字符串；
- 高危、快速、需要外设或大动作范围的状态设置 confirm 和具体提示；
- 除非外部协议要求固定绝对位置，不设置 `manifest.index`。

App 中不切状态的操作来自 action manifest：

```yaml
actions:
  - from: feature
    event: toggle_pause
    action: toggle_pause
    manifest:
      label: 暂停/继续
      ui: pause
```

检查 `state_machine_info.graph.actions[]` 和 states 图快照能完整发布这些信息，而不是只确认代码中存在 manifest。

## 单 Mod 移植验收

完成后用“复制整个 Mod 目录到另一套兼容 ELF3 安装树”反向检查：

1. 状态、策略、模型、配置、nodes 和私有依赖是否都随目录移动；
2. 是否只依赖清单明确声明的其他 Mod、ROS 包和系统库；
3. 是否存在 source/install 绝对路径或机器专属路径；
4. 是否遗漏 remote controller 物理绑定、系统 profile 或平台 Defaults；
5. 禁用或删除该 Mod 后，功能是否整体消失且不残留后台进程和中央代码分支；
6. App 是否仍能仅根据清单发现状态和 action。

交付时把第 4 项的外部前置条件单独列出。能放回 Mod 的内容不要依赖口头部署步骤。
