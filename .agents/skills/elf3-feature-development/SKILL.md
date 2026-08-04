---
name: elf3-feature-development
description: 为 bxi_example_py_elf3 规划、实现、迁移、调试或审查功能、状态、动作和控制模式。用户要求新增机器人行为、感知/遥操能力或状态交互时使用；默认先评估用一个额外的可移植 Mod 封装状态、btn 事件、物理按键绑定、action/route、App manifest、Resource、子节点和资产，并检查 Mod API 4.0、29/31/N 关节兼容性与发布安全。
---

# ELF3 功能、状态与动作开发

## 目标

依据当前框架源码、`src/bxi_example_py_elf3/mods` 和该代码仓库对应的 GitHub Wiki 完成功能开发。本地 `.wiki` 是对应 Wiki Git 仓库的检出目录，不是来源不明的外部文档。把 Mod 视为功能的默认部署边界，而不是开发目标本身：尽量让一个功能连同状态、交互、展示、依赖和资产一起被发现、校验、部署、移植和删除。

## 开始前

1. 确认仓库根目录，并读取当前目录附近的约束文件。
2. 检查 `src/bxi_example_py_elf3/mods` 中最接近需求的实现，但不要默认把新代码塞进该 Mod；先判断能否建立一个额外的功能 Mod。
3. 根据任务读取本 Skill 的参考资料：
   - 收到功能、状态或动作需求，规划封装边界、按键和 App 展示：先读 [feature-planning.md](references/feature-planning.md)。
   - 新建目录、清单、事件或路由：读 [manifest-and-layout.md](references/manifest-and-layout.md)。
   - 编写状态或处理 ROS callback：读 [state-and-lifecycle.md](references/state-and-lifecycle.md)。
   - 使用模型、资产、硬件对象或辅助进程：读 [models-resources-and-nodes.md](references/models-resources-and-nodes.md)。
   - 处理 29/31/N 关节、MIT 命令、混合切换或遥控输入：读 [joints-transitions-and-inputs.md](references/joints-transitions-and-inputs.md)。
   - 调试、真机验证、发布或代码审查：读 [validation-and-release.md](references/validation-and-release.md)。

## 功能请求的默认决策

1. 第一时间评估新建一个额外 Mod。一个状态、动作、控制模式或围绕同一用户目标协作的一组状态，默认作为一个功能 Mod。
2. 尽量在这一个 Mod 内收齐 `states/events/routes/actions/speed_profiles/nodes`、Resource、策略代码、运行依赖和资产。实现包含多个进程也不等于要拆成多个 Mod；把受管进程声明为同一 Mod 的 `nodes`。
3. 编码前先规划完整交互链：物理按键组合 → remote controller output → `btn_*` 槽位和值 → Mod event → route 或 action → 状态行为 → App manifest。
4. 只有需求属于框架通用能力、平台边界、全局输入 Driver、现有功能缺陷，或确实被多个独立 Mod 共享的基础资源时，才优先修改中央框架、既有基础 Mod 或拆分共享 Mod。写明不能单 Mod 完成的原因。
5. 不为减少文件数量把新业务堆进 `com.bxi.basic_actions`；也不为追求形式上的解耦把一个可独立移植的功能拆成需要同步安装的多个业务 Mod。

## API 不确定时去哪里查

不要凭名称猜 API，也不要先从网络上的通用示例照搬。框架代码是第一依据，按以下顺序查证：

1. 先在当前工作树中查框架实现。看 `src/bxi_example_py_elf3/bxi_example_py_elf3/framework/mod_api/__init__.py` 确认公共导出，再看符号所属实现，确认当前签名、类型、默认值、生命周期和异常语义：
   - 状态与上下文：`mod_api/state.py`、`states.py`、`context.py`；
   - `MotorFrame`：`mod_api/frame.py`；
   - Resource 与工厂参数：`mod_api/resource.py`、`mod.py`；
   - Mod Node：`mod_api/node.py`；
   - Composer：`mod_api/composition.py`；
   - Transition：`mod_api/transition.py`；
   - 具名关节类型由公共入口导出，真实定义位于 `framework/joints.py`。
2. 用 `rg` 搜索 `src/bxi_example_py_elf3/mods` 中的实际调用，并搜索仓库测试和加载校验。优先参考与需求最接近、当前仍启用的生产 Mod，核对 API 在正确生命周期中的使用方式和失败行为。
3. 查询当前代码仓库对应的 Wiki Git 仓库。先用 `git remote -v` 和 `git branch -vv` 确定当前代码来自哪个远端；不要假定远端一定名为 `origin`。若代码远端是 `git@github.com:konodoki/bxi_rl_controller_ros2_example.git`，对应 Wiki 就是 `git@github.com:konodoki/bxi_rl_controller_ros2_example.wiki.git`；HTTPS 地址同样在仓库名后添加 `.wiki`。本地 `.wiki` 应是该 Wiki 仓库的独立 Git 检出目录，先核对它的 remote 是否与当前代码远端的 owner/repository 匹配。
4. 在匹配且足够新的 Wiki 中先查 `Mod-Public-API.md`，再按主题查 `Custom-State.md`、`Mod-State-Development-Guide.md`、`YAML-Reference.md`、`Mod-Nodes.md`、`Joint-Layout-And-Mapping.md`、`Joint-Command-Composition.md` 或 `Custom-Transition.md`。需要最新文档时检查 Wiki remote 与提交时间；若要更新本地检出，先确认没有未提交修改，不能用拉取覆盖用户的 Wiki 工作树。
5. 为理解未公开行为、验证 Wiki 描述或定位框架缺陷，可以继续阅读当前工作树的 `framework.runtime` 和加载器实现；但客户 Mod 代码仍不得导入它。若公共 API 无法完成合理需求，先报告缺少的扩展点，再决定是否修改框架。
6. 若源码、生产 Mod、测试和 Wiki 不一致，以当前检出的框架代码及实际加载行为为准，同时明确指出 Wiki 所属远端、提交新旧和文档漂移，不静默选择一种解释。
7. ROS 2、消息包、厂商 SDK 或第三方库不是本仓库定义的 API 时，确认项目实际使用的版本，再查对应版本的官方文档和本机接口；不要依据其他发行版或最新版示例猜测。

常用搜索方式：

```bash
rg -n "SymbolName|def method_name|class ClassName" \
  src/bxi_example_py_elf3/bxi_example_py_elf3/framework/mod_api \
  src/bxi_example_py_elf3/mods .wiki

rg --files | rg '(^|/)(test|tests)/|test_.*\.py$'

git remote -v
git branch -vv
git -C .wiki remote -v
git -C .wiki log -1 --format='%H %cI %s'
```

查不到时明确说明缺少哪一层证据，不虚构方法、字段、默认值或生命周期保证。

## 工作流

### 1. 先固定功能契约

在写实现前确定并尽量保持稳定：

- 单 Mod 边界、Mod `id`、`version`、API 范围和依赖；
- 本地状态、事件、action、route、speed profile 和 node 名称；
- 进入功能、退出功能、紧急退出和不切状态操作分别使用哪个 event/route/action；
- 每个 event 使用的 `btn_*` 槽位和值、对应物理组合键，以及键盘/Xbox/CRSF 等实际部署输入是否需要绑定；
- App 中 state 的 `label/priority/group/icon/confirm/confirm_message` 和 action 的 `label/ui`；
- 模型 observation/action 的真实具名关节布局；
- 状态自然输出布局，以及机器人缺失/新增关节的处理方式；
- 外部话题、消息类型、QoS、超时与失效行为；
- 模型、文件、SDK 或硬件对象的拥有者和加载策略；
- 进入、退出、动作完成、断流和故障时的目标状态。

跨 Mod 引用必须写完整名称并声明 `requires`。不要用数字关节下标、模型数组长度或机器人消息顺序充当跨组件契约。

### 2. 选择单 Mod 内的最小合适实现

按需求选择一种主要状态方案：

| 需求 | 首选 |
| --- | --- |
| 固定姿态 | `PoseState` |
| 公式轨迹、插值、周期动作 | `ProceduralState` |
| 在线推理、history 策略 | `PolicyState` |
| 固定模型动作回放 | `MotionReplayState` |
| 特殊传感器、时序或完整生命周期 | 直接继承 `RobotControlState` |

简单状态使用 `entrypoint: null` 和 `factory: module:Class`。只有需要 Resource、显式依赖注入或自定义 Transition 时才增加 `plugin.py` 与 `entrypoint: plugin:create_mod`。

相机发布、感知、通信桥或 SDK 服务等独立能力优先放进 `mod.yaml` 的 `nodes`；只有明确属于状态对象的轻量订阅、service 或 timer 才放进状态生命周期。

一个功能包含多个紧密协作状态时，把它们都注册在同一个 Mod。复用已有基础状态时通过 `requires` 和完整名称连接，不复制基础状态实现。

### 3. 实现时守住实时边界

- Mod 状态只依赖 `bxi_example_py_elf3.framework.mod_api`；策略实现还可依赖公开的 `framework.inference`。不要依赖 `framework.runtime`。
- `on_bind()` 时首个机器人状态可能尚未到达；不要读取 `ctx.robot_layout`。
- 在 `on_prepare()` 按名称编译关节映射，创建 Composer 和大缓冲，并预热已就绪模型。
- 控制热路径复用 `float32` 数组和 `MotorFrame`，不反复查名字、分配数组、加载文件或输出日志。
- `on_bind()` 创建的 ROS 实体必须在 `on_unbind()` 销毁；Resource 的昂贵对象由资源管理器统一关闭。
- 异步 callback 只更新带锁或双缓冲的最新快照；`on_update()` 读取一致快照并产生唯一电机输出。
- `is_available()` 必须非阻塞、无副作用，只做进入前健康检查。
- 数据超时、依赖断流、模型失败和动作完成都要有明确行为，不能默默沿用不可验证的旧数据。

### 4. 正确形成电机命令

`MotorFrame` 始终包含 MIT 五字段：`qpos/vel/kp/kd/torque`。只传 `qpos/kp/kd` 时，`vel` 和 `torque` 会明确置零；不要依赖上一帧残留。

每个状态按自己的自然具名布局输出。不要假定旧策略控制“机器人前 29 个关节”。状态少于 Robot Layout 时，缺少的硬件关节必须由平台 `JointCommandDefaults` 显式补齐；状态多于 Robot Layout 时，框架只裁剪机器人不存在的输出。

模型、话题、IK、轨迹等多个来源共同控制时，使用 `JointTargetBuffer` 和 `JointCommandComposer`。重叠关节必须由后置 Layer 显式声明 `override=True`。

### 5. 设计切换和输入

- 先扫描所有已启用 Mod 的 `events` 和 `src/remote_controller/config/xbox_default.yaml` 的 `outputs`，再选择未冲突的槽位、值和物理组合键。
- route 负责切换状态；action 负责不切换状态的当前状态操作。
- 进入/退出 route、normal 返回、zero_torque 等安全出口和动作完成后的去向都要显式规划。
- 在 Mod 中声明抽象 event，在 remote controller 配置中完成物理按键绑定；状态类不直接识别某款手柄按键。
- 为 state 和 action 填写 App manifest。普通状态使用 `priority`，不要为了占位置滥用固定 `index`。
- 根据输出连续性选择 `instant`、`hold`、`entry_gain_ramp`、`running_blend` 或 `sequence`。
- 需要进入帧时实现 `EntryFrameProvider`；需要动态混合时实现 `RunningFrameProvider`。
- `sample_running_frame(..., advance=False)` 只能观察，不能推进时间、history、相位或消费游标。
- Transition 插值前必须把两端自然帧解析到完整 Robot Layout；混合全部五个 MIT 字段。
- 状态读取速度时使用 `self.get_cmd_vel(ctx)` 和清单 speed profile，不直接耦合手柄设备。

### 6. 分层验证

按以下顺序验证，每层失败先修复再继续：

1. 单 Mod 包含完整功能文件，清单字段、命名、依赖、工厂、参数和状态图可离线加载；
2. startup/on-demand Resource 行为、节点依赖和关闭顺序正确；
3. 每个 event 的槽位/值无冲突，物理组合键能生成预期 output，App 能展示 state 和 action；
4. 关节布局、第一帧、运行帧、退出帧和 `advance=False` 正确；
5. 异步输入的新鲜度、断流和安全退回正确；
6. 构建并在仿真中完成所有进入/退出路径；
7. 最后在急停可达、低增益和清空场地条件下分阶段验证真机；
8. 发布前检查 protected 依赖、运行生成文件、完整安装树和单 Mod 移植完整性。

## 审查原则

审查现有 Mod 时先报告会造成危险输出、加载失败、线程竞态或部署不一致的问题，再报告可维护性问题。特别检查：

- 一个用户功能是否被无理由分散到多个业务 Mod 或中央状态类；
- btn 槽位/值、物理绑定、route/action 和 App manifest 是否形成完整可用链路；
- 关节顺序是否来自模型真实契约；
- MIT 五字段和增益是否在所有进入路径上定义；
- 状态工厂是否完全消费 YAML 参数；
- Resource 是否在构造器或控制周期被错误加载；
- callback 是否和控制线程并发改同一数组；
- Transition capability 与 `advance` 语义是否匹配；
- 外部依赖、节点生命周期、超时、故障退回和真机保护是否明确。

## 交付

说明功能为何采用新 Mod、为何能或不能保持单 Mod、btn 槽位与物理绑定、route/action、App 展示、状态/资源/节点方案、关节与 Transition 设计、执行过的验证，以及尚未在真机证明的风险。不要把未执行的检查写成已通过。
