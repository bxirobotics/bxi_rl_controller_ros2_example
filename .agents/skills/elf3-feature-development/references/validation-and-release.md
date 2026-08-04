# 验证、调试与发布

## 目录

- [修改前检查](#修改前检查)
- [静态与离线加载](#静态与离线加载)
- [有针对性的测试](#有针对性的测试)
- [构建与安装树](#构建与安装树)
- [仿真验证](#仿真验证)
- [常见错误定位](#常见错误定位)
- [真机分阶段验证](#真机分阶段验证)
- [发布](#发布)
- [交付说明](#交付说明)

## 修改前检查

1. 查看工作树，保留用户已有改动，不改无关文件。
2. 找到最接近的内置 Mod，确认 API 与清单当前写法。
3. 记录外部契约：Mod id、状态/事件名、关节布局、话题、资源、故障退回。
4. 明确本次是创建 Mod、扩展框架，还是仅诊断；普通新动作不要顺手修改 framework runtime。

## 静态与离线加载

至少检查：

- 12 个描述头字段齐全，无未知顶层字段；
- `entrypoint`/factory 可导入，状态工厂键与 `states` 一致；
- dataclass 或 `StateBuildContext` 完全消费所有参数；
- 跨 Mod 引用有 `requires`，版本/API 范围正确；
- event 都被使用，同一状态同一事件没有 route/action 冲突；
- Transition capability、speed profile、node state reference 均存在；
- ResourceKey 全局命名，资产没有逃出 `assets/`；
- runtime requirements 与目标平台一致。

仓库当前可用的离线加载方式：

```python
from pathlib import Path
import yaml

from bxi_example_py_elf3.framework.runtime.mod_loader import load_mod_runtime

package = Path("src/bxi_example_py_elf3")
base = yaml.safe_load((package / "config/elf3_state_machine.yaml").read_text())
runtime = load_mod_runtime(base, built_in_root=package / "mods")
print([mod.id for mod in runtime.mods])
print(sorted(runtime.state_factories))
runtime.close()
```

这是仓库维护/测试入口，Mod 自身代码仍不得依赖 `framework.runtime`。

## 有针对性的测试

优先执行与改动最接近的已有测试。新增逻辑至少覆盖：

- 参数默认值、非法类型、未知字段和边界值；
- 状态生命周期顺序与资源准备失败；
- 乱序具名关节、29→31、31→29、缺 Defaults 和所有权冲突；
- `advance=False` 不改变时间/history/播放游标；
- callback 非法消息、首帧等待、stale、恢复与运行期超时；
- node 依赖顺序、不可用、异常退出、重启上限和 shutdown；
- 动作完成、安全退回和 Transition 中断。

关节映射/合成性能可运行：

```bash
python3 tools/benchmark/joint_mapping_benchmark.py
```

## 构建与安装树

```bash
colcon build --packages-select bxi_example_py_elf3 \
  --symlink-install --merge-install
source install/setup.bash
```

如果还修改遥控器，加入 `remote_controller`。检查安装结果确实包含清单和资产：

```bash
find install/share/bxi_example_py_elf3/mods \
  -name mod.yaml -o -path '*/assets/*'
```

不要因旧 install layout 报错就删除整个工作区；只处理明确属于相关包的构建缓存，并在任何破坏性清理前确认目标。

## 仿真验证

按顺序确认：

1. 启动日志显示目标 Mod `loaded`，不是 `unavailable/disabled`；
2. 状态图包含预期 state、route、action、node；
3. on-demand 模型首次请求时短暂 `preparing`，当前状态继续控制；
4. 进入帧、第一运行帧、动作完成帧和退出帧连续；
5. 所有实际可达的进入/退出路径都能运行；
6. 关节方向、限位、增益、速度和前馈力矩符合预期；
7. 断传感器、停外部参考、杀辅助节点后按设计拒绝进入或安全退出；
8. 状态快速切换和 Transition 中断不会遗留进程、订阅或准备副作用；
9. 控制周期预算内没有持续分配、阻塞 IO 或日志洪泛。

可查看：

```bash
ros2 topic echo /simulation/state_machine_info
ros2 topic echo /motion_commands
```

真机把前缀替换为 `/hardware`。状态信息中的 `preparing`、`mods[]` 和 `nodes[]` 可区分资源准备、依赖不可用和运行故障。

## 常见错误定位

| 现象 | 优先检查 |
| --- | --- |
| Mod 未发现 | 内置 mods/mod_paths、安装树、重复 id、12 个头字段 |
| factory/manifest mismatch | `states` 与 `state_factories` 本地键 |
| unknown params | 工厂漏消费或 dataclass 字段拼错 |
| 模型找不到 | 所属 Mod `assets/`、`context.asset()`、安装树 |
| 一直 preparing | Resource 加载异常或仍未 ready |
| 按键有消息不切状态 | event 槽位和值、当前源状态 route、目标 availability |
| 缺 Entry/Running provider | route 所选 Transition 与状态能力不匹配 |
| 切换突跳 | 进入帧、on_enter 后首帧、advance 配置、五字段连续性 |
| no explicit JointCommandDefaults | 状态输出少于 Robot Layout，平台未补齐固定关节 |
| ownership conflicts | Composer Layer 重叠且没有明确 override |
| node unavailable | runtime requirements/profile/目标平台文件 |
| node faulted | 运行异常、退出码、restart 和子进程日志 |
| root 运行后 `.pyc` Permission denied | 旧 root 所有缓存/文件属主；保持 `PYTHONDONTWRITEBYTECODE=1` |

## 真机分阶段验证

1. 先离线确认模型契约和命令范围；
2. 仿真覆盖所有切换和故障路径；
3. 真机断使能或上吊架验证名称、方向和遥测；
4. 使用低增益、低速度、零/保守前馈力矩；
5. 急停和底层看门狗始终可达；
6. 清空动作范围，逐渐扩大幅度和速度；
7. 最后验证外部输入、断流、节点崩溃和恢复。

不要把仿真通过等同于真机安全。高危动作设置 `confirm`/`confirm_message`，但 UI 确认不能替代限位、急停、底层力矩饱和或看门狗。

## 发布

- 公开发布前运行：

```bash
python3 tools/sanitize_release.py --out /tmp/public_release --self-check
```

- public Mod 不得依赖将被删除的 protected Mod。
- 非 symlink 发布安装树应包含完整 Mod 目录、资产和目标平台 vendor/runtime；部署时完整替换目录，不做覆盖合并。
- vendor bundle 附许可证、来源版本、CPU 架构和 Python ABI 说明。
- 不发布其他架构误选的原生库、机器绝对路径、凭据、日志、缓存或临时参数文件。

## 交付说明

最终报告应包含：

- 修改了哪些 Mod 契约和文件；
- 为什么选该状态基类、Resource 策略、node 生命周期和 Transition；
- 状态输出与 Robot Layout 如何适配；
- 实际执行的静态检查、测试、构建和仿真；
- 未执行的真机验证及剩余风险。
