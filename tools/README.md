# Release Sanitizer

`sanitize_release.py` 用来从内部开发分支生成公开发布树。它不会修改当前仓库源码，而是复制一份目录到 `--out`，再按各 example 的 `release_protection.yaml` 删除受保护状态、模型、动作数据和只用于这些状态的遥控器入口。

典型用法：

```bash
python3 tools/sanitize_release.py \
  --manifest src/bxi_example_py_elf3/config/release_protection.yaml \
  --out dist/public_release \
  --self-check
```

不传 `--manifest` 时，默认读取：

```text
src/bxi_example_py_elf3/config/release_protection.yaml
```

多个 example 可以传多个清单：

```bash
python3 tools/sanitize_release.py \
  --manifest src/bxi_example_py_elf3/config/release_protection.yaml \
  --manifest src/another_example/config/release_protection.yaml \
  --out dist/public_release \
  --self-check
```

## 清单结构

示例：

```yaml
protected_states:
  back_flip:
    behavior:
      - FlipState
      - BackFlipState
    model_keys: [back_flip]
    files:
      - ../data/back_flip.npz
      - ../data/back_flip.onnx
```

`protected_states.<state>` 是受保护状态名。脚本会尝试从状态机配置里的 `states` 删除同名状态。

## 字段影响

### `behavior`

可以写字符串：

```yaml
behavior: BackFlipState
```

也可以写数组：

```yaml
behavior:
  - FlipState
  - BackFlipState
```

影响：

- 从 `robot_states.py` 删除这些 Python class。
- 加入 `--self-check` 检查 token。如果公开树里还残留这些类名，脚本失败。
- 只有当对应状态确实被删除时，这些 class 才会删除。

数组适合状态有基类或辅助类时一起删除。例如 `BackFlipState` 和 `ForwardFlipState` 共用 `FlipState`，可以把 `FlipState` 写进两个状态的 `behavior` 里。脚本会避免因为一个未删除状态仍引用某个 behavior 而误删它。

### `behaviors`

和 `behavior` 等价，只是复数字段。可以用来让配置语义更直观：

```yaml
behaviors:
  - FlipState
  - BackFlipState
```

### 状态机事件

不需要在 `release_protection.yaml` 里声明事件。脚本会从 `elf3_state_machine.yaml` 的 `states.*.transitions.on_event` 自动推导和被删除状态相关的 event。

影响状态机配置：

- 从 `remote_events` 删除这些 event。
- 删除公开状态里指向受保护状态的 `transitions.on_event`。
- 删除公开状态里指向受保护状态的 `transitions.after`。

影响遥控器配置：

- 脚本会从状态机的 `remote_events.<event>` 推导底层输出，例如 `btn_10=1`。
- 如果这个输出只服务于受保护 event，则从 `src/remote_controller/config/xbox_default.yaml` 的 `outputs.level` / `outputs.edge` 删除对应 binding。
- 删除 binding 后，如果相关 `controls` / `sources.*.signals` 没有被剩余公开 binding 引用，也会继续删除。

保留规则：

- 如果某个推导出的 event 仍被非保护状态用于非保护目标或 action，脚本不会删除这个 event，会输出 warning。
- 如果某个底层输出同时被公开 event 使用，脚本不会删除这个遥控器输出 binding，会输出 warning。
- warning 不会中断脚本。这是为了避免把公开功能删坏。

### `model_keys`

声明 `demo_node` 里要删除的模型成员名：

```yaml
model_keys: [back_flip]
```

影响：

- 从 `demo_node` 中删除 `self.<model_key> = ...` 形式的模型初始化代码块。
- 加入 `--self-check` 检查 token。
- 不会删除任何模型文件；要删除的 `.onnx` / `.npz` 必须写在 `files` 字段里。

只有当对应状态确实被删除时，`model_keys` 才会生效。

例如清单里写：

```yaml
model_keys: [back_flip, forward_flip]
```

脚本会在 `bxi_example_demo.py` 这类 demo node 文件里删除同名成员初始化：

```python
self.back_flip = DanceMotionPolicyGravityIsaaclab(
    self.data_file("back_flip.npz"),
    self.data_file("back_flip.onnx"),
    start_frame=40,
)
self.forward_flip = DanceMotionPolicyGravityIsaaclab(
    self.data_file("forward_flip.npz"),
    self.data_file("forward_flip.onnx"),
    start_frame=150,
)
```

匹配规则是 `self.<model_key> =`。如果初始化是多行调用，脚本会按括号深度一起删除整个赋值块。这个逻辑由 `paths.demo_node` 指向的文件决定；默认会解析为当前 example package 下的 `bxi_example_demo.py`。

### `files`

声明额外要删除的文件：

```yaml
files:
  - ../data/back_flip.npz
  - ../data/back_flip.onnx
```

影响：

- 直接从公开树删除这些文件。
- 相对路径按当前 `release_protection.yaml` 所在目录解析。
- 如果路径以 `src`、`tools`、`.github` 开头，则按仓库根目录解析。
- 只有当对应状态确实被删除时，`files` 才会生效。

## `paths`

一般不需要写。脚本会按清单所在 package 自动推导：

```text
config/elf3_state_machine.yaml
<package_python_module>/robot_states.py
<package_python_module>/bxi_example_demo.py
```

特殊 example 可以覆盖：

```yaml
paths:
  state_machine: elf3_state_machine.yaml
  robot_states: ../bxi_example_py_elf3/robot_states.py
  demo_node: ../bxi_example_py_elf3/bxi_example_demo.py
```

字段含义：

- `paths.state_machine`：要清理的状态机 YAML。
- `paths.robot_states`：包含状态 class 的 Python 文件。
- `paths.demo_node`：包含模型成员初始化的 Python 节点文件。

相对路径同样按清单所在目录解析。

## 固定删除项

公开树会固定删除这些 dev-only 文件：

```text
release_protection.yaml
tools/sanitize_release.py
.github/workflows/sync_public_main.yml
```

每个传入的 manifest 文件本身也会从公开树删除，例如：

```text
src/bxi_example_py_elf3/config/release_protection.yaml
```

公开树会保留 `.github/workflows/auto_release.yml`，这样同步到 `main` 后，`main` 分支自己的 workflow 可以独立运行。

## GitHub Action 流程

`.github/workflows/sync_public_main.yml` 不直接 push `main`。流程是：

```text
dev push
  -> 生成 dist/public_release
  -> 推送公开树到 public/main-sync
  -> 创建或更新 public/main-sync -> main 的 PR
  -> 自动以 merge commit 合并 PR
  -> main 分支自己的 auto_release.yml 独立运行
```

这样可以满足 `main` 保护分支要求，也不会强行覆盖 `main` 历史。

`PUBLIC_RELEASE_TOKEN` 需要作为 repository secret 添加。推荐 fine-grained PAT 权限：

- `Contents: Read and write`
- `Pull requests: Read and write`
- `Workflows: Read and write`

公开同步 commit 的 author 会使用触发 workflow 的 dev commit author；本次 push 事件中的提交作者会写入 `Co-authored-by`，用于尽量保留贡献者归属。由于公开树必须先删除受保护内容，不能把原始 dev commit 原封不动 merge 到 `main`。

如果 PR 不能按普通权限立刻合并，workflow 会使用 `gh pr merge --admin --merge` 重试。`PUBLIC_RELEASE_TOKEN` 对应账号必须有绕过保护规则的权限，否则这一步仍会失败。

## 自检

加 `--self-check` 后，脚本会扫描公开树。如果已经删除的状态名、类名、event 名、model key 仍然残留在文本文件里，就直接失败。

注意：如果某个 token 因为共享公开功能被保留，脚本会先输出 warning，并把这个 token 从 self-check 里排除，避免误报。
