# 按 Mod 生成公开发布树

`sanitize_release.py` 不再修改 Python class、状态机 YAML 或模型初始化代码。
它只复制仓库、删除 `mod.yaml` 中声明了 `visibility: protected` 的完整 Mod
目录，并验证所有保留 Mod 的依赖闭包仍然完整。

```bash
python3 tools/sanitize_release.py \
  --out /tmp/public_release \
  --self-check
```

临时额外排除某个公开 Mod：

```bash
python3 tools/sanitize_release.py \
  --exclude com.example.customer_feature \
  --out /tmp/public_release
```

共享推理器或模型应放在没有状态的资源 Mod 中。只要仍有公开 Mod 依赖该资源
Mod，它就会保留；如果公开 Mod 依赖了受保护资源 Mod，生成过程会直接失败。
