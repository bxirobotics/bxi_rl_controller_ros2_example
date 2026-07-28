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

## 查看进程和各核心 CPU 占用

先查找控制进程 PID，再按 1 秒周期查看：

```bash
pgrep -af bxi_example_py_elf3_demo
python3 tools/cpu_usage.py <PID>
```

同时展开最忙的 10 个线程：

```bash
python3 tools/cpu_usage.py <PID> --threads
```

`process` 以一个完整核心为 100%，`system_cores` 是整机各核心总负载。
`process_by_recent_cpu~` 将每个线程在采样周期内使用的 CPU 时间归到该线程
采样结束时最近运行的核心；线程可能迁核，因此这是按核分布估算值。脚本独立运行，
不会在控制进程中增加采样代码或线程。

## 推理 benchmark

自动发现全部模型并测试当前平台可用的 ONNX Runtime、OpenVINO 和 RKNN 后端：

```bash
python3 tools/benchmark/backend_benchmark.py
```

快速检查：

```bash
python3 tools/benchmark/backend_benchmark.py --quick
```

与 Git baseline 比较策略热路径和 Python 内存申请：

```bash
python3 tools/benchmark/inference_benchmark.py --baseline-ref HEAD
```

完整参数和跨平台测试说明见 [`tools/benchmark/README.md`](benchmark/README.md)。
benchmark 脚本纳入 Git；自动生成的 JSON 报告和 RKNN 转换缓存不会纳入 Git。
