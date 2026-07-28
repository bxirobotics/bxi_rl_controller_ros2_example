# Inference benchmark

Run every model against every locally available backend:

```bash
python3 tools/benchmark/backend_benchmark.py
```

The command needs no `PYTHONPATH`. It discovers every `.onnx` under `src/`,
enumerates OpenVINO devices, tests OpenVINO `AUTO`, tests the framework's ONNX
Runtime selection, and includes RKNN when a matching model exists. A timestamped
JSON report is written under `tools/benchmark/results/` for cross-platform
comparison.
When accelerated ONNX Runtime providers are installed, CPU remains the numerical
reference and CUDA/TensorRT/other providers are benchmarked separately.
Each model/backend pair runs in an isolated process, so a native GPU driver crash
is reported without terminating the remaining benchmark cases. Non-Intel GPUs
accidentally enumerated by OpenVINO's Intel GPU plugin are skipped; use ONNX
Runtime CUDA/TensorRT for NVIDIA devices.

Useful variants:

```bash
# Fast deployment smoke test
python3 tools/benchmark/backend_benchmark.py --quick

# One model or an external model directory
python3 tools/benchmark/backend_benchmark.py path/to/model.onnx
python3 tools/benchmark/backend_benchmark.py /opt/models

# Longer, more stable statistics
python3 tools/benchmark/backend_benchmark.py --warmup 500 --iterations 10000

# Override an unresolved dynamic input dimension
python3 tools/benchmark/backend_benchmark.py --shape images=1,3,224,224

# Explicit report filename
python3 tools/benchmark/backend_benchmark.py --json results/my-platform.json
```

RKNN conversion remains opt-in. Converted models are stored in the ignored
benchmark cache instead of beside source assets:

```bash
BXI_RKNN_CONVERT_ON_LOAD=rk3588 \
python3 tools/benchmark/backend_benchmark.py --rknn-target rk3588
```

For quantized conversion:

```bash
BXI_RKNN_CONVERT_ON_LOAD='{"target":"rk3588","do_quantization":true,"dataset":"/data/calibration.txt"}' \
python3 tools/benchmark/backend_benchmark.py --rknn-target rk3588
```

Keep the machine idle, use the same power mode, and use the same benchmark
settings when comparing reports from different platforms. The first backend in
each model is the numerical reference; quantized RKNN outputs may legitimately
have larger differences.

## Compare with a Git baseline

The policy-level benchmark compares the current inference implementation with a
Git revision. It also measures Python allocations:

```bash
python3 tools/benchmark/inference_benchmark.py --baseline-ref HEAD
```

Use this benchmark when changing input construction, history buffers or policy
code. Use `backend_benchmark.py` when comparing runtimes and deployment devices.
