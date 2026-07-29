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

The benchmark converts only the policy output named `actions` by default. This
also avoids an RKNN Toolkit 2.3.2 optimizer bug in ONNX models that expose
several differently shaped reference-trajectory outputs. Select outputs
explicitly when needed:

```bash
BXI_RKNN_CONVERT_ON_LOAD='{"target":"rk3588","outputs":["actions"],"force_rebuild":true}' \
python3 tools/benchmark/backend_benchmark.py --rknn-target rk3588
```

For quantized conversion:

```bash
BXI_RKNN_CONVERT_ON_LOAD='{"target":"rk3588","do_quantization":true,"dataset":"/data/calibration.txt"}' \
python3 tools/benchmark/backend_benchmark.py --rknn-target rk3588
```

Use `rknn-toolkit2` on x86_64 to convert models and
`rknn-toolkit-lite2` on RK3588 to execute and benchmark the generated models.
An x86 run that converts successfully and then reports `rknnlite is not
installed` has completed the conversion. Install every cached model beside its
corresponding ONNX model in one command:

```bash
python3 tools/benchmark/install_rknn_cache.py
```

The cache mirrors repository-relative paths, so only an RKNN file with a
matching ONNX source is copied. Existing adjacent RKNN files are atomically
updated and identical files are skipped. Preview the operation with
`--dry-run`, or use `--cache PATH` for a non-default cache directory. The
adjacent `.rknn` files can then be copied to the target board with the project
and will be selected before OpenVINO and ONNX Runtime.

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

Historical policy constructors are detected from the selected revision, so
pre-framework implementations that accepted only model paths can also be used
as baselines. A failed isolated worker prints its original traceback together
with the case and version instead of only reporting `CalledProcessError`.
The default suite covers both full-body and 15-joint `withoutarm` AMP models,
the legacy MuJoCo-order and Isaac-order motion policies, history motion, and
both cached and fresh depth input paths. Outputs are normalized by joint name
before all 29 joint positions are compared, so a layout-only change is allowed
but a semantic joint swap fails the benchmark.

Use this benchmark when changing input construction, history buffers or policy
code. Use `backend_benchmark.py` when comparing runtimes and deployment devices.

## Joint-layout hot path

Measure the complete-state-to-policy mapping and 29/31/N command resolution:

```bash
python3 tools/benchmark/joint_mapping_benchmark.py
```

It covers 31→29 observation selection, 29→31 commands with explicit defaults,
31→29 command projection, allocation-free multi-source command composition,
full-layout reordering and the exact-layout fast-path check. To keep a local
cross-platform report (the report directory remains ignored):

```bash
python3 tools/benchmark/joint_mapping_benchmark.py \
  --json tools/benchmark/results/joints-my-platform.json
```
