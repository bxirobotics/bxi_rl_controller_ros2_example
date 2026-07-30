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
benchmark cache instead of beside source assets. An unquantized conversion can
still be requested directly:

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

### Capture and build a representative INT8 model

Do not calibrate a control policy with the benchmark's random tensors. Capture
the final, preprocessed tensors seen by the policy on the robot instead. The
depth policy has a non-blocking recorder for this purpose: the control thread
only takes an in-memory snapshot, while a background thread writes the `.npy`
files and ordered `dataset.txt`.

Build and source the package, then start the hardware launch with:

```bash
BXI_RKNN_CALIBRATION_DIR=/tmp/bxi_rknn_calibration \
BXI_RKNN_CALIBRATION_EVERY=5 \
BXI_RKNN_CALIBRATION_MAX=500 \
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

Enter depth walking and exercise representative standing, forward, turning,
obstacle and open-space situations. The default `origin_camera` mode writes:

```text
/tmp/bxi_rknn_calibration/dagger2/dataset.txt
```

Only the active policy is sampled. To capture the legacy `normal_depth` model,
change the Mod state's mode to `depth_walk`, rebuild/restart, and perform a
second run. Existing samples are resumed until `BXI_RKNN_CALIBRATION_MAX` is
reached. Use a new empty root when intentionally starting a new dataset.

Copy the calibration root to the x86_64 RKNN Toolkit2 machine. Validate both
datasets without converting:

```bash
python3 tools/benchmark/quantize_rknn.py \
  src/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/dagger2.onnx \
  src/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/normal_depth.onnx \
  --calibration-root /path/to/bxi_rknn_calibration \
  --validate-only
```

Then build fresh W8A8 artifacts and atomically install them beside the ONNX
files:

```bash
PYTHONNOUSERSITE=1 python3 tools/benchmark/quantize_rknn.py \
  src/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/dagger2.onnx \
  src/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/normal_depth.onnx \
  --calibration-root /path/to/bxi_rknn_calibration \
  --install
```

The tool validates input count/order, concrete shapes, `float32`, finite values
and a minimum of 100 samples before importing RKNN Toolkit. Conversion always
uses `actions`, `optimization_level=3`, representative calibration, and a fresh
cache fingerprint. `--algorithm mmse` or `--quantized-dtype w8a16` can be used
for an accuracy-oriented follow-up build.

The lower-level equivalent remains available when a manually maintained RKNN
dataset is needed:

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
