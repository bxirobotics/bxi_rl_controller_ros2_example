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

For every backend after the first successful reference backend, the terminal and
JSON report include maximum and mean absolute error, RMSE, relative L2 error and
cosine similarity. Models with multiple comparable outputs additionally report
those metrics per output. `match` still uses `--rtol` and `--atol`; the numerical
metrics remain visible when that strict boolean check fails. The default
floating-point input is deterministic uniform data in `[-1, 1]`. Each model
derives its seed from the base `--seed` and repository-relative model path, so
selecting one model or changing discovery order does not change its input. This
remains a numerical smoke test; use representative policy inputs when making a
final deployment-accuracy decision.

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

# Override bounded generation ranges for named floating-point inputs
python3 tools/benchmark/backend_benchmark.py path/to/model.onnx \
  --input-range obs_history=-1,1 \
  --input-range depth_data=-0.5,0.5

# Explicit report filename
python3 tools/benchmark/backend_benchmark.py --json results/my-platform.json
```

The JSON report records the effective per-model seed and observed input minima
and maxima so generated-input performance reports can be reproduced. Real
policy input validation belongs to the live backend-comparison Mod, where the
policy's own observation preprocessing and history construction are active.

RKNN conversion remains opt-in. Converted models are stored in the ignored
benchmark cache instead of beside source assets. An unquantized conversion can
still be requested directly:

```bash
PYTHONNOUSERSITE=1 \
BXI_RKNN_CONVERT_ON_LOAD=rk3588 \
python3 tools/benchmark/backend_benchmark.py --rknn-target rk3588
```

When an existing RKNN cache has a `.rknn.build.json` sidecar, the benchmark uses
that cache's complete output contract automatically. For a new conversion or a
legacy cache without a sidecar, it converts only the learned policy output
named `actions` by default. Deterministic reference-trajectory tensors such as
`joint_pos` must be sampled from the policy's trajectory asset and composed
with `actions` outside the inference backend. Besides making every backend use
the same reference data, this avoids an RKNN Toolkit 2.3.2 optimizer bug in the
exported `Cast/Clip/Gather` lookup graph.

Only list additional outputs when they are genuine learned or recurrent model
outputs consumed by the policy. The tool derives the corresponding physical
input set from the ONNX graph:

```bash
BXI_RKNN_CONVERT_ON_LOAD='{"target":"rk3588","force_rebuild":true}' \
python3 tools/benchmark/backend_benchmark.py path/to/model.onnx \
  --rknn-target rk3588 \
  --rknn-output actions \
  --rknn-output recurrent_state
```

### Capture and build a representative INT8 model

Do not calibrate a control policy with the benchmark's random tensors. Capture
the final, preprocessed tensors seen by the policy on the robot instead.
`collect_calibration.py` launches the application with a tool-side proxy around
every inference backend opened through `InferenceRuntime`. It therefore works
for every framework policy and every registered backend without adding capture
branches or lifecycle code to production policies.

Build and source the package, then start the hardware launch with:

```bash
python3 tools/benchmark/collect_calibration.py \
  --output /tmp/bxi_rknn_calibration \
  --every 5 \
  --max-samples 500 \
  --skip-first 10 \
  -- ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

The proxy records the ordered input mapping passed to `backend.run()`, after all
policy preprocessing and history assembly. The initial calls are skipped so
model warmup does not contaminate the dataset. Each model receives its own
directory named after its source model. For example, default `origin_camera`
depth walking writes:

```text
/tmp/bxi_rknn_calibration/dagger2/dataset.txt
```

Only backends that actually execute are sampled. Exercise representative states
and commands for every model being calibrated. Existing samples are resumed
until `--max-samples` is reached; model hashes and input contracts prevent an
unrelated model from being appended to the same directory. Use a new empty root
when intentionally starting a new dataset.

This launcher affects only Python processes started by that command and leaves
the installed framework unchanged. Collection intentionally copies inputs on
the inference thread and must be used for dataset generation, not latency or
real-time performance measurements.

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

The cache mirrors repository-relative paths. Installation requires both the
RKNN file and its `.rknn.build.json` IO-contract sidecar, and copies both beside
the matching ONNX source. Existing adjacent files are atomically updated and
identical files are skipped. Preview the operation with `--dry-run`, or use
`--cache PATH` for a non-default cache directory. At runtime the framework
rejects a cache whose ONNX digest, input contract or output contract does not
match the production `ModelSpec`, then safely tries OpenVINO or ONNX Runtime.

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

## Framework cycle overhead

Measure the fixed cost of the production control-cycle boundaries without adding
profiling hooks to the framework:

```bash
source /opt/ros/humble/setup.bash
python3 tools/benchmark/framework_performance.py \
  --json tools/benchmark/results/framework-my-platform.json
```

The script creates a temporary API-3 Mod with one allocation-free hold state and
calls the same cycle boundary used by `RobotControlRuntime`. It reports platform
input snapshot, framework update, actuator publication, their accounted sum and
the complete wall-clock call. This deliberately measures the framework's fixed
overhead; model inference belongs in `inference_benchmark.py`, and hardware/ROS
message conversion remains visible in the running controller's periodic timing
report.

The report also compares sleeping-process CPU time with and without an idle
`SubprocessLogRouter`, making the background cost of process log collection
visible. Use identical affinity, governor, power mode and command arguments when
comparing machines or revisions.

## Live process scheduling inspection

Inspect the main controller, all descendants and every Linux thread without
changing or importing anything in the target process:

```bash
python3 tools/diagnostics/process_scheduling.py --pid PID

# Two samples are needed for CPU percentages.
python3 tools/diagnostics/process_scheduling.py \
  --pid PID --count 10 --interval 1 \
  --json tools/benchmark/results/scheduling-my-platform.json

# PID discovery is also available when the regular expression matches once.
python3 tools/diagnostics/process_scheduling.py \
  --match 'bxi_example_py_elf3.*mod_node_runner'
```

Linux affinity and scheduling policy are per-thread. The table therefore shows
each TID separately, including last CPU, effective affinity, `SCHED_*` policy,
reset-on-fork, RT priority, nice value, kernel priority, context switches and
CPU migrations. Process sections add PID/PPID/PGRP/session, cgroup CPU/cpuset
limits, OOM adjustment and realtime/nice resource limits. The first snapshot
also prints topology, frequency/governor information and the kernel RT runtime
limit for every CPU available to the observed tree.
