# Asset provenance and release boundary

This file documents binary/data assets added by the ELF3 SONIC runtime branch.
It does not replace the repository-wide license, which must be selected and
approved by the repository owner.

## BXI action assets

The following company assets are retained on the development/runtime branch but
are intentionally excluded from the sanitized `main` release by
`release_protection.yaml`:


## ELF3 SONIC model

`src/bxi_example_py_elf3/data/sonic_model/elf3_step28800_smpl/model_step_028800_smpl.onnx`
is a BXI-trained derivative SONIC model. NVIDIA model-license terms and
attribution are included in `third_party/GEAR_SONIC_LICENSE.txt` and
`THIRD_PARTY_NOTICES.md`. BXI has confirmed that the model may be publicly
distributed and used commercially. Only the inference model is included; no
source dataset, raw training sample, dataset manifest, experiment log,
optimizer state or training checkpoint is distributed in this repository.

## Self-collected standing reference

`src/bxi_example_py_elf3/data/sonic_reference/elf3_pico_stand_clean_001/stream_reference.npz`
contains a ten-frame standing reference derived from an internally collected
PICO recording. Its embedded source identifier is
`self_collected://elf3_pico_stand_clean_001`; workstation paths and raw capture
files are not distributed. BXI has confirmed that this standing reference may
be publicly distributed and used commercially.

## Excluded runtime dependencies

XRoboToolkit bindings and RoboticsService binaries are external runtime
dependencies and are not distributed in this repository.
