# Vendored GEAR-SONIC file inventory

This repository contains a 68-file runtime subset under
`src/bxi_example_py_elf3/bxi_example_py_elf3/sonic_pico/vendor/gear_sonic`.
The subset remains subject to the licenses and attributions in
`THIRD_PARTY_NOTICES.md` and `third_party/GEAR_SONIC_LICENSE.txt`.

## Files modified by BXI Robotics

- `data/robot_model/instantiation/__init__.py`
- `data/robot_model/instantiation/elf3.py`
- `data/robot_model/model_data/elf3/` (one URDF and 34 STL files)
- `data/robot_model/robot_model.py`
- `data/robot_model/supplemental_info/elf3/__init__.py`
- `data/robot_model/supplemental_info/elf3/elf3_supplemental_info.py`
- `isaac_utils/rotations.py`
- `scripts/pico_manager_thread_server.py`
- `trl/utils/torch_transform.py`
- `utils/teleop/calibration/__init__.py`
- `utils/teleop/calibration/elf3_key_frames.py`
- `utils/teleop/calibration/elf3_native_provider.py`
- `utils/teleop/zmq/zmq_poller.py`

## Files retained from the upstream runtime subset

- `__init__.py`
- `data/__init__.py`
- `data/human/human_joints_info.pkl`
- `data/robot_model/__init__.py`
- `data/robot_model/supplemental_info/__init__.py`
- `data/robot_model/supplemental_info/robot_supplemental_info.py`
- `isaac_utils/__init__.py`
- `isaac_utils/maths.py`
- `scripts/__init__.py`
- `trl/__init__.py`
- `trl/utils/__init__.py`
- `trl/utils/kornia_transform.py`
- `trl/utils/rotation_conversion.py`
- `utils/__init__.py`
- `utils/teleop/__init__.py`
- `utils/teleop/solver/__init__.py`
- `utils/teleop/solver/hand/__init__.py`
- `utils/teleop/solver/solver.py`
- `utils/teleop/vis/__init__.py`
- `utils/teleop/zmq/__init__.py`
- `utils/teleop/zmq/zmq_planner_sender.py`

The package-only `__init__.py` files above may be empty compatibility wrappers
added when the subset was packaged; they contain no upstream implementation
logic.
