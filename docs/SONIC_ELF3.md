# SONIC teleoperation on ELF3

## Scope

This branch adds SONIC as an ELF3 state-machine action. The existing official
unchanged. The SONIC controller still publishes the official 29-joint
`ActuatorCmds` interface on hardware and in the standard Sim2Sim entry point.
The dedicated gripper Sim2Sim entry point appends four simulation-only finger
joints for a 33-actuator command; it never changes the hardware body contract.

The source branch is not the internal offline deployment archive. It excludes
XRT binaries, offline wheels, build/install/log output, internal reports and
vendor SDK binaries. It does include the ELF3/native-calibration and optional
Sim gripper geometry authorized by BXI Robotics and documented in
`THIRD_PARTY_NOTICES.md`.
`bxi_example_bms` is optional battery telemetry and is not required by the
controller, Sim2Sim or the PICO runtime.

## Dependencies

Build the official workspace dependencies first, then install:

```bash
python3 -m pip install -r script/sonic_runtime_requirements.txt
colcon build --packages-select bxi_example_py_elf3 remote_controller
```

Real PICO operation also requires the robot vendor's
`xrobotoolkit_sdk` binding and `/opt/apps/roboticsservice/RoboticsServiceProcess`.
Those files are deliberately not stored in GitHub. PyVista/VTK and robot CAD
visualization are not needed by the headless PICO path; `PICO_ENABLE_VIS=1` is
not supported by this source-only dependency subset. The supervisor normally
uses its own Python interpreter; set `SONIC_PICO_PYTHON` to the deployment
virtualenv interpreter when the PICO dependencies live in a separate venv. The
`xrobotoolkit_sdk` extension also needs RoboticsService native libraries, most
importantly `/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so`. The
supervisor prepends the standard RoboticsService library directories to
`LD_LIBRARY_PATH` for the PICO manager/bridge children before they import the
SDK. The provided Sim2Sim script detects `<repo>/.venv_teleop/bin/python`
automatically.

The main `bxi_example_py_elf3_demo` controller is a separate ROS console script
and runs in the installed ROS Python environment, not in the PICO venv. Make
sure that environment can import `numpy`, `onnxruntime`, `zmq` and
`bxi_example_py_elf3.inference.sonic`; otherwise the controller can exit before
publishing motor commands even when the PICO venv check passes.

On the robot, the hardware launch also auto-detects the known deployment venv
locations, including:

```text
/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python
/home/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python
/home/bxi/bxi_ws/bxi_rl_controller_ros2_example/.venv_teleop/bin/python
/opt/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python
```

You can still override it explicitly with either `SONIC_PICO_PYTHON` or the
launch argument `sonic_pico_python:=...`.

The robot-side PICO manager defaults to CPU. Set `SONIC_PICO_USE_CUDA=1` only
on machines that actually have a usable CUDA device; ELF3 onboard computers are
expected to run this path without CUDA.

## Robot deployment

The tablet app should not be modified for this branch. Deploy the compiled ROS
packages by overwriting the existing example install under `/opt/bxi`, after
taking a backup.

For robots with slow or unavailable GitHub access, use the commit-pinned
offline bundle workflow in `docs/SONIC_ELF3_OFFLINE_DEPLOY.md`. It installs the
PICO venv and the separate controller Python dependencies as well as the ROS
packages, and verifies every payload file before changing `/opt`.

On each robot:

```bash
cd ~

if [ ! -d "$HOME/bxi_rl_controller_ros2_example/.git" ]; then
  git clone -b feature/sonic-elf3-runtime-gripper-clean --single-branch \
    https://github.com/bxirobotics/bxi_rl_controller_ros2_example.git \
    "$HOME/bxi_rl_controller_ros2_example"
fi

export REPO_URL=https://github.com/bxirobotics/bxi_rl_controller_ros2_example.git
export BRANCH=feature/sonic-elf3-runtime-gripper-clean
bash "$HOME/bxi_rl_controller_ros2_example/script/deploy_robot_sonic_example.sh"
```

Then verify the install and PICO dependencies:

```bash
bash "$HOME/bxi_rl_controller_ros2_example/script/check_robot_sonic_runtime.sh"
```

For a direct terminal test on the robot, use two root terminals:

```bash
# T1
bash "$HOME/bxi_rl_controller_ros2_example/script/run_robot_sonic_hw.sh"

# T2
bash "$HOME/bxi_rl_controller_ros2_example/script/run_robot_sonic_controller.sh"
```

The complete pre-tablet terminal acceptance and cleanup procedure is in
`docs/SONIC_ELF3_TERMINAL_ACCEPTANCE.md`.

Keyboard state flow is `!` (PD brake), `1` (normal), then `6` (SONIC). After
SONIC starts and PICO body data is available, press `ABXY` to calibrate/start
and `A+X` to switch to POSE/live.

## Sim2Sim

Build once from the repository root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select bxi_example_py_elf3 remote_controller
```

Then open two terminals. The launch file now owns a lightweight PICO runtime
supervisor, so a separate T3 terminal is not needed in the normal flow.

T1 — MuJoCo and the BXI controller:

```bash
# Standard 29-actuator body model
bash script/run_sonic_bxi_sim2sim.sh

# Or the opt-in 33-actuator body + gripper model
bash script/run_sonic_bxi_sim2sim_gripper.sh
```

The gripper entry point sets `BXI_SIM_GRIPPER_ENABLE=1`, enables the PICO
trigger topics and selects `elf3_gripper.xml`. The standard entry point remains
body-only. Hardware topics never append these four simulation joints.

T2 — keyboard/remote controller:

```bash
bash script/run_sonic_sim2sim_controller.sh
```

Keyboard state flow is `!` (PD brake), `1` (normal), then `6` (body-only
SONIC). Key `g` selects the explicit hardware-gripper SONIC state; in Sim2Sim,
the selected MuJoCo launch script still determines whether the four simulated
finger actuators exist.
Back-flip moves from keyboard key `6` to `0`; its gamepad mapping is unchanged.
The SONIC gamepad mapping is `RT + X`.

When the state-machine transition targets `sonic_teleop` or
`sonic_teleop_gripper`, or either is already current, the supervisor starts
exactly one PICO manager and one PICO-to-SMPL bridge. Leaving both SONIC states
stops the child process groups. A missing or malformed state-machine heartbeat
also fails closed and stops them. Shutdown escalates from SIGINT to SIGTERM and
SIGKILL if needed. The manager uses port 5556 for PICO pose, the bridge uses
port 5557 for `smpl_ref`, and the external XRT service normally uses port
60061.

Do not run `script/run_sonic_pico_sources.sh` alongside the automatic
supervisor because both instances would contend for the same ports. That script
is retained for diagnostics; disable launch auto-start with
`sonic_pico_auto_start:=false` before using it manually.

Only the documented `ABXY -> PLANNER -> A+X -> POSE` path is part of the ELF3
SONIC acceptance flow. The inherited frozen-upper-body/VR3PT planner feedback
path has no ELF3 feedback publisher/schema and is not a supported operating
mode in this branch.

### Sim2Sim verification

1. Start T1 first. While the state is still `zero_torque`, the suspended robot
   receives no controller motor command during reset step 1. Start T2 and select
   `!` and `1`; state events are accepted while suspended, but actuator targets
   remain blocked. The simulator releases only after reaching `normal`. Select
   `6` after release to enter SONIC.
2. Confirm the T1 log reports that the SONIC PICO manager and bridge are being
   started. With no PICO body data, SONIC must continue ONNX inference from the
   fixed `idle_left` window and report `idle_reference`; it must not wait in a
   non-policy default pose.
3. For a live-input test, connect XRobotToolkit and wait for fresh body data.
   Hold the intended calibration pose and press all four PICO buttons
   (`A+B+X+Y`, abbreviated `ABXY`) once. The manager must only leave `OFF` after
   it has a body sample and the three-point calibration handshake succeeds.
4. Press `A+X` once to change `PLANNER` to `POSE`. The bridge requires three
   consecutive finite POSE messages with `calibration_ready=true` and a
   strictly advancing frame index. After its initial window fills (normally
   about 0.2 s at 50 Hz), SONIC reports `live_reference` and follows PICO.
5. Stop the PICO data stream, or switch the manager out of POSE. Live
   publication stops after the freshness timeout (0.2 s by default), and SONIC
   blends back to the fixed idle reference instead of replaying stale motion.
   The transient status is `live_stale_to_idle`, followed by `idle_reference`.
6. Switch from SONIC back to `normal`. The manager and bridge must stop while
   the launch-owned supervisor remains available for the next SONIC entry.

To inspect the state seen by the supervisor:

```bash
ros2 topic echo --once /simulation/state_machine_info std_msgs/msg/String
```

After leaving SONIC, check that no manager/bridge child or listening socket was
left behind:

```bash
pgrep -af 'pico_manager_legacy|pico_manager_thread_server|pico_pose_to_smpl_ref_bridge'
ss -ltnp | grep -E ':(5556|5557|60061)\b'
```

Both commands should print nothing created by this run. If
`RoboticsServiceProcess` or port 60061 was already managed externally before
the test, compare with the pre-test baseline instead. Finally press Ctrl+C in
T1 and confirm that `sonic_pico_runtime_supervisor` is also gone:

```bash
pgrep -af 'sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge'
```

## Runtime policy

- Entering SONIC automatically requests the PICO runtime. Before the source is
  ready, SONIC keeps running the policy against a fixed, calm ten-frame window
  from the clean `idle_left` reference; the reference cursor does not advance.
- Live `smpl_ref` is accepted only when the bridge marks it
  `source_ready=true`. That requires a successful ABXY readiness handshake,
  POSE mode, progressing frame indices, finite tensors and consecutive fresh
  messages. Idle-to-live and live-to-idle target changes are blended.
- When live input becomes stale, SONIC discards the old live packet and returns
  to `idle_left`. Re-entering or resetting SONIC cannot reuse a packet from the
  previous session.
- SONIC intentionally does not use the common approximately 60-degree
  roll/pitch transition to `zero_torque`; other states retain that protection.
- The App exposes `SONIC遥操` (body-only, `btn_10=7`) and the explicitly
  confirmed `SONIC遥操（夹爪）` (`btn_10=8`). Both use the same 29-DoF body
  policy. Only the second state can publish the independent hardware-gripper
  CAN commands; process restart and the normal SONIC state remain body-only.
- Hardware-gripper entry waits for fresh left/right PICO trigger samples with
  both triggers released. It then sends `enter_motor_mode` once per side.
  Left/right trigger values proportionally control the corresponding gripper.
  A short trigger-stream interruption retains and republishes the last valid
  targets, reports one stale edge and one recovery edge, and never converts the
  missing input to zero/open.
- Default hardware parameters remain left bus `5`, right bus `6`, CAN ID `1`,
  `kp=20`, and `kd=1`. Confirm these values, direction and travel on the target
  robot before loaded operation.
- The PICO manager uses only the ELF3-native 31-DoF FK model while preserving
  the controller's 29-DoF body-vector contract. No alternate robot profile or
  alternate robot URDF is packaged.

### Calibration boundary

`CALIB_FULL`/ABXY currently calibrates the manager's three-point VR tracking
and also serves as an explicit operator-readiness handshake. It does **not**
numerically calibrate or remap the raw SMPL tensors consumed by SONIC in POSE
mode (`smpl_joints`, `body_quat_w`, and `joint_pos`). Therefore
`calibration_ready=true` means that the three-point calibration step succeeded;
it must not be interpreted as proof that the full SONIC body-reference tensor
has been calibrated to ELF3.

## Model and references

Default files are installed below the ROS package share directory:

```text
data/sonic_model/elf3_step28800_smpl/model_step_028800_smpl.onnx
data/sonic_reference/elf3_pico_stand_clean_001/stream_reference.npz
```

They can be overridden with `BXI_SONIC_MODEL_ONNX` and
`BXI_SONIC_STREAM_REFERENCE_NPZ`. See `THIRD_PARTY_NOTICES.md` for model
license, attribution, cleanup provenance and SHA256. The installed idle file is
also the fallback source used before PICO readiness and after a live-stream
timeout. The default calm window starts at frame 3509; use
`BXI_SONIC_IDLE_FRAME_START` to change it after a Sim2Sim A/B. Source changes
use a 0.4 second smoothstep blend by default, configurable with
`BXI_SONIC_SOURCE_BLEND_SECONDS`.

## Validation status

The earlier local Sim2Sim chain, clean source build/install, cleaned ONNX
inference-equivalence check, and manual T3 SIGINT cleanup while waiting for body
data passed. The automatic state-driven lifecycle and idle/live readiness path
must pass the verification above before real-robot deployment. Real-robot
cleanup during normal POSE still requires validation before declaring the
true-hardware deployment closed.
