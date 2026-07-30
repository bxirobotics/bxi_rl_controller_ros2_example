#!/usr/bin/env bash
# T1: start ELF3 MuJoCo sim + BXI Python controller with SONIC state enabled.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BXI_RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export BXI_RUNTIME_ROOT
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-91}"

if [[ -x "${BXI_RUNTIME_ROOT}/.venv_teleop/bin/python" ]]; then
  export SONIC_PICO_PYTHON="${SONIC_PICO_PYTHON:-${BXI_RUNTIME_ROOT}/.venv_teleop/bin/python}"
fi

source /opt/ros/humble/setup.bash
if [[ -f /opt/bxi/bxi_ros2_pkg-main/setup.bash ]]; then
  source /opt/bxi/bxi_ros2_pkg-main/setup.bash
elif [[ -f /opt/bxi/bxi_ros2_pkg/setup.bash ]]; then
  source /opt/bxi/bxi_ros2_pkg/setup.bash
fi
source "${BXI_RUNTIME_ROOT}/install/setup.bash"
# Sim2Sim is a source-tree validation flow.  Put the current Python package
# ahead of a possibly stale install copy so local patches are what ROS runs.
export PYTHONPATH="${BXI_RUNTIME_ROOT}/src/bxi_example_py_elf3:${PYTHONPATH:-}"

# Make the standard entry point deterministic. Only the dedicated gripper
# entry point may publish the four extra joints and PICO trigger topics.
export BXI_SIM_GRIPPER_ENABLE=0
export PICO_ENABLE_ROS_BUTTONS=0

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/elf3_bxi_sonic_sim2sim_ros_log}"
if [[ "${BXI_SIM2SIM_KEEP_PYTHONNOUSERSITE:-0}" != "1" ]]; then
  unset PYTHONNOUSERSITE
fi

export BXI_SONIC_USE_SMPL_REF_ZMQ="${BXI_SONIC_USE_SMPL_REF_ZMQ:-1}"
export BXI_SONIC_SMPL_REF_ZMQ_HOST="${BXI_SONIC_SMPL_REF_ZMQ_HOST:-127.0.0.1}"
export BXI_SONIC_SMPL_REF_ZMQ_PORT="${BXI_SONIC_SMPL_REF_ZMQ_PORT:-5557}"
export BXI_SONIC_SMPL_REF_ZMQ_TOPIC="${BXI_SONIC_SMPL_REF_ZMQ_TOPIC:-smpl_ref}"
export BXI_SONIC_YAW_BIAS_RAD="${BXI_SONIC_YAW_BIAS_RAD:-1.57079632679}"
export BXI_SONIC_IDLE_FRAME_START="${BXI_SONIC_IDLE_FRAME_START:-3509}"
export BXI_SONIC_SOURCE_BLEND_SECONDS="${BXI_SONIC_SOURCE_BLEND_SECONDS:-0.4}"

STARTUP_RELEASE_DELAY="${STARTUP_RELEASE_DELAY:-1.0}"
STARTUP_RELEASE_ALLOWED_STATES="${STARTUP_RELEASE_ALLOWED_STATES:-normal}"

echo "[sonic-bxi-sim2sim] BXI runtime root: ${BXI_RUNTIME_ROOT}"
echo "[sonic-bxi-sim2sim] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[sonic-bxi-sim2sim] startup release allowed states: ${STARTUP_RELEASE_ALLOWED_STATES}"
echo "[sonic-bxi-sim2sim] controller flow: pd_brake -> normal -> sonic_teleop"

ros2 launch bxi_example_py_elf3 example_sonic_sim2sim.launch.py \
  startup_release_delay:="${STARTUP_RELEASE_DELAY}" \
  startup_release_allowed_states:="${STARTUP_RELEASE_ALLOWED_STATES}"
