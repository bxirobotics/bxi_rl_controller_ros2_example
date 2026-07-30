#!/usr/bin/env bash
# T2: start the official BXI remote_controller for ELF3 SONIC sim2sim.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BXI_RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-91}"

source /opt/ros/humble/setup.bash
if [[ -f /opt/bxi/bxi_ros2_pkg-main/setup.bash ]]; then
  source /opt/bxi/bxi_ros2_pkg-main/setup.bash
elif [[ -f /opt/bxi/bxi_ros2_pkg/setup.bash ]]; then
  source /opt/bxi/bxi_ros2_pkg/setup.bash
fi
source "${BXI_RUNTIME_ROOT}/install/setup.bash"

DEFAULT_REMOTE_CONFIG="${BXI_RUNTIME_ROOT}/install/share/remote_controller/config/xbox_default.yaml"
if [[ ! -f "${DEFAULT_REMOTE_CONFIG}" ]]; then
  DEFAULT_REMOTE_CONFIG="${BXI_RUNTIME_ROOT}/src/remote_controller/config/xbox_default.yaml"
fi

REMOTE_CONFIG="${REMOTE_CONFIG:-${DEFAULT_REMOTE_CONFIG}}"
REMOTE_KEYBOARD="${REMOTE_KEYBOARD:-1}"

echo "[sonic-sim2sim-controller] config=${REMOTE_CONFIG}"
echo "[sonic-sim2sim-controller] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
if [[ "${REMOTE_KEYBOARD}" == "1" || "${REMOTE_KEYBOARD}" == "true" ]]; then
  echo "[sonic-sim2sim-controller] keyboard: ! -> pd_brake, 1 -> normal, 6 -> sonic_teleop, g -> sonic_teleop_gripper, 3 -> dance"
  ros2 run remote_controller remote_controller \
    --keyboard \
    --config "${REMOTE_CONFIG}" \
    --hot-reload true
else
  echo "[sonic-sim2sim-controller] gamepad: RB+B -> pd_brake, RB+X -> normal, RT+X -> sonic_teleop, LB+X -> dance (gripper mode is App/keyboard only)"
  ros2 run remote_controller remote_controller \
    --config "${REMOTE_CONFIG}" \
    --hot-reload true
fi
