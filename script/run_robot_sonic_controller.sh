#!/usr/bin/env bash
set -euo pipefail

source_setup() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "[robot-sonic-controller] ERROR: missing ${file}"
    exit 2
  fi
  set +u
  # shellcheck disable=SC1090
  source "${file}"
  set -u
}

source_setup /opt/ros/humble/setup.bash
source_setup /opt/bxi/bxi_ros2_pkg/setup.bash
source_setup /opt/bxi/bxi_rl_controller_ros2_example/setup.bash

echo "[robot-sonic-controller] keyboard: ! -> pd_brake, 1 -> normal, 6 -> sonic_teleop, g -> sonic_teleop_gripper"
echo "[robot-sonic-controller] PICO after sonic: ABXY -> calibrate/start, A+X -> POSE/live"

ros2 launch remote_controller remote_controller_keyboard.launch.py "$@"
