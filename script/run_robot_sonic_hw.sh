#!/usr/bin/env bash
set -euo pipefail

source_setup() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "[robot-sonic-hw] ERROR: missing ${file}"
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

if [[ -z "${SONIC_PICO_PYTHON:-}" ]]; then
  for candidate in \
    /home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python \
    /home/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python \
    /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/.venv_teleop/bin/python \
    /opt/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python; do
    if [[ -x "${candidate}" ]]; then
      export SONIC_PICO_PYTHON="${candidate}"
      break
    fi
  done
fi

echo "[robot-sonic-hw] SONIC_PICO_PYTHON=${SONIC_PICO_PYTHON:-python default from launch}"
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py "$@"
