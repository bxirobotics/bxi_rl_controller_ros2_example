#!/usr/bin/env bash
set -u

BLOCKERS=0

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
block() { echo "[BLOCK] $*"; BLOCKERS=$((BLOCKERS + 1)); }
section() { echo; echo "== $* =="; }

section "Host"
echo "hostname=$(hostname)"
echo "architecture=$(uname -m)"
echo "kernel=$(uname -r)"
if [[ -r /etc/os-release ]]; then
  sed -n -E 's/^(NAME|VERSION|VERSION_ID)=/\1=/p' /etc/os-release
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]]; then
    ok "Ubuntu 22.04 baseline"
  else
    block "offline payload is validated only for Ubuntu 22.04"
  fi
else
  block "missing /etc/os-release"
fi
if [[ "$(uname -m)" == "x86_64" ]]; then
  ok "x86_64 architecture"
else
  block "offline payload only supports x86_64"
fi

if python3 - <<'PY'
import sys
print("python=" + sys.version.replace("\n", " "))
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
then
  ok "Python 3.10"
else
  block "Python 3.10 is required by the cp310 wheels"
fi

section "Disk"
df -h /home /opt /tmp 2>/dev/null | awk 'NR == 1 || !seen[$1]++'
available_mb="$(df -Pm /tmp | awk 'NR == 2 {print $4}')"
if [[ "${available_mb}" =~ ^[0-9]+$ ]] && (( available_mb >= 1800 )); then
  ok "/tmp has at least 1800 MiB free"
else
  block "/tmp needs at least 1800 MiB free for bundle, build and backup"
fi

section "Required base installs"
for command in colcon fuser flock pgrep ss systemctl sha256sum tar; do
  command -v "${command}" >/dev/null 2>&1 && \
    ok "command ${command}" || block "required command is missing: ${command}"
done
for path in \
  /opt/ros/humble/setup.bash \
  /opt/bxi/bxi_ros2_pkg/setup.bash; do
  if [[ -r "${path}" ]]; then
    ok "${path}"
  else
    block "missing ${path}"
  fi
done

if hardware_prefix="$(bash -c '
  set +u
  source /opt/ros/humble/setup.bash
  source /opt/bxi/bxi_ros2_pkg/setup.bash
  ros2 pkg prefix hardware_elf3
' 2>/dev/null)"; then
  ok "ELF3 hardware package: $(printf '%s\n' "${hardware_prefix}" | head -n 1)"
else
  block "hardware_elf3 is not discoverable from the base ELF3 install"
fi

section "Current example install"
for path in \
  /opt/bxi/bxi_rl_controller_ros2_example/setup.bash \
  /opt/bxi/bxi_rl_controller_ros2_example/lib/python3.10/site-packages; do
  if [[ -e "${path}" ]]; then
    ok "${path}"
  else
    warn "missing ${path}; the deployment can create it"
  fi
done

section "Existing PICO runtime"
for path in \
  /home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python \
  /opt/apps/roboticsservice/RoboticsServiceProcess \
  /opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so; do
  if [[ -e "${path}" ]]; then
    ok "${path}"
  else
    warn "missing ${path}; the offline payload will install it"
  fi
done

section "Control service integration"
CONTROL_SERVICE="${SONIC_CONTROL_SERVICE:-ros_elf_launch.service}"
EXPECTED_SETUP="/opt/bxi/bxi_rl_controller_ros2_example/setup.bash"
if command -v systemctl >/dev/null 2>&1 && \
   systemctl cat "${CONTROL_SERVICE}" >/dev/null 2>&1; then
  service_exec="$(systemctl show "${CONTROL_SERVICE}" -p ExecStart --value 2>/dev/null || true)"
  service_env="$(systemctl show "${CONTROL_SERVICE}" -p Environment --value 2>/dev/null || true)"
  service_load="$(systemctl show "${CONTROL_SERVICE}" -p LoadState --value 2>/dev/null || true)"
  service_active="$(systemctl show "${CONTROL_SERVICE}" -p ActiveState --value 2>/dev/null || true)"
  service_domain="$(printf '%s\n' "${service_env}" | tr ' ' '\n' | sed -n -E 's/^"?ROS_DOMAIN_ID=([0-9]+)"?$/\1/p' | head -n 1)"
  echo "service=${CONTROL_SERVICE}"
  echo "LoadState=${service_load} ActiveState=${service_active}"
  echo "ExecStart=${service_exec}"
  echo "Environment=${service_env}"
  if [[ "${service_exec}" != *"ros2 launch remote_controller remote_controller.launch.py"* ]] || \
     [[ "${service_exec}" == *"robot_gateway"* ]]; then
    block "${CONTROL_SERVICE} is not a recognized remote-controller service"
  fi
  if [[ "${service_load}" != "loaded" ]]; then
    warn "${CONTROL_SERVICE} LoadState=${service_load:-unknown}"
  fi
  if [[ "${service_exec}" == *"${EXPECTED_SETUP}"* ]] && \
     [[ "${service_exec}" == *"ros2 launch remote_controller remote_controller.launch.py"* ]] && \
     [[ "${service_exec}" != *"bxi_rl_controller_ros2_example/install/setup.bash"* ]]; then
    ok "${CONTROL_SERVICE} starts the /opt example install"
  else
    warn "${CONTROL_SERVICE} does not yet start ${EXPECTED_SETUP}; configure the service after deployment"
  fi
  if [[ "${service_domain}" =~ ^[0-9]+$ ]] && (( service_domain <= 232 )); then
    ok "${CONTROL_SERVICE} has numeric ROS_DOMAIN_ID=${service_domain}; preserve this robot-specific value"
  else
    warn "${CONTROL_SERVICE} has no valid explicit ROS_DOMAIN_ID (expected 0..232)"
  fi
else
  warn "${CONTROL_SERVICE} is not installed; terminal deployment remains possible but App startup needs separate integration"
fi

section "Active control processes"
if processes="$(pgrep -af 'ros2 launch bxi_example_py_elf3|hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' || true)" && \
   [[ -n "${processes}" ]]; then
  echo "${processes}"
  block "stop active robot control/PICO processes before replacing /opt"
else
  ok "no active robot control/PICO processes"
fi

lock_owner=""
if [[ -L /tmp/bxi_example_hw.lock ]]; then
  block "/tmp/bxi_example_hw.lock is a symlink"
elif [[ "${EUID}" -eq 0 ]]; then
  lock_owner="$(fuser -v /tmp/bxi_example_hw.lock 2>&1 || true)"
elif sudo -n true >/dev/null 2>&1; then
  lock_owner="$(sudo -n fuser -v /tmp/bxi_example_hw.lock 2>&1 || true)"
else
  block "sudo credential is required to verify all hardware lock owners; run sudo -v"
  lock_owner="$(fuser -v /tmp/bxi_example_hw.lock 2>&1 || true)"
fi
if [[ -n "${lock_owner}" ]]; then
  echo "${lock_owner}"
  block "/tmp/bxi_example_hw.lock is still owned"
else
  ok "/tmp/bxi_example_hw.lock has no owner"
fi

section "Relevant ports"
if ports="$(ss -lntup 2>/dev/null | grep -E ':(8081|60061|5556|5557)\b' || true)" && \
   [[ -n "${ports}" ]]; then
  echo "${ports}"
else
  ok "8081/60061/5556/5557 are currently free"
fi

section "ROS environment hints"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
hostname -I 2>/dev/null || true

section "Result"
if (( BLOCKERS == 0 )); then
  ok "host is ready for the offline deployment"
else
  echo "[BLOCK] ${BLOCKERS} pre-deployment blocker(s)"
fi
exit "${BLOCKERS}"
