#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT=""
SERVICE="${SONIC_CONTROL_SERVICE:-ros_elf_launch.service}"
GATEWAY_SERVICE="${SONIC_GATEWAY_SERVICE:-bxi_rc_ros2.service}"
ASSUME_YES=0
TERMINAL_ONLY=0
SERVICE_STOPPED=0

failure_context() {
  if (( SERVICE_STOPPED == 1 )); then
    echo "[one-click] SAFE STOP: ${SERVICE} remains stopped; no automatic restart was attempted" >&2
    latest_backup="$(ls -t /opt/bxi/deploy_backups/bxi_rl_controller_ros2_example.before_sonic_*.tgz 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_backup}" ]]; then
      echo "[one-click] latest available /opt backup: ${latest_backup}" >&2
    fi
  fi
}

usage() {
  cat <<'EOF'
usage: install_robot_sonic_bundle.sh [--yes] [--terminal-only] [--service NAME] /path/to/extracted-bundle

Safe one-command installer for the validated ELF3 Ubuntu 22.04 baseline.
It verifies the bundle, asks for an explicit host confirmation, stops only the
robot control service, refuses remaining control processes or dpkg locks, runs
the offline deployment, and points the existing service at /opt while preserving
its ROS/RMW Environment.  It never starts the service or robot hardware.
EOF
}

die() {
  echo "[one-click] ERROR: $*" >&2
  failure_context
  exit 2
}

on_error() {
  local status="$?"
  trap - ERR INT TERM HUP
  failure_context
  exit "${status}"
}

on_signal() {
  local status="$1"
  trap - ERR INT TERM HUP
  failure_context
  exit "${status}"
}

trap on_error ERR
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

while (( $# > 0 )); do
  case "$1" in
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --service)
      (( $# >= 2 )) || die "--service requires a unit name"
      SERVICE="$2"
      shift 2
      ;;
    --terminal-only)
      TERMINAL_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "unknown argument: $1"
      ;;
    *)
      [[ -z "${BUNDLE_ROOT}" ]] || die "only one bundle directory may be provided"
      BUNDLE_ROOT="$1"
      shift
      ;;
  esac
done

[[ "${SERVICE}" =~ ^[A-Za-z0-9_.@-]+\.service$ ]] || die "invalid systemd unit name: ${SERVICE}"
[[ "${GATEWAY_SERVICE}" =~ ^[A-Za-z0-9_.@-]+\.service$ ]] || die "invalid gateway unit name: ${GATEWAY_SERVICE}"

# This entry point is intentionally tied to the validated robot filesystem
# layout.  Lower-level scripts remain reusable for development, but accepting
# their path overrides here would make a successful "one-click" result
# ambiguous (for example, building one prefix while App starts another).
STANDARD_PATH_OVERRIDE_VARS=(
  OPT_PREFIX
  BUILD_INSTALL
  BACKUP_DIR
  PICO_VENV
  CONTROLLER_PREFIX
  SONIC_XRT_SERVICE_DIR
  ROS_SETUP
  BXI_ROS_SETUP
  SONIC_OFFLINE_RUNTIME_DIR
  SONIC_PICO_PYTHON
  SONIC_SYSTEMD_CONFIG_ROOT
  SONIC_SERVICE_BACKUP_DIR
)
for override_name in "${STANDARD_PATH_OVERRIDE_VARS[@]}"; do
  if [[ -v "${override_name}" ]]; then
    die "refusing path override environment variable ${override_name}; the standard robot paths are mandatory"
  fi
done

[[ -n "${BUNDLE_ROOT}" ]] || die "an extracted bundle directory is required"
BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}" && pwd)"
SOURCE_DIR="${BUNDLE_ROOT}/source"
[[ -r "${BUNDLE_ROOT}/MANIFEST.sha256" ]] || die "missing bundle manifest"
[[ -x "${SOURCE_DIR}/script/deploy_robot_sonic_bundle.sh" || \
   -r "${SOURCE_DIR}/script/deploy_robot_sonic_bundle.sh" ]] || \
  die "missing bundle deployment script"

echo "[one-click] verifying bundle before touching robot services"
(cd "${BUNDLE_ROOT}" && sha256sum -c MANIFEST.sha256)

HOSTNAME_NOW="$(hostname)"
echo "[one-click] target=${HOSTNAME_NOW} bundle=${BUNDLE_ROOT} service=${SERVICE}"
echo "[one-click] The robot must be supported, the area clear, and the App control session stopped."
if (( ASSUME_YES == 0 )); then
  [[ -t 0 ]] || die "interactive confirmation is required (or use --yes in controlled automation)"
  read -r -p "Type DEPLOY ${HOSTNAME_NOW} to continue: " confirmation
  [[ "${confirmation}" == "DEPLOY ${HOSTNAME_NOW}" ]] || die "confirmation did not match"
fi

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
  sudo -v
fi

for command in systemctl pgrep fuser flock sha256sum; do
  command -v "${command}" >/dev/null 2>&1 || die "required command is missing: ${command}"
done

# Keep the deployment mutex under a root-owned directory rather than /tmp so
# an unprivileged symlink cannot redirect the lock-file open.  The file itself
# is writable only so the invoking operator can hold the flock without running
# the complete installer as root.
DEPLOY_LOCK_DIR="/run/lock/bxi-sonic-deploy"
DEPLOY_LOCK="${DEPLOY_LOCK_DIR}/installer.lock"
"${SUDO[@]}" install -d -m 0755 "${DEPLOY_LOCK_DIR}"
if ! "${SUDO[@]}" test -e "${DEPLOY_LOCK}"; then
  "${SUDO[@]}" install -m 0666 /dev/null "${DEPLOY_LOCK}"
fi
"${SUDO[@]}" test -f "${DEPLOY_LOCK}" || die "deployment lock is not a regular file: ${DEPLOY_LOCK}"
"${SUDO[@]}" test ! -L "${DEPLOY_LOCK}" || die "deployment lock must not be a symlink: ${DEPLOY_LOCK}"
"${SUDO[@]}" chmod 0666 "${DEPLOY_LOCK}"
exec {DEPLOY_LOCK_FD}<>"${DEPLOY_LOCK}"
flock -n "${DEPLOY_LOCK_FD}" || die "another SONIC deployment is already running"

validate_control_service() {
  local service_exec service_fragment service_load service_env service_domain
  case "${SERVICE}" in
    bxi_rc_ros2.service|robot_gateway.service|ssh.service|sshd.service)
      die "refusing protected/non-control service: ${SERVICE}"
      ;;
  esac
  service_exec="$(systemctl show "${SERVICE}" -p ExecStart --value 2>/dev/null || true)"
  service_fragment="$(systemctl show "${SERVICE}" -p FragmentPath --value 2>/dev/null || true)"
  service_load="$(systemctl show "${SERVICE}" -p LoadState --value 2>/dev/null || true)"
  service_env="$(systemctl show "${SERVICE}" -p Environment --value 2>/dev/null || true)"
  service_domain="$(printf '%s\n' "${service_env}" | tr ' ' '\n' | sed -n -E 's/^"?ROS_DOMAIN_ID=([0-9]+)"?$/\1/p' | head -n 1)"
  echo "[one-click] service FragmentPath=${service_fragment}"
  echo "[one-click] service effective ExecStart=${service_exec}"
  echo "[one-click] service effective Environment=${service_env}"
  [[ "${service_load}" == "loaded" ]] || die "service is not loaded: ${SERVICE}"
  [[ "${service_exec}" == *"ros2 launch remote_controller remote_controller.launch.py"* ]] || \
    die "${SERVICE} is not a recognized remote-controller service; refusing to stop or modify it"
  [[ "${service_exec}" != *"robot_gateway"* ]] || \
    die "${SERVICE} appears to start robot_gateway; refusing to continue"
  if [[ "${service_domain}" =~ ^[0-9]+$ ]] && (( service_domain <= 232 )); then
    echo "[one-click] preserving robot-specific ROS_DOMAIN_ID=${service_domain}"
  elif (( TERMINAL_ONLY == 0 )); then
    die "${SERVICE} has no valid explicit ROS_DOMAIN_ID (expected 0..232)"
  else
    echo "[one-click] WARN: terminal-only deployment has no valid service ROS_DOMAIN_ID"
  fi
}

GATEWAY_SERVICE_WAS_ACTIVE=0
GATEWAY_PROCESS_WAS_ACTIVE=0
if systemctl cat "${GATEWAY_SERVICE}" >/dev/null 2>&1 && \
   systemctl is-active --quiet "${GATEWAY_SERVICE}"; then
  GATEWAY_SERVICE_WAS_ACTIVE=1
  echo "[one-click] gateway service baseline: ${GATEWAY_SERVICE}=active (must remain active)"
elif pgrep -af 'robot_gateway|api_server_node|ros_bridge_node' >/dev/null 2>&1; then
  GATEWAY_PROCESS_WAS_ACTIVE=1
  echo "[one-click] gateway process baseline: active (must remain active)"
else
  echo "[one-click] WARN: no gateway baseline process detected"
fi

ensure_gateway_alive() {
  if (( GATEWAY_SERVICE_WAS_ACTIVE == 1 )) && \
     ! systemctl is-active --quiet "${GATEWAY_SERVICE}"; then
    die "gateway service ${GATEWAY_SERVICE} disappeared during deployment"
  fi
  if (( GATEWAY_PROCESS_WAS_ACTIVE == 1 )) && \
     ! pgrep -af 'robot_gateway|api_server_node|ros_bridge_node' >/dev/null 2>&1; then
    die "gateway process baseline disappeared during deployment"
  fi
}

SERVICE_EXISTS=0
if command -v systemctl >/dev/null 2>&1 && \
   systemctl cat "${SERVICE}" >/dev/null 2>&1; then
  SERVICE_EXISTS=1
  validate_control_service
  echo "[one-click] stopping only ${SERVICE}; gateway services are left running"
  "${SUDO[@]}" systemctl stop "${SERVICE}"
  SERVICE_STOPPED=1
  if systemctl is-active --quiet "${SERVICE}"; then
    die "${SERVICE} is still active after systemctl stop"
  fi
elif (( TERMINAL_ONLY == 0 )); then
  die "${SERVICE} is absent; use --terminal-only only when App integration is intentionally out of scope"
fi

ensure_gateway_alive

for _ in {1..20}; do
  active_processes="$(pgrep -af 'ros2 launch bxi_example_py_elf3|hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' || true)"
  [[ -z "${active_processes}" ]] && break
  sleep 1
done
if [[ -n "${active_processes:-}" ]]; then
  echo "${active_processes}" >&2
  die "control/PICO processes remain; refusing to use pkill or replace /opt"
fi

HARDWARE_LOCK="/tmp/bxi_example_hw.lock"
if lock_owner="$("${SUDO[@]}" fuser -v "${HARDWARE_LOCK}" 2>&1)"; then
  echo "${lock_owner}" >&2
  die "${HARDWARE_LOCK} is still owned; refusing to bypass the single-instance guard"
fi

LOCK_FILES=(
  /var/lib/dpkg/lock-frontend
  /var/lib/dpkg/lock
  /var/cache/apt/archives/lock
)
if lock_owners="$("${SUDO[@]}" fuser -v "${LOCK_FILES[@]}" 2>&1)"; then
  echo "${lock_owners}" >&2
  die "apt/dpkg lock is active; wait for or safely stop the package manager first"
fi

SONIC_CONTROL_SERVICE="${SERVICE}" bash "${SOURCE_DIR}/script/audit_robot_sonic_host.sh"

# Hold the same lock used by example_demo_hw.launch.py for the entire mutation
# window.  This prevents an App or terminal race from starting hardware while
# colcon/copy/pip/dpkg is changing the installed runtime.
if [[ -L "${HARDWARE_LOCK}" ]]; then
  die "hardware lock must not be a symlink: ${HARDWARE_LOCK}"
fi
if [[ ! -e "${HARDWARE_LOCK}" ]]; then
  if ! (set -o noclobber; : >"${HARDWARE_LOCK}") 2>/dev/null; then
    die "could not safely create hardware lock: ${HARDWARE_LOCK}"
  fi
fi
[[ -f "${HARDWARE_LOCK}" && ! -L "${HARDWARE_LOCK}" ]] || \
  die "hardware lock is not a regular file: ${HARDWARE_LOCK}"
chmod 0666 "${HARDWARE_LOCK}" 2>/dev/null || \
  "${SUDO[@]}" chmod 0666 "${HARDWARE_LOCK}"
exec {HARDWARE_LOCK_FD}<>"${HARDWARE_LOCK}"
flock -n "${HARDWARE_LOCK_FD}" || \
  die "hardware session started during preflight; deployment aborted"
echo "[one-click] acquired hardware launch lock for the installation window"

bash "${SOURCE_DIR}/script/deploy_robot_sonic_bundle.sh" "${BUNDLE_ROOT}"

if (( SERVICE_EXISTS == 1 && TERMINAL_ONLY == 0 )); then
  SONIC_CONTROL_SERVICE="${SERVICE}" \
    bash "${SOURCE_DIR}/script/configure_robot_sonic_service.sh" \
      --install --service "${SERVICE}"
elif (( SERVICE_EXISTS == 0 )); then
  echo "[one-click] WARN: ${SERVICE} is absent; terminal runtime is installed but App startup needs integration"
else
  echo "[one-click] terminal-only requested; existing service configuration was not changed"
fi

if (( SERVICE_EXISTS == 1 && TERMINAL_ONLY == 0 )); then
  SONIC_REQUIRE_SERVICE_INTEGRATION=1 \
    SONIC_CONTROL_SERVICE="${SERVICE}" \
    bash "${SOURCE_DIR}/script/check_robot_sonic_runtime.sh"
else
  SONIC_CONTROL_SERVICE="${SERVICE}" bash "${SOURCE_DIR}/script/check_robot_sonic_runtime.sh"
fi

ensure_gateway_alive

if (( SERVICE_EXISTS == 1 )); then
  final_service_state="$(systemctl show "${SERVICE}" -p ActiveState --value 2>/dev/null || true)"
  [[ "${final_service_state}" == "inactive" ]] || \
    die "${SERVICE} ActiveState=${final_service_state:-unknown} after installation; expected inactive"
  echo "[one-click] verified ${SERVICE} remains inactive"
fi

echo "[one-click] installation complete"
echo "[one-click] ${SERVICE} remains stopped; no hardware process was started"
echo "[one-click] next: perform the documented T1/T2 terminal acceptance using this robot's ROS domain"
trap - ERR INT TERM HUP
