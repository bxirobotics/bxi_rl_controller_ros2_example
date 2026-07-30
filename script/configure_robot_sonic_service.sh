#!/usr/bin/env bash
set -Eeuo pipefail

MODE="check"
SERVICE="${SONIC_CONTROL_SERVICE:-ros_elf_launch.service}"
CONTROLLER_PREFIX="${CONTROLLER_PREFIX:-/opt/bxi/bxi_rl_controller_ros2_example}"
DROPIN_NAME="sonic-runtime.conf"

usage() {
  cat <<'EOF'
usage: configure_robot_sonic_service.sh [--check|--install] [--service NAME]

Checks or installs a systemd drop-in that makes the robot remote-controller
service source the deployed /opt example package.  Existing Environment entries
(ROS_DOMAIN_ID, RMW and CycloneDDS settings) are deliberately left untouched.
The script never starts the service or robot hardware.
EOF
}

die() {
  echo "[service-config] ERROR: $*" >&2
  exit 2
}

while (( $# > 0 )); do
  case "$1" in
    --check)
      MODE="check"
      shift
      ;;
    --install)
      MODE="install"
      shift
      ;;
    --service)
      (( $# >= 2 )) || die "--service requires a unit name"
      SERVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${SERVICE}" =~ ^[A-Za-z0-9_.@-]+\.service$ ]] || die "invalid systemd unit name: ${SERVICE}"
[[ "${CONTROLLER_PREFIX}" =~ ^/[A-Za-z0-9_./-]+$ ]] || die "invalid controller prefix: ${CONTROLLER_PREFIX}"
command -v realpath >/dev/null 2>&1 || die "realpath is unavailable"
CONTROLLER_PREFIX="$(realpath -m "${CONTROLLER_PREFIX}")"

command -v systemctl >/dev/null 2>&1 || die "systemctl is unavailable"
systemctl cat "${SERVICE}" >/dev/null 2>&1 || die "systemd unit not found: ${SERVICE}"

EXPECTED_SETUP="${CONTROLLER_PREFIX}/setup.bash"
[[ -r "${EXPECTED_SETUP}" ]] || die "missing deployed setup: ${EXPECTED_SETUP}"

validate_service_identity() {
  local exec_start load_state fragment_path
  case "${SERVICE}" in
    bxi_rc_ros2.service|robot_gateway.service|ssh.service|sshd.service)
      die "refusing protected/non-control service: ${SERVICE}"
      ;;
  esac
  exec_start="$(systemctl show "${SERVICE}" -p ExecStart --value 2>/dev/null || true)"
  load_state="$(systemctl show "${SERVICE}" -p LoadState --value 2>/dev/null || true)"
  fragment_path="$(systemctl show "${SERVICE}" -p FragmentPath --value 2>/dev/null || true)"
  [[ "${load_state}" == "loaded" ]] || die "service is not loaded: ${SERVICE}"
  [[ "${exec_start}" == *"ros2 launch remote_controller remote_controller.launch.py"* ]] || \
    die "${SERVICE} is not a recognized remote-controller service"
  [[ "${exec_start}" != *"robot_gateway"* ]] || \
    die "${SERVICE} appears to start robot_gateway; refusing to modify it"
  echo "[service-config] validated remote-controller unit FragmentPath=${fragment_path}"
}

validate_service_identity

show_effective() {
  local exec_start environment load_state active_state dropin_paths
  exec_start="$(systemctl show "${SERVICE}" -p ExecStart --value 2>/dev/null || true)"
  environment="$(systemctl show "${SERVICE}" -p Environment --value 2>/dev/null || true)"
  load_state="$(systemctl show "${SERVICE}" -p LoadState --value 2>/dev/null || true)"
  active_state="$(systemctl show "${SERVICE}" -p ActiveState --value 2>/dev/null || true)"
  dropin_paths="$(systemctl show "${SERVICE}" -p DropInPaths --value 2>/dev/null || true)"
  echo "[service-config] service=${SERVICE}"
  echo "[service-config] LoadState=${load_state} ActiveState=${active_state}"
  echo "[service-config] ExecStart=${exec_start}"
  echo "[service-config] Environment=${environment}"
  echo "[service-config] DropInPaths=${dropin_paths}"
  [[ "${exec_start}" == *"${EXPECTED_SETUP}"* ]] && \
    [[ "${exec_start}" == *"ros2 launch remote_controller remote_controller.launch.py"* ]] && \
    [[ "${exec_start}" != *"bxi_rl_controller_ros2_example/install/setup.bash"* ]]
}

if [[ "${MODE}" == "check" ]]; then
  if show_effective; then
    echo "[service-config] OK: service starts the deployed /opt package"
    exit 0
  fi
  echo "[service-config] WARN: service does not start ${EXPECTED_SETUP}" >&2
  exit 1
fi

if systemctl is-active --quiet "${SERVICE}"; then
  die "${SERVICE} is active; stop the control session before changing its launch command"
fi

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

SYSTEMD_CONFIG_ROOT="${SONIC_SYSTEMD_CONFIG_ROOT:-/etc/systemd/system}"
[[ "${SYSTEMD_CONFIG_ROOT}" == /* ]] || die "systemd config root must be absolute: ${SYSTEMD_CONFIG_ROOT}"
SYSTEMD_CONFIG_ROOT="$(realpath -m "${SYSTEMD_CONFIG_ROOT}")"
DROPIN_DIR="${SYSTEMD_CONFIG_ROOT}/${SERVICE}.d"
DROPIN_PATH="${DROPIN_DIR}/${DROPIN_NAME}"
TMP_FILE="$(mktemp)"
TMP_METADATA="$(mktemp)"
HAD_DROPIN=0
BEFORE_ENV="$(systemctl show "${SERVICE}" -p Environment --value 2>/dev/null || true)"
BEFORE_EXEC="$(systemctl show "${SERVICE}" -p ExecStart --value 2>/dev/null || true)"
BEFORE_DROPINS="$(systemctl show "${SERVICE}" -p DropInPaths --value 2>/dev/null || true)"
DROPIN_MUTATED=0
BACKUP_DIR="${SONIC_SERVICE_BACKUP_DIR:-/opt/bxi/deploy_backups/systemd}"
[[ "${BACKUP_DIR}" == /* ]] || die "service backup directory must be absolute: ${BACKUP_DIR}"
BACKUP_DIR="$(realpath -m "${BACKUP_DIR}")"
BACKUP_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BACKUP_PREFIX="${BACKUP_DIR}/${SERVICE}.${BACKUP_STAMP}"
DURABLE_DROPIN_BACKUP=""
trap 'rm -f "${TMP_FILE}" "${TMP_METADATA}"' EXIT

cat >"${TMP_FILE}" <<EOF
[Service]
ExecStart=
ExecStart=/bin/bash -lc "source /opt/ros/humble/setup.bash && source /opt/bxi/bxi_ros2_pkg/setup.bash && source ${EXPECTED_SETUP} && exec ros2 launch remote_controller remote_controller.launch.py"
EOF

"${SUDO[@]}" install -d -m 0755 "${DROPIN_DIR}"
"${SUDO[@]}" install -d -m 0755 "${BACKUP_DIR}"
if "${SUDO[@]}" test -e "${DROPIN_PATH}"; then
  HAD_DROPIN=1
  DURABLE_DROPIN_BACKUP="${BACKUP_PREFIX}.${DROPIN_NAME}.bak"
  "${SUDO[@]}" cp -a "${DROPIN_PATH}" "${DURABLE_DROPIN_BACKUP}"
fi

printf '%s\n' \
  "service=${SERVICE}" \
  "dropin=${DROPIN_PATH}" \
  "had_dropin=${HAD_DROPIN}" \
  "previous_exec=${BEFORE_EXEC}" \
  "previous_environment=${BEFORE_ENV}" \
  "previous_dropins=${BEFORE_DROPINS}" \
  >"${TMP_METADATA}"
"${SUDO[@]}" install -m 0600 "${TMP_METADATA}" "${BACKUP_PREFIX}.effective-state.txt"

rollback_dropin() {
  local failed=0
  if (( HAD_DROPIN == 1 )); then
    "${SUDO[@]}" install -m 0644 "${DURABLE_DROPIN_BACKUP}" "${DROPIN_PATH}" || failed=1
  else
    "${SUDO[@]}" rm -f "${DROPIN_PATH}" || failed=1
  fi
  "${SUDO[@]}" systemctl daemon-reload || failed=1
  if (( failed != 0 )); then
    echo "[service-config] CRITICAL: automatic rollback failed" >&2
    echo "[service-config] recovery metadata: ${BACKUP_PREFIX}.effective-state.txt" >&2
    [[ -z "${DURABLE_DROPIN_BACKUP}" ]] || \
      echo "[service-config] previous drop-in backup: ${DURABLE_DROPIN_BACKUP}" >&2
    return 1
  fi
  DROPIN_MUTATED=0
}

handle_failure() {
  local status="${1:-1}"
  trap - ERR INT TERM HUP
  set +e
  if (( DROPIN_MUTATED == 1 )); then
    echo "[service-config] restoring previous drop-in after failure" >&2
    rollback_dropin || true
  fi
  exit "${status}"
}

trap 'handle_failure $?' ERR
trap 'handle_failure 130' INT
trap 'handle_failure 143' TERM
trap 'handle_failure 129' HUP

DROPIN_MUTATED=1
"${SUDO[@]}" install -m 0644 "${TMP_FILE}" "${DROPIN_PATH}"
"${SUDO[@]}" systemctl daemon-reload

AFTER_ENV="$(systemctl show "${SERVICE}" -p Environment --value 2>/dev/null || true)"
AFTER_EXEC="$(systemctl show "${SERVICE}" -p ExecStart --value 2>/dev/null || true)"

if [[ "${AFTER_ENV}" != "${BEFORE_ENV}" ]]; then
  rollback_dropin
  die "effective Environment changed; restored the previous drop-in (before=${BEFORE_ENV@Q}, after=${AFTER_ENV@Q})"
fi

if [[ "${AFTER_EXEC}" != *"${EXPECTED_SETUP}"* ]]; then
  rollback_dropin
  die "effective ExecStart did not update; restored the previous drop-in"
fi

if show_effective; then
  DROPIN_MUTATED=0
  trap - ERR INT TERM HUP
  echo "[service-config] installed ${DROPIN_PATH}"
  echo "[service-config] service remains stopped; existing Environment was preserved"
  echo "[service-config] recovery metadata=${BACKUP_PREFIX}.effective-state.txt"
  [[ -z "${DURABLE_DROPIN_BACKUP}" ]] || \
    echo "[service-config] previous drop-in backup=${DURABLE_DROPIN_BACKUP}"
  echo "[service-config] previous ExecStart=${BEFORE_EXEC}"
  echo "[service-config] previous DropInPaths=${BEFORE_DROPINS}"
else
  rollback_dropin
  die "effective ExecStart did not update after installing ${DROPIN_PATH}"
fi
