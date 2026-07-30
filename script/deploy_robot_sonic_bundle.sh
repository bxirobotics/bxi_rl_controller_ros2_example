#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT="${1:-}"
if [[ -z "${BUNDLE_ROOT}" ]]; then
  echo "usage: $0 /path/to/extracted-bundle" >&2
  exit 2
fi
BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}" && pwd)"
SOURCE_DIR="${BUNDLE_ROOT}/source"
RUNTIME_DIR="${BUNDLE_ROOT}/runtime"
MANIFEST="${BUNDLE_ROOT}/MANIFEST.sha256"

[[ -r "${MANIFEST}" ]] || { echo "missing ${MANIFEST}" >&2; exit 2; }
[[ -r "${SOURCE_DIR}/SONIC_DEPLOY_COMMIT" ]] || {
  echo "missing source commit marker" >&2
  exit 2
}

echo "[bundle-deploy] verifying bundle checksums"
(cd "${BUNDLE_ROOT}" && sha256sum -c MANIFEST.sha256)

EXPECTED_COMMIT="$(tr -d '[:space:]' < "${SOURCE_DIR}/SONIC_DEPLOY_COMMIT")"
echo "[bundle-deploy] expected commit=${EXPECTED_COMMIT}"

ACTIVE_PROCESSES_FILE="$(mktemp)"
trap 'rm -f -- "${ACTIVE_PROCESSES_FILE}"' EXIT
if pgrep -af 'hardware_elf3|bxi_example_py_elf3_demo|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess' >"${ACTIVE_PROCESSES_FILE}"; then
  echo "[bundle-deploy] ERROR: robot control/PICO processes are still active:" >&2
  sed -n '1,80p' "${ACTIVE_PROCESSES_FILE}" >&2
  echo "[bundle-deploy] stop the active robot control session before replacing /opt" >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  sudo -v
fi

echo "[bundle-deploy] deploying ROS packages"
env \
  -u OPT_PREFIX \
  -u BUILD_INSTALL \
  -u BACKUP_DIR \
  OFFLINE_SOURCE=1 \
  EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  SRC_DIR="${SOURCE_DIR}" \
  bash "${SOURCE_DIR}/script/deploy_robot_sonic_example.sh"

echo "[bundle-deploy] installing offline runtime"
env \
  -u SONIC_OFFLINE_RUNTIME_DIR \
  -u PICO_VENV \
  -u CONTROLLER_PREFIX \
  -u SONIC_XRT_SERVICE_DIR \
  -u ROS_SETUP \
  -u BXI_ROS_SETUP \
  bash "${SOURCE_DIR}/script/install_robot_sonic_runtime_offline.sh" "${RUNTIME_DIR}"

echo "[bundle-deploy] running complete health check"
env \
  -u SONIC_PICO_PYTHON \
  -u SONIC_XRT_SERVICE_DIR \
  bash "${SOURCE_DIR}/script/check_robot_sonic_runtime.sh"

echo "[bundle-deploy] deployment complete: ${EXPECTED_COMMIT}"
