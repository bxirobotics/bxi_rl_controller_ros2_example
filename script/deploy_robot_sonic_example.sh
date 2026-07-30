#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bxirobotics/bxi_rl_controller_ros2_example.git}"
BRANCH="${BRANCH:-feature/sonic-elf3-runtime-gripper-clean}"
SRC_DIR="${SRC_DIR:-${HOME}/bxi_rl_controller_ros2_example}"
OPT_PREFIX="${OPT_PREFIX:-/opt/bxi/bxi_rl_controller_ros2_example}"
BUILD_INSTALL="${BUILD_INSTALL:-/tmp/elf3_sonic_install}"
BACKUP_DIR="${BACKUP_DIR:-/opt/bxi/deploy_backups}"
OFFLINE_SOURCE="${OFFLINE_SOURCE:-0}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

die() {
  echo "[deploy] ERROR: $*" >&2
  exit 2
}

command -v realpath >/dev/null 2>&1 || die "realpath is required for path validation"
SRC_DIR="$(realpath -m "${SRC_DIR}")"
OPT_PREFIX="$(realpath -m "${OPT_PREFIX}")"
BUILD_INSTALL="$(realpath -m "${BUILD_INSTALL}")"
BACKUP_DIR="$(realpath -m "${BACKUP_DIR}")"

[[ "${OPT_PREFIX}" == /opt/bxi/* && "${OPT_PREFIX}" != "/opt/bxi" ]] || \
  die "OPT_PREFIX must be a package directory below /opt/bxi: ${OPT_PREFIX}"
[[ "${BACKUP_DIR}" == /opt/bxi/* && "${BACKUP_DIR}" != "/opt/bxi" ]] || \
  die "BACKUP_DIR must be below /opt/bxi: ${BACKUP_DIR}"
[[ "${BUILD_INSTALL}" == /tmp/elf3_sonic_install* && "${BUILD_INSTALL}" != "/tmp" ]] || \
  die "BUILD_INSTALL must use the dedicated /tmp/elf3_sonic_install* prefix: ${BUILD_INSTALL}"
[[ "${SRC_DIR}" != "/" && -n "${SRC_DIR}" ]] || die "unsafe SRC_DIR: ${SRC_DIR}"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

source_setup() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "[deploy] ERROR: missing ${file}"
    exit 2
  fi
  set +u
  # shellcheck disable=SC1090
  source "${file}"
  set -u
}

echo "[deploy] repo=${REPO_URL}"
echo "[deploy] branch=${BRANCH}"
echo "[deploy] src=${SRC_DIR}"
echo "[deploy] opt=${OPT_PREFIX}"

if [[ "${OFFLINE_SOURCE}" == "1" ]]; then
  if [[ ! -f "${SRC_DIR}/src/bxi_example_py_elf3/package.xml" ]] || \
     [[ ! -f "${SRC_DIR}/src/remote_controller/package.xml" ]]; then
    echo "[deploy] ERROR: offline source tree is incomplete: ${SRC_DIR}"
    exit 2
  fi
  cd "${SRC_DIR}"
  if [[ -d .git ]]; then
    source_commit="$(git rev-parse HEAD)"
  elif [[ -r SONIC_DEPLOY_COMMIT ]]; then
    source_commit="$(tr -d '[:space:]' < SONIC_DEPLOY_COMMIT)"
  else
    source_commit="unknown"
  fi
elif [[ -d "${SRC_DIR}/.git" ]]; then
  cd "${SRC_DIR}"
  if [[ "${ALLOW_DIRTY:-0}" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "[deploy] ERROR: ${SRC_DIR} has local changes."
    echo "[deploy] Commit/stash them first, or rerun with ALLOW_DIRTY=1 to overwrite intentionally."
    exit 2
  fi
  git remote add sonic "${REPO_URL}" 2>/dev/null || git remote set-url sonic "${REPO_URL}"
  git fetch sonic "${BRANCH}"
  git checkout -B "${BRANCH}" "sonic/${BRANCH}"
else
  git clone -b "${BRANCH}" --single-branch "${REPO_URL}" "${SRC_DIR}"
  cd "${SRC_DIR}"
  source_commit="$(git rev-parse HEAD)"
fi

if [[ "${OFFLINE_SOURCE}" != "1" ]]; then
  source_commit="$(git rev-parse HEAD)"
fi
if [[ -n "${EXPECTED_COMMIT}" ]] && [[ "${source_commit}" != "${EXPECTED_COMMIT}"* ]]; then
  echo "[deploy] ERROR: source commit ${source_commit} does not match expected ${EXPECTED_COMMIT}"
  exit 2
fi
echo "[deploy] commit=${source_commit} offline=${OFFLINE_SOURCE}"

source_setup /opt/ros/humble/setup.bash
source_setup /opt/bxi/bxi_ros2_pkg/setup.bash

rm -rf build log "${BUILD_INSTALL}"

colcon build \
  --merge-install \
  --install-base "${BUILD_INSTALL}" \
  --packages-select bxi_example_py_elf3 remote_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

"${SUDO[@]}" mkdir -p "${BACKUP_DIR}"
backup_name="bxi_rl_controller_ros2_example.before_sonic_$(date +%Y%m%d_%H%M%S).tgz"

if [[ -d "${OPT_PREFIX}" ]]; then
  "${SUDO[@]}" tar -C "$(dirname "${OPT_PREFIX}")" -czf "${BACKUP_DIR}/${backup_name}" "$(basename "${OPT_PREFIX}")"
  echo "[deploy] backup=${BACKUP_DIR}/${backup_name}"
else
  echo "[deploy] warning: ${OPT_PREFIX} does not exist; creating it"
  "${SUDO[@]}" mkdir -p "${OPT_PREFIX}"
fi

"${SUDO[@]}" cp -a "${BUILD_INSTALL}/." "${OPT_PREFIX}/"

source_setup "${OPT_PREFIX}/setup.bash"

ros2 pkg prefix bxi_example_py_elf3
ros2 pkg prefix remote_controller
ros2 pkg executables bxi_example_py_elf3 | grep sonic_pico_runtime_supervisor
grep -n "sonic_pico_python" "${OPT_PREFIX}/share/bxi_example_py_elf3/launch/example_demo_hw.launch.py"
grep -n "sonic_teleop" "${OPT_PREFIX}/share/bxi_example_py_elf3/config/elf3_state_machine.yaml" | head

echo "[deploy] done"
echo "[deploy] next: bash ${SRC_DIR}/script/check_robot_sonic_runtime.sh"
