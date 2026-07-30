#!/usr/bin/env bash
# T3 supervisor: own the PICO manager and bridge for their complete lifetime.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BXI_RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BXI_RUNTIME_ROOT}"
export BXI_RUNTIME_ROOT
export PYTHONPATH="${BXI_RUNTIME_ROOT}/src/bxi_example_py_elf3:${BXI_RUNTIME_ROOT}:${PYTHONPATH:-}"

if [[ -f .venv_teleop/bin/activate ]]; then
  source .venv_teleop/bin/activate
fi

PICO_HOST="${PICO_HOST:-127.0.0.1}"
PICO_PORT="${PICO_PORT:-5556}"
SMPL_REF_ZMQ_HOST="${SMPL_REF_ZMQ_HOST:-127.0.0.1}"
SMPL_REF_ZMQ_PORT="${SMPL_REF_ZMQ_PORT:-5557}"
SMPL_REF_ZMQ_TOPIC="${SMPL_REF_ZMQ_TOPIC:-smpl_ref}"
PICO_ENABLE_ROS_BUTTONS="${PICO_ENABLE_ROS_BUTTONS:-0}"
SONIC_PICO_USE_CUDA="${SONIC_PICO_USE_CUDA:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STOP_INT_TIMEOUT="${SONIC_PICO_STOP_INT_TIMEOUT:-3}"
STOP_TERM_TIMEOUT="${SONIC_PICO_STOP_TERM_TIMEOUT:-2}"
STOP_KILL_TIMEOUT="${SONIC_PICO_STOP_KILL_TIMEOUT:-1}"

manager_arg_text="${PICO_MANAGER_ARGS:-}"
if [[ -z "${manager_arg_text}" ]]; then
  PICO_MANAGER_ARGS=(
    --manager
    --num_frames_to_send 10
    --target_fps 50
    --port "${PICO_PORT}"
  )
  if [[ "${SONIC_PICO_USE_CUDA,,}" == "1" || "${SONIC_PICO_USE_CUDA,,}" == "true" || "${SONIC_PICO_USE_CUDA,,}" == "yes" || "${SONIC_PICO_USE_CUDA,,}" == "on" ]]; then
    PICO_MANAGER_ARGS+=(--cuda)
  fi
else
  read -r -a PICO_MANAGER_ARGS <<< "${manager_arg_text}"
fi

if [[ "${PICO_ENABLE_VIS:-0}" == "1" ]]; then
  echo "[sonic-pico-sources] ERROR: PICO_ENABLE_VIS is not supported by the headless ELF3 runtime" >&2
  exit 2
fi

LOCK_FILE="${SONIC_PICO_LOCK_FILE:-/tmp/sonic_pico_sources_${UID}.lock}"
exec {lock_fd}>"${LOCK_FILE}"
if ! flock -n "${lock_fd}"; then
  echo "[sonic-pico-sources] ERROR: another T3 supervisor owns ${LOCK_FILE}" >&2
  exit 75
fi

pico_pid=""
pico_pgid=""
bridge_pid=""
bridge_pgid=""
cleanup_started=0

process_group_alive() {
  local pgid="$1"
  [[ -n "${pgid}" ]] && kill -0 -- "-${pgid}" 2>/dev/null
}

wait_for_process_group() {
  local pgid="$1"
  local timeout_seconds="$2"
  local attempts=$((timeout_seconds * 10))
  local attempt

  for ((attempt = 0; attempt < attempts; attempt++)); do
    if ! process_group_alive "${pgid}"; then
      return 0
    fi
    sleep 0.1
  done
  ! process_group_alive "${pgid}"
}

stop_process_group() {
  local name="$1"
  local pid="$2"
  local pgid="$3"

  [[ -n "${pid}" && -n "${pgid}" ]] || return 0
  if ! process_group_alive "${pgid}"; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "[sonic-pico-sources] stopping ${name} pid=${pid} pgid=${pgid} with SIGINT"
    kill -INT "${pid}" 2>/dev/null || true
    if wait_for_process_group "${pgid}" "${STOP_INT_TIMEOUT}"; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
  else
    echo "[sonic-pico-sources] ${name} leader exited; cleaning remaining pgid=${pgid}"
  fi

  echo "[sonic-pico-sources] ${name} exceeded INT grace; sending SIGTERM to pgid=${pgid}"
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  if wait_for_process_group "${pgid}" "${STOP_TERM_TIMEOUT}"; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi

  echo "[sonic-pico-sources] ${name} exceeded TERM grace; sending SIGKILL to pgid=${pgid}" >&2
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  wait_for_process_group "${pgid}" "${STOP_KILL_TIMEOUT}" || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local exit_code=$?
  if ((cleanup_started)); then
    return
  fi
  cleanup_started=1
  trap - EXIT INT TERM
  set +e

  stop_process_group "bridge" "${bridge_pid}" "${bridge_pgid}"
  stop_process_group "manager" "${pico_pid}" "${pico_pgid}"
  echo "[sonic-pico-sources] cleanup complete"
  exit "${exit_code}"
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

echo "[sonic-pico-sources] starting legacy PICO manager wrapper"
setsid "${PYTHON_BIN}" -m bxi_example_py_elf3.sonic_pico.pico_manager_legacy \
  "${PICO_MANAGER_ARGS[@]}" &
pico_pid=$!
pico_pgid="${pico_pid}"

sleep 1.0
if ! kill -0 "${pico_pid}" 2>/dev/null; then
  set +e
  wait "${pico_pid}"
  manager_exit=$?
  set -e
  echo "[sonic-pico-sources] ERROR: manager exited during startup rc=${manager_exit}" >&2
  exit "${manager_exit}"
fi

bridge_args=(
  -m bxi_example_py_elf3.sonic_pico.pico_pose_to_smpl_ref_bridge
  --pico-host "${PICO_HOST}"
  --pico-port "${PICO_PORT}"
  --pico-topic pose
  --out-host "${SMPL_REF_ZMQ_HOST}"
  --out-port "${SMPL_REF_ZMQ_PORT}"
  --out-topic "${SMPL_REF_ZMQ_TOPIC}"
)
if [[ "${PICO_ENABLE_ROS_BUTTONS}" != "1" ]]; then
  bridge_args+=(--disable-ros-pico-topics)
fi

echo "[sonic-pico-sources] starting PICO pose -> ELF3 smpl_ref bridge"
setsid "${PYTHON_BIN}" "${bridge_args[@]}" &
bridge_pid=$!
bridge_pgid="${bridge_pid}"

echo "[sonic-pico-sources] ready"
echo "  manager pid=${pico_pid} pgid=${pico_pgid}"
echo "  bridge pid=${bridge_pid} pgid=${bridge_pgid} ros_buttons=${PICO_ENABLE_ROS_BUTTONS}"
echo "  controller flow: pd_brake -> normal -> sonic_teleop"
echo "  PICO mode: CALIB_FULL / PLANNER, then POSE"
echo "  wrist_source=elf3_native"

exited_pid=""
set +e
wait -n -p exited_pid "${pico_pid}" "${bridge_pid}"
child_exit=$?
set -e
echo "[sonic-pico-sources] child exited pid=${exited_pid:-unknown} rc=${child_exit}; stopping T3"
exit "${child_exit}"
