#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILURES=0
STATE_TMP_DIR=""
if ! STATE_TMP_DIR="$(mktemp -d)" || [[ -z "${STATE_TMP_DIR}" || ! -d "${STATE_TMP_DIR}" ]]; then
  echo "[FAIL] could not create a private runtime-check temporary directory" >&2
  exit 2
fi
STATE_OUTPUT="${STATE_TMP_DIR}/state_machine_info.txt"
STATE_ERROR="${STATE_TMP_DIR}/state_machine_info.err"
trap 'rm -rf -- "${STATE_TMP_DIR}"' EXIT

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; FAILURES=$((FAILURES + 1)); }
section() { echo; echo "== $* =="; }

source_if_exists() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${file}"
    set -u
    ok "sourced ${file}"
  else
    warn "missing ${file}"
  fi
}

section "ROS environment"
source_if_exists /opt/ros/humble/setup.bash
source_if_exists /opt/bxi/bxi_ros2_pkg/setup.bash
source_if_exists /opt/bxi/bxi_rl_controller_ros2_example/setup.bash
echo "[INFO] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-}"

section "Installed package"
EXPECTED_PREFIX="/opt/bxi/bxi_rl_controller_ros2_example"
if prefix="$(ros2 pkg prefix bxi_example_py_elf3 2>/dev/null)"; then
  if [[ "${prefix}" == "${EXPECTED_PREFIX}" ]]; then
    ok "bxi_example_py_elf3 prefix=${prefix}"
  else
    fail "bxi_example_py_elf3 prefix=${prefix}; expected exactly ${EXPECTED_PREFIX}"
  fi
else
  fail "ros2 cannot find bxi_example_py_elf3"
fi

if prefix="$(ros2 pkg prefix remote_controller 2>/dev/null)"; then
  if [[ "${prefix}" == "${EXPECTED_PREFIX}" ]]; then
    ok "remote_controller prefix=${prefix}"
  else
    fail "remote_controller prefix=${prefix}; expected exactly ${EXPECTED_PREFIX}"
  fi
else
  fail "ros2 cannot find remote_controller"
fi

if ros2 pkg executables bxi_example_py_elf3 2>/dev/null | grep -q sonic_pico_runtime_supervisor; then
  ok "sonic_pico_runtime_supervisor executable is installed"
else
  fail "missing sonic_pico_runtime_supervisor executable"
fi

HW_LAUNCH="${EXPECTED_PREFIX}/share/bxi_example_py_elf3/launch/example_demo_hw.launch.py"
if [[ -f "${HW_LAUNCH}" ]]; then
  if grep -q "sonic_pico_python" "${HW_LAUNCH}"; then
    ok "hardware launch exposes sonic_pico_python"
  else
    fail "hardware launch does not expose sonic_pico_python; redeploy latest package"
  fi
  if grep -q '"target_states"' "${HW_LAUNCH}" && \
      grep -q 'sonic_teleop_gripper' "${HW_LAUNCH}" && \
      grep -q 'PICO_ENABLE_ROS_BUTTONS' "${HW_LAUNCH}"; then
    ok "hardware launch enables both SONIC modes and PICO trigger topics"
  else
    fail "hardware launch is missing dual-SONIC/PICO-trigger integration"
  fi
else
  fail "installed hardware launch file not found under ${EXPECTED_PREFIX}"
fi

STATE_MACHINE_CONFIG="${EXPECTED_PREFIX}/share/bxi_example_py_elf3/config/elf3_state_machine.yaml"
if [[ ! -r "${STATE_MACHINE_CONFIG}" ]]; then
  fail "installed ELF3 state-machine config is missing: ${STATE_MACHINE_CONFIG}"
elif python3 - "${STATE_MACHINE_CONFIG}" <<'PY'
import sys

import yaml


path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

if not isinstance(config, dict):
    raise SystemExit("state-machine document is not a mapping")

expected = {
    "sonic_teleop": ("sonic_teleop_event", 7, False),
    "sonic_teleop_gripper": ("sonic_teleop_gripper_event", 8, True),
}
for state_name, (event_name, value, hardware_gripper) in expected.items():
    event = config.get("remote_events", {}).get(event_name)
    if not isinstance(event, dict):
        raise SystemExit(f"remote_events.{event_name} is missing")
    if event.get("slot") != "btn_10" or event.get("value") != value:
        raise SystemExit(
            f"remote_events.{event_name} must be slot=btn_10,value={value}; "
            f"found slot={event.get('slot')!r},value={event.get('value')!r}"
        )

    state = config.get("states", {}).get(state_name)
    if not isinstance(state, dict):
        raise SystemExit(f"states.{state_name} is missing")
    if not isinstance(state.get("manifest"), dict) or not state["manifest"]:
        raise SystemExit(f"states.{state_name}.manifest is missing or empty")
    if state.get("behavior") != "SonicTeleopState":
        raise SystemExit(f"states.{state_name}.behavior must be SonicTeleopState")
    if bool(state.get("params", {}).get("hardware_gripper")) != hardware_gripper:
        raise SystemExit(
            f"states.{state_name}.params.hardware_gripper must be "
            f"{hardware_gripper}"
        )
    print(
        f"[OK] {state_name} mapping: btn_10={value}, "
        f"hardware_gripper={hardware_gripper}"
    )
PY
then
  ok "installed ELF3 state-machine SONIC mapping is valid"
else
  fail "installed ELF3 state-machine SONIC mapping is invalid"
fi

section "Control service integration"
CONTROL_SERVICE="${SONIC_CONTROL_SERVICE:-ros_elf_launch.service}"
EXPECTED_SETUP="/opt/bxi/bxi_rl_controller_ros2_example/setup.bash"
REQUIRE_SERVICE_INTEGRATION="${SONIC_REQUIRE_SERVICE_INTEGRATION:-0}"
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
  if [[ "${service_exec}" == *"${EXPECTED_SETUP}"* ]] && \
     [[ "${service_exec}" == *"ros2 launch remote_controller remote_controller.launch.py"* ]] && \
     [[ "${service_exec}" != *"bxi_rl_controller_ros2_example/install/setup.bash"* ]]; then
    ok "${CONTROL_SERVICE} starts the /opt example install"
  elif [[ "${REQUIRE_SERVICE_INTEGRATION}" == "1" ]]; then
    fail "${CONTROL_SERVICE} does not start ${EXPECTED_SETUP}"
  else
    warn "${CONTROL_SERVICE} still points outside ${EXPECTED_SETUP}; terminal tests use /opt but App may start old code"
  fi
  if [[ "${service_domain}" =~ ^[0-9]+$ ]] && (( service_domain <= 232 )); then
    ok "${CONTROL_SERVICE} has numeric ROS_DOMAIN_ID=${service_domain}"
  elif [[ "${REQUIRE_SERVICE_INTEGRATION}" == "1" ]]; then
    fail "${CONTROL_SERVICE} has no valid explicit ROS_DOMAIN_ID (expected 0..232)"
  else
    warn "${CONTROL_SERVICE} has no valid explicit ROS_DOMAIN_ID (expected 0..232)"
  fi
  if [[ "${REQUIRE_SERVICE_INTEGRATION}" == "1" ]]; then
    if [[ "${service_active}" == "inactive" ]]; then
      ok "${CONTROL_SERVICE} is inactive during the deployment check"
    else
      fail "${CONTROL_SERVICE} ActiveState=${service_active:-unknown}; expected inactive"
    fi
  fi
else
  if [[ "${REQUIRE_SERVICE_INTEGRATION}" == "1" ]]; then
    fail "${CONTROL_SERVICE} is not installed; App startup integration is incomplete"
  else
    warn "${CONTROL_SERVICE} is not installed; App startup integration was not checked"
  fi
fi

section "Controller Python dependencies"
if python3 - <<'PY'
import importlib
import sys

required = [
    "numpy",
    "onnxruntime",
    "zmq",
    "bxi_example_py_elf3.inference.sonic",
]

failed = False
for name in required:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        location = getattr(module, "__file__", "unknown")
        print(f"[OK] import {name}: {version} ({location})")
    except Exception as exc:
        failed = True
        print(f"[FAIL] import {name}: {exc!r}")

sys.exit(1 if failed else 0)
PY
then
  ok "controller Python dependency check passed"
else
  fail "controller Python dependency check failed; bxi_example_py_elf3_demo may exit before controlling the robot"
fi

section "PICO Python dependencies"
if "${SCRIPT_DIR}/check_sonic_pico_python.sh"; then
  ok "PICO Python dependency check passed"
else
  fail "PICO Python dependency check failed"
fi

section "Current ROS state"
if timeout 3 ros2 topic echo --once /hardware/state_machine_info >"${STATE_OUTPUT}" 2>"${STATE_ERROR}"; then
  ok "/hardware/state_machine_info is publishing"
  sed -n '1,8p' "${STATE_OUTPUT}"
else
  warn "no /hardware/state_machine_info sample within 3s; T1 may not be running"
  sed -n '1,8p' "${STATE_ERROR}"
fi

section "Processes"
if pgrep -af 'bxi_example_py_elf3_demo|hardware_elf3|remote_controller|sonic_pico_runtime_supervisor|pico_manager_legacy|pico_pose_to_smpl_ref_bridge|RoboticsServiceProcess'; then
  ok "runtime processes listed above"
else
  warn "no matching runtime processes"
fi

section "Ports"
if ss -lntup 2>/dev/null | grep -E ':(5556|5557|60061|8081)\b'; then
  ok "SONIC/PICO-related ports listed above"
else
  warn "no 5556/5557/60061/8081 listeners; this is expected before SONIC/PICO starts"
fi

section "Observed port 8081"
if port_8081="$(ss -lntup 2>/dev/null | grep -E ':8081\b' || true)" && [[ -n "${port_8081}" ]]; then
  echo "${port_8081}"
  warn "recording 8081 ownership for diagnostics; it is not a deployment blocker by itself"
else
  ok "8081 is currently free"
fi

section "Result"
if (( FAILURES == 0 )); then
  ok "robot SONIC runtime checks passed"
else
  fail "${FAILURES} blocking check(s) failed"
fi

exit "${FAILURES}"
