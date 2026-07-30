#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${1:-${SONIC_OFFLINE_RUNTIME_DIR:-}}"
PICO_VENV="${PICO_VENV:-/home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop}"
CONTROLLER_PREFIX="${CONTROLLER_PREFIX:-/opt/bxi/bxi_rl_controller_ros2_example}"
XRT_SERVICE_DIR="${SONIC_XRT_SERVICE_DIR:-/opt/apps/roboticsservice}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
BXI_ROS_SETUP="${BXI_ROS_SETUP:-/opt/bxi/bxi_ros2_pkg/setup.bash}"

die() {
  echo "[runtime-install] ERROR: $*" >&2
  exit 2
}

if [[ -z "${RUNTIME_DIR}" ]]; then
  die "usage: $0 /path/to/runtime"
fi
RUNTIME_DIR="$(cd "${RUNTIME_DIR}" && pwd)"
WHEEL_DIR="${RUNTIME_DIR}/wheels"
XRT_DIR="${RUNTIME_DIR}/xrt"

[[ "$(uname -m)" == "x86_64" ]] || die "runtime bundle requires x86_64"
[[ -d "${WHEEL_DIR}" ]] || die "missing wheel directory: ${WHEEL_DIR}"
[[ -d "${XRT_DIR}" ]] || die "missing XRT directory: ${XRT_DIR}"
command -v python3 >/dev/null 2>&1 || die "python3 is not installed"

python3 - <<'PY' || die "runtime bundle requires Python 3.10"
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY

if [[ "${EUID}" -eq 0 ]] || [[ "${SONIC_NO_SUDO:-0}" == "1" ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

one_file() {
  local directory="$1"
  local pattern="$2"
  local label="$3"
  local matches=()
  mapfile -t matches < <(find "${directory}" -maxdepth 1 -type f -name "${pattern}" -print | sort)
  if (( ${#matches[@]} != 1 )); then
    die "expected exactly one ${label} matching ${pattern}, found ${#matches[@]}"
  fi
  printf '%s\n' "${matches[0]}"
}

wheel() {
  one_file "${WHEEL_DIR}" "$1" "$1 wheel"
}

XRT_DEB="$(one_file "${XRT_DIR}" 'roboticsservice_*_amd64.deb' 'RoboticsService deb')"
XRT_PYTHON="$(one_file "${XRT_DIR}" 'xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so' 'XRoboToolkit Python extension')"
BOOTSTRAP_PIP="$(wheel 'pip-23.0.1-*.whl')"
BOOTSTRAP_SETUPTOOLS="$(wheel 'setuptools-79.0.1-*.whl')"

PICO_WHEELS=(
  "$(wheel 'numpy-1.26.4-*.whl')"
  "$(wheel 'scipy-1.15.3-*.whl')"
  "$(wheel 'pyzmq-27.1.0-*.whl')"
  "$(wheel 'msgpack-1.1.2-*.whl')"
  "$(wheel 'torch-2.6.0+cpu-*.whl')"
  "$(wheel 'typing_extensions-4.16.0-*.whl')"
  "$(wheel 'filelock-3.29.4-*.whl')"
  "$(wheel 'fsspec-2026.6.0-*.whl')"
  "$(wheel 'networkx-3.4.2-*.whl')"
  "$(wheel 'jinja2-3.1.6-*.whl')"
  "$(wheel 'markupsafe-3.0.3-*.whl')"
  "$(wheel 'sympy-1.13.1-*.whl')"
  "$(wheel 'mpmath-1.3.0-*.whl')"
  "$(wheel 'cmeel-0.60.1-*.whl')"
  "$(wheel 'cmeel_assimp-5.4.3.1-*.whl')"
  "$(wheel 'cmeel_boost-1.83.0-*.whl')"
  "$(wheel 'cmeel_console_bridge-1.0.2.3-*.whl')"
  "$(wheel 'cmeel_octomap-1.10.0-*.whl')"
  "$(wheel 'cmeel_qhull-8.0.2.1-*.whl')"
  "$(wheel 'cmeel_tinyxml-2.6.2.3-*.whl')"
  "$(wheel 'cmeel_urdfdom-3.1.1.1-*.whl')"
  "$(wheel 'cmeel_zlib-1.3.2-*.whl')"
  "$(wheel 'eigenpy-3.5.1-*.whl')"
  "$(wheel 'hpp_fcl-2.4.4-*.whl')"
  "$(wheel 'pin-2.7.0-*.whl')"
)

CONTROLLER_WHEELS=(
  "$(wheel 'numpy-1.26.4-*.whl')"
  "$(wheel 'pyzmq-27.1.0-*.whl')"
  "$(wheel 'onnxruntime-1.23.2-*.whl')"
  "$(wheel 'coloredlogs-15.0.1-*.whl')"
  "$(wheel 'humanfriendly-10.0-*.whl')"
  "$(wheel 'flatbuffers-25.12.19-*.whl')"
  "$(wheel 'packaging-26.2-*.whl')"
  "$(wheel 'protobuf-7.35.1-*.whl')"
  "$(wheel 'sympy-1.13.1-*.whl')"
  "$(wheel 'mpmath-1.3.0-*.whl')"
)

if [[ "${SONIC_RUNTIME_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "[runtime-install] offline payload is complete and version-unique"
  exit 0
fi

if [[ ! -x "${XRT_SERVICE_DIR}/RoboticsServiceProcess" ]] || \
   [[ ! -r "${XRT_SERVICE_DIR}/SDK/x64/libPXREARobotSDK.so" ]]; then
  echo "[runtime-install] installing RoboticsService from ${XRT_DEB}"
  "${SUDO[@]}" dpkg -i "${XRT_DEB}"
else
  echo "[runtime-install] RoboticsService already present"
fi

if [[ ! -x "${PICO_VENV}/bin/python" ]]; then
  echo "[runtime-install] creating PICO venv at ${PICO_VENV}"
  mkdir -p "$(dirname "${PICO_VENV}")"
  python3 -m venv --without-pip "${PICO_VENV}" || die "failed to create Python venv"
fi
PICO_PYTHON="${PICO_VENV}/bin/python"

if ! env -u PYTHONPATH "${PICO_PYTHON}" -m pip --version >/dev/null 2>&1; then
  echo "[runtime-install] bootstrapping pip from the offline wheel"
  PYTHONPATH="${BOOTSTRAP_PIP}" "${PICO_PYTHON}" -m pip install \
    --no-index --no-deps "${BOOTSTRAP_PIP}" "${BOOTSTRAP_SETUPTOOLS}" || \
    die "offline pip bootstrap failed"
fi

echo "[runtime-install] installing pinned PICO dependencies"
env -u PYTHONPATH "${PICO_PYTHON}" -m pip install \
  --no-index --no-deps --force-reinstall "${PICO_WHEELS[@]}"

PICO_SITE="$(env -u PYTHONPATH "${PICO_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
install -m 0755 "${XRT_PYTHON}" "${PICO_SITE}/"

CONTROLLER_SITE="${CONTROLLER_PREFIX}/lib/python3.10/site-packages"
[[ -d "${CONTROLLER_SITE}" ]] || \
  die "controller install is missing at ${CONTROLLER_SITE}; deploy the ROS package first"

echo "[runtime-install] installing pinned controller Python dependencies"
"${SUDO[@]}" env -u PYTHONPATH "${PICO_PYTHON}" -m pip install \
  --no-index --no-deps --upgrade --target "${CONTROLLER_SITE}" \
  "${CONTROLLER_WHEELS[@]}"

echo "[runtime-install] validating PICO venv"
env -u PYTHONPATH \
  LD_LIBRARY_PATH="${XRT_SERVICE_DIR}/SDK/x64:${XRT_SERVICE_DIR}:${LD_LIBRARY_PATH:-}" \
  "${PICO_PYTHON}" - <<'PY'
import importlib

expected = {
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "zmq": "27.1.0",
    "msgpack": "1.1.2",
    "torch": "2.6.0+cpu",
    "eigenpy": "3.5.1",
    "hppfcl": "2.4.4",
    "pinocchio": "2.7.0",
}
for name, wanted in expected.items():
    module = importlib.import_module(name)
    found = getattr(module, "__version__", "unknown")
    if found != wanted:
        raise RuntimeError(f"{name}: expected {wanted}, found {found}")
    print(f"[OK] {name}={found}")
importlib.import_module("xrobotoolkit_sdk")
print("[OK] xrobotoolkit_sdk")
PY

source_setup() {
  local file="$1"
  [[ -f "${file}" ]] || die "missing setup file: ${file}"
  set +u
  # shellcheck disable=SC1090
  source "${file}"
  set -u
}

source_setup "${ROS_SETUP}"
source_setup "${BXI_ROS_SETUP}"
source_setup "${CONTROLLER_PREFIX}/setup.bash"

echo "[runtime-install] validating controller Python environment"
python3 - <<'PY'
import importlib

for name in ["numpy", "onnxruntime", "zmq", "bxi_example_py_elf3.inference.sonic"]:
    module = importlib.import_module(name)
    print(
        f"[OK] {name}: "
        f"{getattr(module, '__version__', 'unknown')} "
        f"({getattr(module, '__file__', 'unknown')})"
    )
PY

echo "[runtime-install] done"
