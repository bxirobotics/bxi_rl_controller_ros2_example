#!/usr/bin/env bash
set -u

PYTHON_BIN="${1:-${SONIC_PICO_PYTHON:-}}"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in \
    /home/bxi/bxi_rl_controller_ros2_example-main/.venv_teleop/bin/python \
    /home/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python \
    /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/.venv_teleop/bin/python \
    /opt/bxi/bxi_rl_controller_ros2_example/.venv_teleop/bin/python \
    python3; do
    if command -v "${candidate}" >/dev/null 2>&1 || [[ -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[FAIL] no Python interpreter found"
  exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FAIL] Python interpreter is not executable: ${PYTHON_BIN}"
  exit 2
fi

echo "[INFO] PICO Python: ${PYTHON_BIN}"

XRT_SERVICE_DIR="${SONIC_XRT_SERVICE_DIR:-/opt/apps/roboticsservice}"
PRECHECK_FAILED=0
for candidate in \
  "${XRT_SERVICE_DIR}/SDK/x64" \
  "${XRT_SERVICE_DIR}" \
  "${XRT_SERVICE_DIR}/lib"; do
  if [[ -d "${candidate}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *:"${candidate}":*) ;;
      *) export LD_LIBRARY_PATH="${candidate}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
done

if [[ -r "${XRT_SERVICE_DIR}/SDK/x64/libPXREARobotSDK.so" ]]; then
  echo "[OK] libPXREARobotSDK.so exists"
else
  echo "[FAIL] missing ${XRT_SERVICE_DIR}/SDK/x64/libPXREARobotSDK.so"
  PRECHECK_FAILED=1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = [
    "numpy",
    "scipy",
    "zmq",
    "msgpack",
    "torch",
    "xrobotoolkit_sdk",
    "eigenpy",
    "hppfcl",
    "pinocchio",
]
failed = False

for name in required:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] import {name}: {version}")
        if name == "numpy":
            major = int(str(version).split(".", 1)[0])
            if major >= 2:
                failed = True
                print(
                    "[FAIL] numpy must be <2 for the bundled pin/eigenpy wheels; "
                    f"found {version}"
                )
    except Exception as exc:
        failed = True
        print(f"[FAIL] import {name}: {exc!r}")

try:
    import torch
    print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()}")
except Exception:
    pass

sys.exit(1 if failed else 0)
PY
PY_STATUS=$?

if [[ -x "${XRT_SERVICE_DIR}/RoboticsServiceProcess" ]]; then
  echo "[OK] RoboticsServiceProcess exists"
else
  echo "[FAIL] missing ${XRT_SERVICE_DIR}/RoboticsServiceProcess"
  PRECHECK_FAILED=1
fi

if (( PRECHECK_FAILED != 0 )); then
  exit 1
fi
exit "${PY_STATUS}"
