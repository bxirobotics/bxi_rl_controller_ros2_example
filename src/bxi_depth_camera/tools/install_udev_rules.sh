#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="bxi_depth_camera"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Install the bundled Orbbec and Intel RealSense udev rules.

Usage:
  install-udev-rules
  install_udev_rules.sh

The script supports both the bxi_depth_camera source tree and its ROS 2
install space. It asks sudo for privilege when not already running as root,
reloads udev, and retriggers connected camera devices.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

case "$(uname -m)" in
  x86_64 | amd64)
    PLATFORM="linux-x86_64"
    ;;
  aarch64 | arm64)
    PLATFORM="linux-aarch64"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 2
    ;;
esac

SOURCE_PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_VENDOR_ROOT="${SOURCE_PACKAGE_DIR}/vendor/cpp/${PLATFORM}"
INSTALL_PREFIX="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_VENDOR_ROOT="${INSTALL_PREFIX}/share/${PROJECT_NAME}/vendor"

if [[ -f "${SOURCE_VENDOR_ROOT}/orbbec_sdk_v2/shared/99-obsensor-libusb.rules" &&
      -f "${SOURCE_VENDOR_ROOT}/realsense2/shared/60-librealsense2-udev-rules.rules" ]]; then
  ORBBEC_RULE="${SOURCE_VENDOR_ROOT}/orbbec_sdk_v2/shared/99-obsensor-libusb.rules"
  REALSENSE_RULE="${SOURCE_VENDOR_ROOT}/realsense2/shared/60-librealsense2-udev-rules.rules"
  REALSENSE_MIPI_RULE="${SOURCE_VENDOR_ROOT}/realsense2/shared/60-librealsense2-mipi-udev-rules.rules"
elif [[ -f "${INSTALL_VENDOR_ROOT}/orbbec_sdk_v2/99-obsensor-libusb.rules" &&
        -f "${INSTALL_VENDOR_ROOT}/realsense2/60-librealsense2-udev-rules.rules" ]]; then
  ORBBEC_RULE="${INSTALL_VENDOR_ROOT}/orbbec_sdk_v2/99-obsensor-libusb.rules"
  REALSENSE_RULE="${INSTALL_VENDOR_ROOT}/realsense2/60-librealsense2-udev-rules.rules"
  REALSENSE_MIPI_RULE="${INSTALL_VENDOR_ROOT}/realsense2/60-librealsense2-mipi-udev-rules.rules"
else
  echo "Cannot find the bundled Orbbec and RealSense udev rules." >&2
  echo "Checked source bundle: ${SOURCE_VENDOR_ROOT}" >&2
  echo "Checked install bundle: ${INSTALL_VENDOR_ROOT}" >&2
  echo "Build and source bxi_depth_camera, or run this script from its source tree." >&2
  exit 1
fi

for command_name in install udevadm; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done

if ((EUID == 0)); then
  AS_ROOT=()
elif command -v sudo >/dev/null 2>&1; then
  AS_ROOT=(sudo)
else
  echo "Root privilege is required, but sudo is not installed." >&2
  exit 2
fi

install_rule() {
  local source_path="$1"
  local destination_name="$2"
  echo "Installing ${destination_name}"
  "${AS_ROOT[@]}" install -m 0644 -- "${source_path}" \
    "/etc/udev/rules.d/${destination_name}"
}

install_rule "${ORBBEC_RULE}" "99-obsensor-libusb.rules"
install_rule "${REALSENSE_RULE}" "60-librealsense2-udev-rules.rules"
if [[ -f "${REALSENSE_MIPI_RULE}" ]]; then
  install_rule "${REALSENSE_MIPI_RULE}" \
    "60-librealsense2-mipi-udev-rules.rules"
fi

echo "Reloading udev rules"
"${AS_ROOT[@]}" udevadm control --reload-rules

for subsystem in usb iio hidraw video4linux; do
  if [[ -d "/sys/class/${subsystem}" || -d "/sys/bus/${subsystem}" ]]; then
    echo "Retriggering ${subsystem} devices"
    "${AS_ROOT[@]}" udevadm trigger \
      --subsystem-match="${subsystem}" --action=add
  fi
done
"${AS_ROOT[@]}" udevadm settle

echo
echo "Installed Orbbec and RealSense udev rules successfully."
echo "Unplug and reconnect each camera before starting the camera node."
