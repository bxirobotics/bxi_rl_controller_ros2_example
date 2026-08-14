#!/usr/bin/env bash
set -euo pipefail

ORBBEC_VERSION="2.9.3"
ORBBEC_COMMIT="2f6561c28255d805b34aa00a690199ce40e96c81"
REALSENSE_VERSION="2.57.7"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

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

OUTPUT_DIR="${1:-${PACKAGE_DIR}/vendor/cpp/${PLATFORM}}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists: ${OUTPUT_DIR}" >&2
  echo "Move or remove that exact platform directory before rebuilding it." >&2
  exit 2
fi

for command_name in cmake git g++ make sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing build command: ${command_name}" >&2
    exit 2
  fi
done

WORK_DIR="$(mktemp -d -t bxi-camera-sdk-bundle.XXXXXXXX)"
cleanup() {
  if [[ -n "${WORK_DIR:-}" && -d "${WORK_DIR}" ]]; then
    rm -rf -- "${WORK_DIR}"
  fi
}
trap cleanup EXIT

ORBBEC_SOURCE="${WORK_DIR}/OrbbecSDK_v2"
ORBBEC_BUILD="${WORK_DIR}/orbbec-build"
ORBBEC_STAGE="${WORK_DIR}/orbbec-stage"
REALSENSE_SOURCE="${WORK_DIR}/librealsense"
REALSENSE_BUILD="${WORK_DIR}/realsense-build"
REALSENSE_STAGE="${WORK_DIR}/realsense-stage"
BUNDLE_STAGE="${WORK_DIR}/${PLATFORM}"

git clone --branch "v${ORBBEC_VERSION}" --depth 1 \
  https://github.com/orbbec/OrbbecSDK_v2.git "${ORBBEC_SOURCE}"
if [[ "$(git -C "${ORBBEC_SOURCE}" rev-parse HEAD)" != "${ORBBEC_COMMIT}" ]]; then
  echo "Orbbec v${ORBBEC_VERSION} no longer resolves to the pinned commit." >&2
  exit 1
fi

cmake -S "${ORBBEC_SOURCE}" -B "${ORBBEC_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ORBBEC_STAGE}" \
  -DOB_BUILD_EXAMPLES=OFF \
  -DOB_BUILD_TESTS=OFF \
  -DOB_BUILD_TOOLS=OFF \
  -DOB_BUILD_DOCS=OFF \
  -DOB_INSTALL_EXAMPLES_SOURCE=OFF \
  -DOB_INSTALL_LICENSES=ON \
  -DOB_BUILD_NET_PAL=OFF \
  -DOB_BUILD_GMSL_PAL=OFF \
  -DOB_BUILD_MAIN_PROJECT=ON
cmake --build "${ORBBEC_BUILD}" --parallel "$(nproc)"
cmake --install "${ORBBEC_BUILD}"

git clone --branch "v${REALSENSE_VERSION}" --depth 1 \
  https://github.com/IntelRealSense/librealsense.git "${REALSENSE_SOURCE}"
cmake -S "${REALSENSE_SOURCE}" -B "${REALSENSE_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${REALSENSE_STAGE}" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_WITH_CUDA=OFF \
  -DFORCE_RSUSB_BACKEND=ON
cmake --build "${REALSENSE_BUILD}" --parallel "$(nproc)"
cmake --install "${REALSENSE_BUILD}"

ORBBEC_LIBRARY="$(find "${ORBBEC_STAGE}" -type f \
  -name "libOrbbecSDK.so.${ORBBEC_VERSION}" -print -quit)"
REALSENSE_LIBRARY="$(find "${REALSENSE_STAGE}" -type f \
  -name "librealsense2.so.${REALSENSE_VERSION}" -print -quit)"
if [[ -z "${ORBBEC_LIBRARY}" || -z "${REALSENSE_LIBRARY}" ]]; then
  echo "SDK install did not produce the expected versioned libraries." >&2
  exit 1
fi

ORBBEC_LIB_DIR="$(dirname -- "${ORBBEC_LIBRARY}")"
REALSENSE_LIB_DIR="$(dirname -- "${REALSENSE_LIBRARY}")"
mkdir -p \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/include" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/lib" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/shared" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/licenses" \
  "${BUNDLE_STAGE}/realsense2/include" \
  "${BUNDLE_STAGE}/realsense2/lib" \
  "${BUNDLE_STAGE}/realsense2/shared" \
  "${BUNDLE_STAGE}/realsense2/licenses"

cp -a "${ORBBEC_STAGE}/include/libobsensor" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/include/"
cp -a "${ORBBEC_LIB_DIR}/libOrbbecSDK.so"* \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/lib/"
cp -a "${ORBBEC_LIB_DIR}/extensions" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/lib/"
cp -a "${ORBBEC_LIB_DIR}/OrbbecSDKConfig.xml" \
  "${ORBBEC_LIB_DIR}/OrbbecSDKConfig.md" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/lib/"
cp -a "${ORBBEC_SOURCE}/scripts/env_setup/99-obsensor-libusb.rules" \
  "${ORBBEC_SOURCE}/scripts/env_setup/install_udev_rules.sh" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/shared/"
cp -a "${ORBBEC_SOURCE}/LICENSE.txt" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/licenses/OrbbecSDK_v2_LICENSE.txt"
cp -a "${ORBBEC_SOURCE}/extensions/license.txt" \
  "${BUNDLE_STAGE}/orbbec_sdk_v2/licenses/Orbbec_extensions_LICENSE.txt"

cp -a "${REALSENSE_STAGE}/include/librealsense2" \
  "${BUNDLE_STAGE}/realsense2/include/"
cp -a "${REALSENSE_LIB_DIR}/librealsense2.so"* \
  "${BUNDLE_STAGE}/realsense2/lib/"
cp -a "${REALSENSE_SOURCE}/config/99-realsense-libusb.rules" \
  "${BUNDLE_STAGE}/realsense2/shared/60-librealsense2-udev-rules.rules"
cp -a "${REALSENSE_SOURCE}/LICENSE" \
  "${BUNDLE_STAGE}/realsense2/licenses/Apache-2.0.txt"

ORBBEC_SHA="$(sha256sum "${BUNDLE_STAGE}/orbbec_sdk_v2/lib/libOrbbecSDK.so.${ORBBEC_VERSION}" | awk '{print $1}')"
REALSENSE_SHA="$(sha256sum "${BUNDLE_STAGE}/realsense2/lib/librealsense2.so.${REALSENSE_VERSION}" | awk '{print $1}')"
{
  printf '# bxi_depth_camera SDK bundle\n\n'
  printf -- '- Platform: `%s`\n' "${PLATFORM}"
  printf -- '- Orbbec SDK: `OrbbecSDK_v2 v%s`\n' "${ORBBEC_VERSION}"
  printf -- '- Orbbec source commit: `%s`\n' "${ORBBEC_COMMIT}"
  printf -- '- Orbbec library SHA-256: `%s`\n' "${ORBBEC_SHA}"
  printf -- '- librealsense: `v%s`\n' "${REALSENSE_VERSION}"
  printf -- '- librealsense library SHA-256: `%s`\n' "${REALSENSE_SHA}"
  printf '\nBuilt natively on `%s` with `FORCE_RSUSB_BACKEND=ON`.\n' "$(uname -m)"
  printf 'glibc, libstdc++, libudev and libusb come from the target OS.\n'
} >"${BUNDLE_STAGE}/MANIFEST.md"

mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
mv "${BUNDLE_STAGE}" "${OUTPUT_DIR}"
echo "Created ${OUTPUT_DIR}"
