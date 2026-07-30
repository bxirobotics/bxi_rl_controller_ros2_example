#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_SOURCE="${1:-/tmp/elf3_sonic_runtime_deps_20260716}"
OUTPUT_ROOT="${2:-/tmp}"

die() {
  echo "[bundle-prepare] ERROR: $*" >&2
  exit 2
}

[[ -d "${RUNTIME_SOURCE}/wheels" ]] || die "missing ${RUNTIME_SOURCE}/wheels"
[[ -d "${RUNTIME_SOURCE}/xrt" ]] || die "missing ${RUNTIME_SOURCE}/xrt"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

cd "${REPO_ROOT}"
[[ -z "$(git status --porcelain)" ]] || die "repository must be clean before bundling"
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
COMMIT_TIME="$(git show -s --format=%ct HEAD)"
BRANCH="$(git branch --show-current)"
[[ -n "${BRANCH}" ]] || BRANCH="detached-head"
BUNDLE_NAME="elf3_sonic_deploy_${SHORT_COMMIT}_ubuntu22_amd64"
BUNDLE_DIR="${OUTPUT_ROOT}/${BUNDLE_NAME}"
ARCHIVE="${OUTPUT_ROOT}/${BUNDLE_NAME}.tgz"

[[ ! -e "${BUNDLE_DIR}" ]] || die "output exists: ${BUNDLE_DIR}"
[[ ! -e "${ARCHIVE}" ]] || die "output exists: ${ARCHIVE}"
mkdir -p "${BUNDLE_DIR}/source" "${BUNDLE_DIR}/runtime/wheels" "${BUNDLE_DIR}/runtime/xrt"

echo "[bundle-prepare] exporting source commit ${COMMIT}"
# The robot build selects only bxi_example_py_elf3 and remote_controller.
# Export those packages, the deployment scripts, and their licence/provenance
# material instead of copying the entire example repository.  In particular,
# the top-level CAD resources/ tree is not consumed by either selected package
# and accounted for roughly half of the source snapshot size.
SOURCE_PATHS=(
  ASSET_PROVENANCE.md
  README.md
  README.en.md
  THIRD_PARTY_NOTICES.md
  docs
  script
  src/bxi_example_py_elf3
  src/remote_controller
  third_party
)
if git cat-file -e HEAD:LICENSE 2>/dev/null; then
  SOURCE_PATHS+=(LICENSE)
fi
for path in "${SOURCE_PATHS[@]}"; do
  git cat-file -e "HEAD:${path}" 2>/dev/null || die "required source path is absent from HEAD: ${path}"
done
git archive HEAD -- "${SOURCE_PATHS[@]}" | tar -x -C "${BUNDLE_DIR}/source"
printf '%s\n' "${COMMIT}" > "${BUNDLE_DIR}/source/SONIC_DEPLOY_COMMIT"

one_file() {
  local directory="$1"
  local pattern="$2"
  local matches=()
  mapfile -t matches < <(find "${directory}" -maxdepth 1 -type f -name "${pattern}" -print | sort)
  if (( ${#matches[@]} != 1 )); then
    die "expected exactly one file matching ${pattern}, found ${#matches[@]}"
  fi
  printf '%s\n' "${matches[0]}"
}

WHEEL_PATTERNS=(
  'pip-23.0.1-*.whl'
  'setuptools-79.0.1-*.whl'
  'numpy-1.26.4-*.whl'
  'scipy-1.15.3-*.whl'
  'pyzmq-27.1.0-*.whl'
  'msgpack-1.1.2-*.whl'
  'torch-2.6.0+cpu-*.whl'
  'typing_extensions-4.16.0-*.whl'
  'filelock-3.29.4-*.whl'
  'fsspec-2026.6.0-*.whl'
  'networkx-3.4.2-*.whl'
  'jinja2-3.1.6-*.whl'
  'markupsafe-3.0.3-*.whl'
  'sympy-1.13.1-*.whl'
  'mpmath-1.3.0-*.whl'
  'cmeel-0.60.1-*.whl'
  'cmeel_assimp-5.4.3.1-*.whl'
  'cmeel_boost-1.83.0-*.whl'
  'cmeel_console_bridge-1.0.2.3-*.whl'
  'cmeel_octomap-1.10.0-*.whl'
  'cmeel_qhull-8.0.2.1-*.whl'
  'cmeel_tinyxml-2.6.2.3-*.whl'
  'cmeel_urdfdom-3.1.1.1-*.whl'
  'cmeel_zlib-1.3.2-*.whl'
  'eigenpy-3.5.1-*.whl'
  'hpp_fcl-2.4.4-*.whl'
  'pin-2.7.0-*.whl'
  'onnxruntime-1.23.2-*.whl'
  'coloredlogs-15.0.1-*.whl'
  'humanfriendly-10.0-*.whl'
  'flatbuffers-25.12.19-*.whl'
  'packaging-26.2-*.whl'
  'protobuf-7.35.1-*.whl'
)

for pattern in "${WHEEL_PATTERNS[@]}"; do
  file="$(one_file "${RUNTIME_SOURCE}/wheels" "${pattern}")"
  cp -a "${file}" "${BUNDLE_DIR}/runtime/wheels/"
done

for pattern in \
  'roboticsservice_*_amd64.deb' \
  'xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so'; do
  file="$(one_file "${RUNTIME_SOURCE}/xrt" "${pattern}")"
  cp -a "${file}" "${BUNDLE_DIR}/runtime/xrt/"
done

SONIC_RUNTIME_VALIDATE_ONLY=1 \
bash "${BUNDLE_DIR}/source/script/install_robot_sonic_runtime_offline.sh" \
  "${BUNDLE_DIR}/runtime"

printf '%s\n' \
  'ELF3 SONIC offline deployment bundle' \
  "commit=${COMMIT}" \
  "branch=${BRANCH}" \
  'platform=Ubuntu 22.04 x86_64 Python 3.10' \
  "created=$(date --iso-8601=seconds)" \
  > "${BUNDLE_DIR}/BUNDLE_INFO.txt"

(
  cd "${BUNDLE_DIR}"
  find source runtime -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum BUNDLE_INFO.txt
) > "${BUNDLE_DIR}/MANIFEST.sha256"

echo "[bundle-prepare] creating ${ARCHIVE}"
tar \
  --sort=name \
  --mtime="@${COMMIT_TIME}" \
  --owner=0 --group=0 --numeric-owner \
  -C "${OUTPUT_ROOT}" \
  -czf "${ARCHIVE}" \
  "${BUNDLE_NAME}"

(
  cd "${OUTPUT_ROOT}"
  sha256sum "$(basename "${ARCHIVE}")"
) > "${ARCHIVE}.sha256"
echo "[bundle-prepare] bundle=${ARCHIVE}"
echo "[bundle-prepare] checksum=${ARCHIVE}.sha256"
