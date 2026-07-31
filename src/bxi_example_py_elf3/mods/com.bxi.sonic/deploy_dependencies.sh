#!/usr/bin/env bash

set -Eeuo pipefail

readonly MOD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REQUIREMENTS_FILE="${MOD_ROOT}/requirements-pico.txt"
readonly VENDOR_LICENSE_DIR="${MOD_ROOT}/vendor/licenses"
readonly XRT_SERVICE_LICENSE="${VENDOR_LICENSE_DIR}/XRoboToolkit-PC-Service.LICENSE"
readonly XRT_SERVICE_NOTICE="${VENDOR_LICENSE_DIR}/XRoboToolkit-PC-Service.THIRD_PARTY_NOTICE.txt"
readonly XRT_SERVICE_PROVENANCE="${VENDOR_LICENSE_DIR}/XRoboToolkit-PC-Service.PROVENANCE.txt"
readonly XRT_SERVICE_CONFIG="${MOD_ROOT}/config/roboticsservice-minimal.ini"

runtime_platform_tag() {
    local os_name machine
    os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"
    machine="$(uname -m | tr '[:upper:]' '[:lower:]')"
    case "$machine" in
        amd64) machine="x86_64" ;;
        arm64) machine="aarch64" ;;
    esac
    printf '%s-%s\n' "$os_name" "$machine"
}

readonly RUNTIME_PLATFORM_TAG="$(runtime_platform_tag)"
readonly BUNDLED_SERVICE_DIR="${MOD_ROOT}/runtime/${RUNTIME_PLATFORM_TAG}/roboticsservice"
readonly VENDOR_PYTHON_ROOT="${MOD_ROOT}/vendor/python"
readonly DEFAULT_MOD_RUNTIME="${MOD_ROOT}/.runtime/${RUNTIME_PLATFORM_TAG}/pico"

service_sdk_directory_name() {
    case "$RUNTIME_PLATFORM_TAG" in
        linux-x86_64) printf 'x64\n' ;;
        linux-aarch64) printf 'arm64\n' ;;
        *) die "XRoboToolkit is unsupported on platform: $RUNTIME_PLATFORM_TAG" ;;
    esac
}

mode="check"
check_requested="false"
install_requested="false"
bundle_service_source=""
python_executable=""
venv_directory=""
wheelhouse=""
offline="false"
service_directory="${SONIC_XRT_SERVICE_DIR:-}"

usage() {
    cat <<'EOF'
Deploy or check the external dependencies of the com.bxi.sonic Mod.

Usage:
  ./deploy_dependencies.sh [--check] [options]
  ./deploy_dependencies.sh --install [options]

Modes:
  --check                 Check only; this is the default and changes nothing.
  --install               Install Python dependencies, then run all checks.

Python target:
  --python PATH           Existing Python used for checking or installation.
  --venv DIR              Create/use a virtual environment at DIR. Implies
                          --install; --python selects the base interpreter.
  --mod-runtime           Create/use .runtime/<platform>/pico inside this Mod.
                          Implies --install and is auto-discovered at run time.

Local/offline inputs:
  --wheelhouse DIR        Prefer Python packages from a local wheel directory.
  --offline               Disable package-index access; requires --wheelhouse.

XR service:
  --bundle-service-from DIR
                          Extract the minimal portable service runtime from an
                          existing verified installation into this Mod.
  --service-dir DIR       Explicit RoboticsService root for checking. Default:
                          bundled runtime, then /opt/apps/roboticsservice.
  -h, --help              Show this help.

Examples:
  ./deploy_dependencies.sh --check --python /path/to/python
  ./deploy_dependencies.sh --bundle-service-from /opt/apps/roboticsservice
  ./deploy_dependencies.sh --mod-runtime
  ./deploy_dependencies.sh --mod-runtime --offline \
      --wheelhouse /path/to/wheels

The script intentionally does not call apt, modify systemd, or install
Torch/CUDA. Bundling only copies the service's minimal runtime closure.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

resolve_existing_directory() {
    local value="$1"
    [[ -d "$value" ]] || die "directory does not exist: $value"
    (cd -- "$value" && printf '%s\n' "$PWD")
}

resolve_target_directory() {
    local value="$1"
    local parent
    parent="$(dirname -- "$value")"
    mkdir -p -- "$parent"
    parent="$(cd -- "$parent" && printf '%s\n' "$PWD")"
    printf '%s/%s\n' "$parent" "$(basename -- "$value")"
}

while (($#)); do
    case "$1" in
        --check)
            check_requested="true"
            shift
            ;;
        --install)
            install_requested="true"
            shift
            ;;
        --python)
            (($# >= 2)) || die "--python requires a path"
            python_executable="$2"
            shift 2
            ;;
        --venv)
            (($# >= 2)) || die "--venv requires a directory"
            venv_directory="$2"
            install_requested="true"
            shift 2
            ;;
        --mod-runtime)
            venv_directory="$DEFAULT_MOD_RUNTIME"
            install_requested="true"
            shift
            ;;
        --wheelhouse)
            (($# >= 2)) || die "--wheelhouse requires a directory"
            wheelhouse="$2"
            shift 2
            ;;
        --offline)
            offline="true"
            shift
            ;;
        --service-dir)
            (($# >= 2)) || die "--service-dir requires a directory"
            service_directory="$2"
            shift 2
            ;;
        --bundle-service-from)
            (($# >= 2)) || die "--bundle-service-from requires a directory"
            bundle_service_source="$2"
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

if [[ "$check_requested" == "true" && ( "$install_requested" == "true" || -n "$bundle_service_source" ) ]]; then
    die "--check cannot be combined with an install or bundle operation"
fi
if [[ "$install_requested" == "true" ]]; then
    mode="install"
fi

[[ -f "$REQUIREMENTS_FILE" ]] || die "missing $REQUIREMENTS_FILE"
if [[ "$offline" == "true" && -z "$wheelhouse" ]]; then
    die "--offline requires --wheelhouse"
fi
if [[ -n "$wheelhouse" ]]; then
    wheelhouse="$(resolve_existing_directory "$wheelhouse")"
fi
bundle_service_runtime() {
    local source="$1"
    local target="$BUNDLED_SERVICE_DIR"
    local target_parent stage
    local relative mode sdk_directory dependency_path dependency_soname
    local closure_library_path ldd_output
    local -a executable_files=(RoboticsServiceProcess)
    local -a data_files=(
        libBusiness.so
        libCommonUtils.so
        libDeviceConnectionManager.so
        libPXREAGRPCServer.so
    )
    local -a dependency_queue=()
    local -A inspected_dependencies=()

    sdk_directory="$(service_sdk_directory_name)"
    data_files+=("SDK/${sdk_directory}/libPXREARobotSDK.so")

    source="$(resolve_existing_directory "$source")"
    command -v ldd >/dev/null 2>&1 || die "ldd is required to bundle the service runtime"
    command -v patchelf >/dev/null 2>&1 || die "patchelf is required to normalize the portable service runtime"
    command -v readelf >/dev/null 2>&1 || die "readelf is required to inspect the portable service runtime"
    [[ ! -e "$target" ]] || die "bundled service already exists: $target"
    [[ -f "$XRT_SERVICE_LICENSE" ]] || die "missing XRoboToolkit license: $XRT_SERVICE_LICENSE"
    [[ -f "$XRT_SERVICE_NOTICE" ]] || die "missing XRoboToolkit notices: $XRT_SERVICE_NOTICE"
    [[ -f "$XRT_SERVICE_PROVENANCE" ]] || die "missing XRoboToolkit provenance: $XRT_SERVICE_PROVENANCE"
    [[ -f "$XRT_SERVICE_CONFIG" ]] || die "missing minimal service settings: $XRT_SERVICE_CONFIG"

    target_parent="$(dirname -- "$target")"
    mkdir -p -- "$target_parent"
    stage="$(mktemp -d "${target_parent}/.roboticsservice.tmp.XXXXXX")"
    trap 'rm -rf -- "$stage"' EXIT

    for relative in "${executable_files[@]}"; do
        [[ -f "$source/$relative" ]] || die "source runtime is missing: $source/$relative"
        install -D -m 0755 -- "$source/$relative" "$stage/$relative"
        dependency_queue+=("$source/$relative")
    done
    for relative in "${data_files[@]}"; do
        [[ -f "$source/$relative" ]] || die "source runtime is missing: $source/$relative"
        mode=0644
        install -D -m "$mode" -- "$source/$relative" "$stage/$relative"
        dependency_queue+=("$source/$relative")
    done

    # Resolve the actual ELF closure on the target architecture. Vendor
    # releases use different Qt/ICU versions on x86_64 and ARM64, so copying
    # versioned filenames from one release is neither portable nor stable.
    closure_library_path="${source}/SDK/${sdk_directory}:${source}:${source}/lib"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        closure_library_path="${closure_library_path}:${LD_LIBRARY_PATH}"
    fi
    while ((${#dependency_queue[@]})); do
        dependency_path="${dependency_queue[0]}"
        dependency_queue=("${dependency_queue[@]:1}")
        [[ -z "${inspected_dependencies[$dependency_path]:-}" ]] || continue
        inspected_dependencies["$dependency_path"]=1

        if ! ldd_output="$(LD_LIBRARY_PATH="$closure_library_path" ldd "$dependency_path" 2>&1)"; then
            die "cannot inspect ELF dependencies for $dependency_path: $ldd_output"
        fi
        while read -r dependency_soname _ dependency_path _; do
            [[ -n "$dependency_soname" ]] || continue
            if [[ "$dependency_soname" == "linux-vdso.so."* ]]; then
                continue
            fi
            if [[ "$dependency_path" == "not" ]]; then
                die "unresolved dependency $dependency_soname required by the service runtime"
            fi
            [[ "$dependency_path" == /* ]] || continue
            case "$dependency_soname" in
                ld-linux-*.so.*|libc.so.*|libdl.so.*|libgcc_s.so.*|libm.so.*|\
                libpthread.so.*|libresolv.so.*|librt.so.*|libstdc++.so.*|\
                libutil.so.*)
                    continue
                    ;;
            esac
            if [[ -e "$stage/$dependency_soname" || \
                  -e "$stage/lib/$dependency_soname" || \
                  -e "$stage/SDK/$sdk_directory/$dependency_soname" ]]; then
                continue
            fi
            install -D -m 0644 -- "$dependency_path" "$stage/lib/$dependency_soname"
            dependency_queue+=("$dependency_path")
        done <<< "$ldd_output"
    done

    # Remove release-machine paths embedded by the vendor build. Application
    # libraries search the portable root and its private directories; SDK and
    # dependency libraries only search beside themselves.
    while IFS= read -r -d '' dependency_path; do
        if ! readelf -h "$dependency_path" >/dev/null 2>&1; then
            continue
        fi
        relative="${dependency_path#"$stage"/}"
        case "$relative" in
            SDK/*|lib/*)
                patchelf --set-rpath '$ORIGIN' "$dependency_path"
                ;;
            *)
                patchelf --set-rpath \
                    "\$ORIGIN:\$ORIGIN/lib:\$ORIGIN/SDK/${sdk_directory}" \
                    "$dependency_path"
                ;;
        esac
    done < <(find "$stage" -type f -print0)

    if command -v strip >/dev/null 2>&1; then
        while IFS= read -r -d '' dependency_path; do
            if readelf -h "$dependency_path" >/dev/null 2>&1; then
                strip --strip-unneeded "$dependency_path"
            fi
        done < <(find "$stage" -type f -print0)
    fi

    install -D -m 0644 -- "$XRT_SERVICE_LICENSE" "$stage/LICENSE"
    install -D -m 0644 -- "$XRT_SERVICE_NOTICE" "$stage/THIRD_PARTY_NOTICE.txt"
    install -D -m 0644 -- "$XRT_SERVICE_PROVENANCE" "$stage/PROVENANCE.txt"
    install -D -m 0644 -- "$XRT_SERVICE_CONFIG" "$stage/setting.ini"

    mv -- "$stage" "$target"
    trap - EXIT
    printf 'Bundled minimal RoboticsService runtime: %s\n' "$target"
}

if [[ -n "$bundle_service_source" ]]; then
    bundle_service_runtime "$bundle_service_source"
    service_directory="$BUNDLED_SERVICE_DIR"
fi

if [[ -z "$service_directory" ]]; then
    if [[ -e "/opt/apps/roboticsservice" ]]; then
        service_directory="/opt/apps/roboticsservice"
    else
        service_directory="$BUNDLED_SERVICE_DIR"
    fi
fi

find_default_python() {
    local candidate
    if [[ -n "${SONIC_PICO_PYTHON:-}" ]]; then
        printf '%s\n' "$SONIC_PICO_PYTHON"
        return
    fi
    for candidate in \
        "$DEFAULT_MOD_RUNTIME/bin/python" \
        python3.10 \
        python3; do
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] && printf '%s\n' "$candidate" && return
        elif command -v -- "$candidate" >/dev/null 2>&1; then
            command -v -- "$candidate"
            return
        fi
    done
    return 1
}

resolve_python() {
    local value="$1"
    local resolved
    if [[ "$value" == */* ]]; then
        [[ -x "$value" && -f "$value" ]] || die "Python is not executable: $value"
        (cd -- "$(dirname -- "$value")" && printf '%s/%s\n' "$PWD" "$(basename -- "$value")")
        return
    fi
    resolved="$(command -v -- "$value" 2>/dev/null || true)"
    [[ -n "$resolved" && -x "$resolved" ]] || die "Python was not found: $value"
    printf '%s\n' "$resolved"
}

if [[ -z "$python_executable" ]]; then
    python_executable="$(find_default_python || true)"
    [[ -n "$python_executable" ]] || die "no Python found; pass --python PATH"
fi
python_executable="$(resolve_python "$python_executable")"

clean_python() {
    env \
        -u PYTHONHOME \
        -u PYTHONPATH \
        -u PYTHONEXECUTABLE \
        -u __PYVENV_LAUNCHER__ \
        "$@"
}

if [[ -n "$venv_directory" ]]; then
    venv_directory="$(resolve_target_directory "$venv_directory")"
    if [[ ! -e "$venv_directory" ]]; then
        printf 'Creating virtual environment: %s\n' "$venv_directory"
        clean_python "$python_executable" -m venv "$venv_directory"
    elif [[ ! -x "$venv_directory/bin/python" ]]; then
        die "existing target is not a usable virtual environment: $venv_directory"
    fi
    python_executable="$(resolve_python "$venv_directory/bin/python")"
fi

python_identity="$(clean_python "$python_executable" -E -s -c \
    'import platform, sys; print(f"{platform.python_implementation()} {platform.python_version()} ({sys.executable})")')" \
    || die "cannot execute Python: $python_executable"
python_runtime_tag="$(clean_python "$python_executable" -E -s -c '
import re, sys, sysconfig
safe = lambda value: re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
cache_tag = sys.implementation.cache_tag or f"python-{sys.version_info.major}{sys.version_info.minor}"
print(f"{safe(sysconfig.get_platform())}-{safe(cache_tag)}")
')" || die "cannot determine Python runtime tag: $python_executable"
readonly VENDOR_PYTHON_DIR="${VENDOR_PYTHON_ROOT}/${python_runtime_tag}"
printf 'Python target: %s\n' "$python_identity"

if [[ "$mode" == "install" ]]; then
    clean_python "$python_executable" -E -s -m pip --version >/dev/null 2>&1 || \
        die "pip is unavailable in $python_executable; use a Python build with ensurepip/venv support"

    pip_source_args=()
    if [[ -n "$wheelhouse" ]]; then
        pip_source_args+=(--find-links "$wheelhouse")
    fi
    if [[ "$offline" == "true" ]]; then
        pip_source_args+=(--no-index)
    fi

    printf 'Installing PICO Python requirements from: %s\n' "$REQUIREMENTS_FILE"
    clean_python "$python_executable" -E -s -m pip install \
        --disable-pip-version-check \
        "${pip_source_args[@]}" \
        -r "$REQUIREMENTS_FILE"

fi

resolve_service_sdk_directory() {
    local root="$1"
    local preferred=""
    local candidate
    local -a matches=()

    case "$RUNTIME_PLATFORM_TAG" in
        linux-x86_64) preferred="x64" ;;
        linux-aarch64) preferred="arm64" ;;
    esac
    if [[ -n "$preferred" && -f "$root/SDK/$preferred/libPXREARobotSDK.so" ]]; then
        printf '%s\n' "$preferred"
        return
    fi
    if [[ -d "$root/SDK" ]]; then
        while IFS= read -r -d '' candidate; do
            matches+=("$(basename -- "$(dirname -- "$candidate")")")
        done < <(find "$root/SDK" -mindepth 2 -maxdepth 2 \
            -type f -name libPXREARobotSDK.so -print0)
    fi
    ((${#matches[@]} == 1)) || return 1
    printf '%s\n' "${matches[0]}"
}

service_directory="${service_directory%/}"
service_executable="${service_directory}/RoboticsServiceProcess"
service_sdk_directory="$(resolve_service_sdk_directory "$service_directory" || true)"
service_library=""
if [[ -n "$service_sdk_directory" ]]; then
    service_library="${service_directory}/SDK/${service_sdk_directory}/libPXREARobotSDK.so"
fi
status=0
vendor_bindings=("${VENDOR_PYTHON_DIR}"/xrobotoolkit_sdk*.so)

printf '\nDependency check\n'
if [[ -x "$service_executable" ]]; then
    printf '  OK   RoboticsServiceProcess: %s\n' "$service_executable"
else
    printf '  MISS RoboticsServiceProcess is not executable: %s\n' "$service_executable" >&2
    status=1
fi
if [[ -n "$service_library" && -f "$service_library" ]]; then
    printf '  OK   XR SDK library: %s\n' "$service_library"
else
    printf '  MISS XR SDK library under: %s/SDK\n' "$service_directory" >&2
    status=1
fi

library_path="${service_directory}:${service_directory}/lib"
if [[ -n "$service_sdk_directory" ]]; then
    library_path="${service_directory}/SDK/${service_sdk_directory}:${library_path}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    library_path="${library_path}:${LD_LIBRARY_PATH}"
fi

if [[ -x "$service_executable" ]] && command -v ldd >/dev/null 2>&1; then
    missing_libraries="$(LD_LIBRARY_PATH="$library_path" ldd "$service_executable" 2>&1 | grep 'not found' || true)"
    if [[ -z "$missing_libraries" ]]; then
        printf '  OK   RoboticsService dynamic-library closure\n'
    else
        printf '  MISS RoboticsService unresolved libraries:\n%s\n' "$missing_libraries" >&2
        status=1
    fi
fi

python_imports=(numpy scipy zmq msgpack pinocchio)
for import_name in "${python_imports[@]}"; do
    if LD_LIBRARY_PATH="$library_path" clean_python \
        "$python_executable" -E -s -c \
        'import importlib, sys; importlib.import_module(sys.argv[1])' \
        "$import_name" >/dev/null 2>&1; then
        printf '  OK   Python import: %s\n' "$import_name"
    else
        printf '  MISS Python import: %s (%s)\n' "$import_name" "$python_executable" >&2
        status=1
    fi
done

check_xrt_api() {
    local vendor_path="$1"
    LD_LIBRARY_PATH="$library_path" clean_python \
        "$python_executable" -E -s -c '
import sys
if sys.argv[1]:
    sys.path.insert(0, sys.argv[1])
import xrobotoolkit_sdk as sdk
required = (
    "init", "close", "is_body_data_available", "get_body_joints_pose",
    "get_time_stamp_ns", "get_left_trigger", "get_right_trigger",
    "get_left_grip", "get_right_grip", "get_left_axis", "get_right_axis",
    "get_left_menu_button", "get_A_button", "get_B_button", "get_X_button",
    "get_Y_button",
)
missing = [name for name in required if not callable(getattr(sdk, name, None))]
if missing:
    raise SystemExit("missing API: " + ", ".join(missing))
' "$vendor_path" >/dev/null 2>&1
}

if check_xrt_api ""; then
    printf '  OK   xrobotoolkit_sdk: user environment (%s)\n' "$python_executable"
elif [[ -f "${vendor_bindings[0]}" ]] && check_xrt_api "$VENDOR_PYTHON_DIR"; then
    printf '  OK   xrobotoolkit_sdk: bundled fallback (%s)\n' "${vendor_bindings[0]}"
else
    printf '  MISS xrobotoolkit_sdk is unavailable or incompatible in both the user environment and bundled fallback\n' >&2
    status=1
fi

if ((status != 0)); then
    printf '\nSONIC dependencies are incomplete.\n' >&2
    printf 'Install Python packages with --install and install xrobotoolkit_sdk in the selected environment, or provide a compatible bundled fallback at %s.\n' \
        "$VENDOR_PYTHON_DIR" >&2
    exit "$status"
fi

printf '\nSONIC dependencies are ready.\n'
printf 'Use SONIC_PICO_PYTHON=%s when an explicit interpreter is desired.\n' "$python_executable"
if [[ "$python_executable" == "$DEFAULT_MOD_RUNTIME"/bin/python* ]]; then
    printf 'The Mod will auto-discover this in-Mod runtime after restart.\n'
fi
