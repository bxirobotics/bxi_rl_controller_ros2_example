#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirement.txt"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/usr/bin/python3}"
CHECK_ONLY=false

usage() {
    cat <<'EOF'
Usage: deploy_environment.sh [OPTIONS]

Install the project's base Python dependencies from requirement.txt located
next to this script. The script can be invoked from any working directory.

Options:
  --python PATH  Python interpreter to use (default: /usr/bin/python3;
                 can also be set with PYTHON_EXECUTABLE)
  --check        Do not install; only validate Python/pip and run pip check
  -h, --help     Show this help message
EOF
}

while (($# > 0)); do
    case "$1" in
        --python)
            if (($# < 2)); then
                echo "error: --python requires an interpreter path" >&2
                usage >&2
                exit 2
            fi
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -r "$REQUIREMENTS_FILE" ]]; then
    echo "error: requirements file not found or unreadable: $REQUIREMENTS_FILE" >&2
    exit 1
fi

if [[ "$PYTHON_EXECUTABLE" == */* ]]; then
    if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
        echo "error: Python interpreter is not executable: $PYTHON_EXECUTABLE" >&2
        exit 1
    fi
else
    RESOLVED_PYTHON="$(command -v -- "$PYTHON_EXECUTABLE" || true)"
    if [[ -z "$RESOLVED_PYTHON" ]]; then
        echo "error: Python interpreter was not found: $PYTHON_EXECUTABLE" >&2
        exit 1
    fi
    PYTHON_EXECUTABLE="$RESOLVED_PYTHON"
fi

if ! "$PYTHON_EXECUTABLE" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
    PYTHON_VERSION="$($PYTHON_EXECUTABLE -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo unknown)"
    echo "error: Python 3.10 is required, but $PYTHON_EXECUTABLE is $PYTHON_VERSION" >&2
    exit 1
fi

if ! "$PYTHON_EXECUTABLE" -m pip --version >/dev/null 2>&1; then
    echo "error: pip is unavailable for $PYTHON_EXECUTABLE" >&2
    echo "Install python3-pip, then run this script again." >&2
    exit 1
fi

check_environment() {
    if ! "$PYTHON_EXECUTABLE" -m pip check; then
        echo "warning: pip found conflicts in the interpreter's full environment." >&2
        echo "The conflicts may come from Ubuntu/ROS packages outside requirement.txt." >&2
    fi
}

echo "Python:       $PYTHON_EXECUTABLE"
echo "Requirements: $REQUIREMENTS_FILE"

if [[ "$CHECK_ONLY" == true ]]; then
    echo "Check-only mode: dependency installation skipped."
    check_environment
    exit 0
fi

"$PYTHON_EXECUTABLE" -m pip install --upgrade -r "$REQUIREMENTS_FILE" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
check_environment

echo "Python environment deployment completed."
