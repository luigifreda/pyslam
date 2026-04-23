#!/usr/bin/env bash

set -euo pipefail

# Resolve the script directory even when entered through a symlink.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BUILD_DIR="$SCRIPT_DIR/build"
OPENCV_DIR="$SCRIPT_DIR/../../thirdparty/opencv/install/lib/cmake/opencv4"
PYTHON_BIN="${PYTHON_BIN:-}"

cmake_args=("$@")

if ((${#cmake_args[@]} > 0)); then
    echo "Extra CMake args: ${cmake_args[*]}"
fi

if [[ -d "$OPENCV_DIR" ]]; then
    echo "Using OpenCV_DIR: $OPENCV_DIR"
    cmake_args+=("-DOpenCV_DIR=$OPENCV_DIR")
else
    echo "OpenCV_DIR not found at: $OPENCV_DIR"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "ERROR: neither 'python' nor 'python3' was found in PATH"
        exit 1
    fi
fi

# Resolve launcher/shim paths (for example pyenv shims) to the actual interpreter executable
# that owns the matching headers and libpython.
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if [[ -n "$PYTHON_BIN" ]]; then
    echo "Using Python executable: $PYTHON_BIN"
    # pybind11/cvcasters_test and the native benchmarks must be configured against the same
    # interpreter to avoid import/runtime mismatches.
    cmake_args+=(
        "-DPython3_EXECUTABLE=$PYTHON_BIN"
        "-DPYTHON_EXECUTABLE=$PYTHON_BIN"
    )
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
