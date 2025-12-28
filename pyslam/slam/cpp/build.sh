#!/bin/bash

# Build script for PYSLAM C++ Core Module
# This script builds the cpp_core pybind11 module

set -e  # Exit on any error

echo "Building PYSLAM C++ Core Module..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
CPP_DIR="$SCRIPT_DIR"

# ====================================================

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
    CONDA_INSTALLED=false
fi

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
    echo "Pixi environment detected: $PIXI_PROJECT_NAME"

    source "$SCRIPTS_DIR/pixi_cuda_config.sh"
    source "$SCRIPTS_DIR/pixi_python_config.sh"

    PIXI_PYTHON_CMAKE_OPTS=$(get_pixi_python_cmake_options)
    echo "PIXI_PYTHON_CMAKE_OPTS: $PIXI_PYTHON_CMAKE_OPTS"
else
    PIXI_ACTIVATED=false
fi

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$PROJECT_ROOT/thirdparty/opencv/install/lib/cmake/opencv4"
echo "OpenCV_DIR: $OpenCV_DIR"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

export CONDA_OPTIONS=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ "$CONDA_INSTALLED" = true ]; then
        CONDA_OPTIONS="-DOPENGL_opengl_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenGL.so \
            -DOPENGL_glx_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLX.so"
        echo "Using CONDA_OPTIONS for build: $CONDA_OPTIONS"
    fi
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++"

    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

# Create build directory
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

BUILD_TYPE="Release"
#BUILD_TYPE="Debug"

echo "BUILD_TYPE: $BUILD_TYPE"

# check if WITH_MARCH_NATIVE is not set
if [[ -z "$WITH_MARCH_NATIVE" ]]; then
    WITH_MARCH_NATIVE=ON
fi

# Configure with CMake
echo "Configuring with CMake..."

# Ensure we use the correct Python executable from the current environment
PYTHON_EXEC=$(which python)
if [[ -n "$PYTHON_EXEC" ]]; then
    PYTHON_CMAKE_OPTS="-DPython_EXECUTABLE=$PYTHON_EXEC"
    echo "Using Python executable: $PYTHON_EXEC"
else
    PYTHON_CMAKE_OPTS=""
fi

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -Dpybind11_DIR="$PROJECT_ROOT/thirdparty/pybind11/share/cmake/pybind11" \
    -DWITH_MARCH_NATIVE="$WITH_MARCH_NATIVE" $PYTHON_CMAKE_OPTS $PIXI_PYTHON_CMAKE_OPTS $EXTERNAL_OPTIONS

# Build the module
echo "Building the module..."
make -j$(nproc)

# The module is already built to the correct location by CMake
echo "Module built successfully!"
echo "Module location: $CPP_DIR/lib/cpp_core*.so"

#echo "Testing the module..."
#cd "$PROJECT_ROOT"
#python3 $CPP_DIR/test/test_module.py
