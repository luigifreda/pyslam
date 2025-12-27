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
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -Dpybind11_DIR="$PROJECT_ROOT/thirdparty/pybind11/share/cmake/pybind11" \
    -DOpenCV_DIR="$PROJECT_ROOT/thirdparty/opencv/install/lib/cmake/opencv4"  \
    -DWITH_MARCH_NATIVE="$WITH_MARCH_NATIVE" $PIXI_PYTHON_CMAKE_OPTS

# Build the module
echo "Building the module..."
make -j$(nproc)

# The module is already built to the correct location by CMake
echo "Module built successfully!"
echo "Module location: $CPP_DIR/lib/cpp_core*.so"

#echo "Testing the module..."
#cd "$PROJECT_ROOT"
#python3 $CPP_DIR/test/test_module.py
