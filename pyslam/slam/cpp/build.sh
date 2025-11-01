#!/bin/bash

# Build script for PYSLAM C++ Core Module
# This script builds the cpp_core pybind11 module

set -e  # Exit on any error

echo "Building PYSLAM C++ Core Module..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CPP_DIR="$SCRIPT_DIR"

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
    -DWITH_MARCH_NATIVE="$WITH_MARCH_NATIVE"

# Build the module
echo "Building the module..."
make -j$(nproc)

# The module is already built to the correct location by CMake
echo "Module built successfully!"
echo "Module location: $CPP_DIR/lib/cpp_core*.so"

#echo "Testing the module..."
#cd "$PROJECT_ROOT"
#python3 $CPP_DIR/test/test_module.py
