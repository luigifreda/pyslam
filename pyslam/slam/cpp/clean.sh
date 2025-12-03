#!/bin/bash

# Build script for PYSLAM C++ Core Module
# This script builds the cpp_core pybind11 module

set -e  # Exit on any error

echo "Cleaning PYSLAM C++ Core Module..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CPP_DIR="$SCRIPT_DIR"

cd "$SCRIPT_DIR"

if [ -d build ]; then
    rm -rf build
    rm -rf lib
fi
