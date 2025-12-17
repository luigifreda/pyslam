#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

cd $SCRIPT_DIR


echo "Building OpenCV Type Casters Tests..."
echo "=================================="
./build.sh
echo "=================================="
echo "Building completed!"


echo "Running OpenCV Type Casters Tests..."
echo "=================================="

# Check if library exists
if [ ! -f "lib/cvcasters_test.cpython-311-x86_64-linux-gnu.so" ]; then
    echo "ERROR: Library not found. Please build first with ./build.sh"
    exit 1
fi

# Run tests
# Disable auto-loading external pytest plugins from the user environment to
# avoid unrelated failures (e.g., torchtyping with incompatible torch).
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_casters.py -v --tb=short

echo "=================================="
echo "Tests completed!"