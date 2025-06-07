#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

STARTING_DIR=`pwd`
cd $SCRIPT_DIR

ROOT_DIR="$SCRIPT_DIR"/../..

. $ROOT_DIR/cuda_config.sh

# Instead of building with setup.py use cmake and ninja
# pip install . --verbose

sudo apt update
sudo apt install ninja-build

# Create build folder and run cmake
if [ ! -d build ]; then
    mkdir build
fi
cd build

# Get how many cores are available
cores=$(grep -c ^processor /proc/cpuinfo)
# Use half of them 
cores=$((cores/2))

echo "Building with $cores cores"

CUDA_OPTIONS=""
if [ "$CUDA_VERSION" != "0" ]; then
    CUDA_OPTIONS="-DCMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER"
fi

# Use ninja if available 
if command -v ninja >/dev/null 2>&1; then
    cmake -G Ninja .. $CUDA_OPTIONS
    # launch parallel build with 8 threads
    ninja -j$cores
    sudo $(which ninja) install   # "sudo ninja install" does not work!
else 
    echo "ninja not found, falling back to make"    
    cmake .. $CUDA_OPTIONS
    make -j$cores
    sudo make install  
fi

cd "$STARTING_DIR"