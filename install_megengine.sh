#!/usr/bin/env bash
# Author: Luigi Freda 

#set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Installing megengine from source"
print_blue '================================================'

#echo ROOT_DIR: $ROOT_DIR
cd "$ROOT_DIR"  # from bash_utils.sh

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")



cd thirdparty
if [ ! -d megengine ]; then
    git clone --recursive https://github.com/MegEngine/MegEngine.git megengine
    cd megengine
    git checkout v1.13.4
    cd ..
fi

cd megengine

# Install Prerequisites
./third_party/prepare.sh
./third_party/install-mkl.sh

CUDA_OPTION=""
export CUDA_VERSION="cuda-11.8"  # must be an installed CUDA path in "/usr/local"; 
                                 # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
if [ ! -d /usr/local/$CUDA_VERSION ]; then
    CUDA_VERSION="cuda"  # use last installed CUDA path in standard path as a fallback 
    export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export CUDADIR=/usr/local/$CUDA_VERSION
    CUDA_OPTION="-c"
fi 

# Build MegEngine

if [ -d /usr/local/$CUDA_VERSION ]; then
    
    print_blue "Using CUDA: /usr/local/cuda/bin"
    export CUDA_ROOT_DIR=$CUDADIR

    # Check for cuDNN installation
    if [ -d "$CUDADIR/include/cudnn.h" ]; then
        export CUDNN_ROOT_DIR=$CUDADIR
        print_blue "Using cuDNN: $CUDADIR"
    else
        echo "cuDNN not found. Please install cuDNN."
    fi

    # Check for TensorRT installation
    if [ -d "/usr/local/tensorrt" ]; then
        export TRT_ROOT_DIR="/usr/local/tensorrt"
        print_blue "Using TensorRT: /usr/local/tensorrt"
    else
        echo "TensorRT not found. Please install TensorRT."
    fi
else
    echo "CUDA not found. Please install CUDA."
fi

./scripts/cmake-build/host_build.sh -d $CUDA_OPTION