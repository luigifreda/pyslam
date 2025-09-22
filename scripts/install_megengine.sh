#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


print_blue '================================================'
print_blue "Installing megengine from source"
print_blue '================================================'

PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")


# Download MegEngine source code and build it from source
cd thirdparty
if [ ! -d megengine ]; then
    git clone --recursive https://github.com/MegEngine/MegEngine.git megengine
    cd megengine
    git checkout v1.13.4
    git apply $ROOT_DIR/thirdparty/megengine.patch

    # Install Prerequisites
    ./third_party/prepare.sh
    ./third_party/install-mkl.sh

    cd ..
fi

cd megengine

export CUDA_OPTION=""
. "$ROOT_DIR"/cuda_config.sh
if [ "$CUDA_VERSION" != "0" ]; then
    #export CUDA_OPTION="-c"   # we need a more complex detection with cuDNN and TensorRT discovery

    export CUDA_ROOT_DIR=$CUDA_HOME
fi

echo "CUDA_OPTION: $CUDA_OPTION"

cd $ROOT_DIR/thirdparty/megengine
EXTRA_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo" ./scripts/cmake-build/host_build.sh -d $CUDA_OPTION

cd $ROOT_DIR/thirdparty/megengine/scripts/whl/manylinux2014
./build_image.sh

./build_wheel_common.sh -sdk cpu # build the megengine wheel for cpu

# Search for any MegEngine wheel file in the cpu directory
WHEEL_DIR="$ROOT_DIR/thirdparty/megengine/scripts/whl/manylinux2014/output/wheelhouse/cpu"
WHEEL_FILE=$(find "$WHEEL_DIR" -name "MegEngine-*.whl" | head -1)

if [ -n "$WHEEL_FILE" ] && [ -f "$WHEEL_FILE" ]; then 
    echo "Found MegEngine wheel: $WHEEL_FILE"
    pip install "$WHEEL_FILE"
else
    echo "MegEngine wheel not found in $WHEEL_DIR. Something went wrong. Please build it first."
    exit 1
fi

cd "$STARTING_DIR"




# export CUDA_OPTION=""
# export CUDA_VERSION="cuda"  # must be an installed CUDA path in "/usr/local"; 
#                            # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
# if [ ! -d "/usr/local/$CUDA_VERSION" ]; then
#     CUDA_VERSION="cuda"  # use last installed CUDA path in standard path as a fallback 
#     export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}
#     export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#     export CUDADIR=/usr/local/$CUDA_VERSION
#     CUDA_OPTION="-c"
# fi 

# # Build MegEngine

# if [ -d /usr/local/$CUDA_VERSION ]; then
    
#     print_blue "Using CUDA: /usr/local/cuda/bin"
#     export CUDA_ROOT_DIR=$CUDADIR

#     # Check for cuDNN installation
#     if [ -d "$CUDADIR/include/cudnn.h" ]; then
#         export CUDNN_ROOT_DIR=$CUDADIR
#         print_blue "Using cuDNN: $CUDADIR"
#     else
#         echo "cuDNN not found. Please install cuDNN."
#     fi

#     # Check for TensorRT installation
#     if [ -d "/usr/local/tensorrt" ]; then
#         export TRT_ROOT_DIR="/usr/local/tensorrt"
#         print_blue "Using TensorRT: /usr/local/tensorrt"
#     else
#         echo "TensorRT not found. Please install TensorRT."
#     fi
# else
#     echo "CUDA not found. Please install CUDA."
# fi