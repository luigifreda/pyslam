#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install -y libsuitesparse-dev libeigen3-dev python3-dev
fi   

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# Allow overriding Python executable (e.g., conda env); resolved early so we can query its arch
PYTHON_EXE=${Python3_EXECUTABLE:-$(which python3)}
echo "PYTHON_EXE: $PYTHON_EXE"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    TARGET_ARCH=${CMAKE_OSX_ARCHITECTURES:-${OSX_ARCH:-${PYTHON_ARCH:-$(uname -m)}}}
    echo "TARGET_ARCH: $TARGET_ARCH"

    export CONDA_PREFIX=/Users/luigi/miniconda/envs/pyslam
    export CMAKE_PREFIX_PATH=$CONDA_PREFIX
    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
    export CMAKE_IGNORE_PATH=/opt/homebrew
    export CSPARSE_INCLUDE_DIR=$CONDA_PREFIX/include/suitesparse
    export CSPARSE_LIBRARY=$CONDA_PREFIX/lib/libcxsparse.dylib
    export CHOLMOD_INCLUDE_DIR=$CONDA_PREFIX/include/suitesparse
    export CHOLMOD_LIBRARY=$CONDA_PREFIX/lib/libcholmod.dylib

    # Respect user-requested architecture (e.g., build x86_64 when running under Rosetta)
    MAC_OPTIONS="\
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_OSX_ARCHITECTURES=${TARGET_ARCH}\
    -DCHOLMOD_INCLUDE_DIR=$CHOLMOD_INCLUDE_DIR -DCHOLMOD_LIBRARY=$CHOLMOD_LIBRARY \
    -DCSPARSE_INCLUDE_DIR=$CSPARSE_INCLUDE_DIR -DCSPARSE_LIBRARY=$CSPARSE_LIBRARY \
    -DOPENGL_opengl_LIBRARY= -DOPENGL_glx_LIBRARY="

    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

EXTERNAL_OPTIONS+=" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython3_EXECUTABLE=${PYTHON_EXE}" 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

BUILD_TYPE="Release"
echo "BUILD_TYPE: $BUILD_TYPE"

make_dir build
cd build
cmake .. $EXTERNAL_OPTIONS $MAC_OPTIONS -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j 8

cd ..
