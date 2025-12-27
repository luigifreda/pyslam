#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
ROOT_DIR="$SCRIPT_DIR/.."
SCRIPTS_DIR="$ROOT_DIR/scripts"

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

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
else
    PIXI_ACTIVATED=false
fi

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$SCRIPT_DIR/../thirdparty/opencv/install/lib/cmake/opencv4"
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

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================


BUILD_TYPE="Release"
#BUILD_TYPE="Debug"

echo "BUILD_TYPE: $BUILD_TYPE"

make_dir build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $EXTERNAL_OPTIONS $CONDA_OPTIONS $MAC_OPTIONS 
make -j 4

cd ..
