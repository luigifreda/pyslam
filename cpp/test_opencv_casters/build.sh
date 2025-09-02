#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used


function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
    CONDA_INSTALLED=false
fi


# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$SCRIPT_DIR/../../thirdparty/opencv/install/lib/cmake/opencv4"
echo "OpenCV_DIR: $OpenCV_DIR"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

export CONDA_OPTIONS=""
if [ "$CONDA_INSTALLED" = true ]; then
    CONDA_OPTIONS="-DOPENGL_opengl_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenGL.so \
        -DOPENGL_glx_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLX.so"
    echo "Using CONDA_OPTIONS for build: $CONDA_OPTIONS"
fi

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

make_dir build
cd build
cmake .. $EXTERNAL_OPTIONS $CONDA_OPTIONS
make -j 4

cd ..
