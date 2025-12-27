#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

# ====================================================

# NOTE: ROS2 libraries are built with system libstdc++, 
# so they may have ABI compatibility issues in pixi or conda environments 
# (which use their own libstdc++). Building is disabled in these environments.

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
    echo "Pixi environment detected: $PIXI_PROJECT_NAME"
    echo "Pixi environment is not supported for building ros2_pybindings"

    exit 1
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    CONDA_INSTALLED=true
    echo "Conda is installed"
    echo "Conda is not supported for building ros2_pybindings"
    exit 1
fi

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
# Check if OpenCV_DIR was already set in the external options
if [[ ! "$EXTERNAL_OPTIONS" == *"OpenCV_DIR"* ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

. $SCRIPT_DIR/find_ros.sh

echo "ROS2_INSTALL_PATH: $ROS2_INSTALL_PATH"

export ROS2_OPTIONS=""
if [[ "$ROS2_INSTALL_PATH" == *"jazzy"* ]]; then
    echo "Building with ROS2 Jazzy"
    export ROS2_OPTIONS="-DWITH_JAZZY=ON"
    echo "ROS2_OPTIONS: $ROS2_OPTIONS"
fi
if [[ "$ROS2_INSTALL_PATH" == *"rolling"* ]]; then
    echo "Building with ROS2 Rolling"
    export ROS2_OPTIONS="-DWITH_ROLLING=ON"
    echo "ROS2_OPTIONS: $ROS2_OPTIONS"
fi

if [ -n "$ROS2_INSTALL_PATH" ]; then

    source $ROS2_INSTALL_PATH/setup.bash

    make_dir build
    cd build
    cmake .. $EXTERNAL_OPTIONS $ROS2_OPTIONS
    make -j 4
    cd ..
fi
