#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
# if [[ -d "$OpenCV_DIR" ]]; then
#     EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
# fi 

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
