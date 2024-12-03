#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

cd modules/dbow2
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTIONS
make -j8

cd ${SCRIPT_DIR}
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTIONS
make -j8
