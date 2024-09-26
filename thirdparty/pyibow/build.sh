#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}
cd modules/obindex2/lib
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release 
make -j8


cd ${SCRIPT_DIR}
cd modules/ibow-lcd
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release 
make -j8


cd ${SCRIPT_DIR}
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
