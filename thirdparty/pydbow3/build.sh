#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

cd modules/dbow3
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/modules/dbow3/install
make -j8
make install

cd ${SCRIPT_DIR}
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
