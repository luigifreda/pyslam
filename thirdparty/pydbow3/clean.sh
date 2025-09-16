#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}
cd modules/dbow3
if [ -d "build" ]; then
    rm -Rf build
fi
if [ -d "install" ]; then
    rm -Rf install
fi  

cd ${SCRIPT_DIR}
if [ -d "build" ]; then
    rm -Rf build
fi
if [ -d "lib" ]; then
    rm -Rf lib
fi