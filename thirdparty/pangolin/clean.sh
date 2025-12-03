#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used


if [ -d "build" ]; then
    rm -rf build
fi
if [ -d "lib" ]; then
    rm -rf lib
fi
if [ -d "bin" ]; then
    rm -rf bin
fi
if [ -f pypangolin.cpython-*.so ]; then
    rm pypangolin.cpython-*.so 
fi