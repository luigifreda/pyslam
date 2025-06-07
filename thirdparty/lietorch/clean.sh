#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

STARTING_DIR=`pwd`
cd $SCRIPT_DIR

ROOT_DIR="$SCRIPT_DIR"/../..

if [ -d build ]; then
    echo "Removing build directory..."
    rm -Rf build
fi
if [ -d lib ]; then
    echo "Removing lib directory..."
    rm -Rf lib
fi

cd "$STARTING_DIR"