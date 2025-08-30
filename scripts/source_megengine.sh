#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


# Start `Python3 ` with env for support `MegEngine` after build: `PYTHONPATH=imperative/python:$PYTHONPATH python3 `
# Start `Python3 ` with env for support `MegEngineLite` after build: `PYTHONPATH=lite/pylite:$PYTHONPATH python3 `


if [ -d thirdparty/megengine/imperative/python ]; then
    PYTHONPATH=imperative/python:$PYTHONPATH python3 
fi

if [ -d thirdparty/megengine/lite/pylite ]; then
    PYTHONPATH=lite/pylite:$PYTHONPATH python3 
fi

cd "$STARTING_DIR"

