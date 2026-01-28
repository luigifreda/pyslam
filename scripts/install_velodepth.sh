#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."
SCRIPTS_DIR="$SCRIPT_DIR_"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================


cd $ROOT_DIR/thirdparty


if [ ! -d "velodepth" ]; then
    echo "Cloning velodepth..."
    git clone https://github.com/lpiccinelli-eth/velodepth velodepth

    cd velodepth
    git checkout fcb9a899ba6fdb960f8ff5fb3ad81d0ac22f49b2

    git apply ../velodepth.patch

    pip install --no-build-isolation -e .
fi


cd $ROOT_DIR
