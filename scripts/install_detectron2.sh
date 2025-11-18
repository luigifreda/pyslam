#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."


cd $ROOT_DIR/thirdparty
if [ ! -d "detectron2" ]; then
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    git checkout v0.6

    git apply ../detectron2.patch
    
    # install detectron2
    pip install --no-build-isolation -e .

fi
