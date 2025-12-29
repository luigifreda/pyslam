#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."


cd $ROOT_DIR/thirdparty
if [ ! -d "fast3r" ]; then
    git clone https://github.com/facebookresearch/fast3r
    cd fast3r
    git checkout 33104d4b5b8df43795ecded236194958bbdac572 
    git apply ../fast3r.patch

    pip install -r requirements.txt
    # install fast3r as a package
    pip install -e .


fi
