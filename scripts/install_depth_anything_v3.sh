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

# Setup environment for detectron2 installation
$SCRIPT_DIR_/install_detectron2.sh

cd $ROOT_DIR/thirdparty


if [ ! -d "depth_anything_v3" ]; then
    echo "Cloning detic..."
    git clone https://github.com/ByteDance-Seed/depth-anything-3 depth_anything_v3

    cd depth_anything_v3

    # tested version ed6989a23cd389e975ed9f7cbd7385396e6d867e
    git checkout ed6989a23cd389e975ed9f7cbd7385396e6d867e

    # apply the patch
    git apply ../depth_anything_v3.patch

    # Install build dependencies required for --no-build-isolation
    pip install "hatchling>=1.25" "hatch-vcs>=0.4" "editables"

    # pip install --no-build-isolation -e . # Basic
    # pip install --no-build-isolation -e ".[gs]" # Gaussians Estimation and Rendering
    # pip install --no-build-isolation -e ".[app]" # Gradio, python>=3.10
    pip install --no-build-isolation -e ".[all]" # ALL
fi


cd $ROOT_DIR
