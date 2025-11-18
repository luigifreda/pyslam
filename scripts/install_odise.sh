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


if [ ! -d "odise" ]; then
    echo "Cloning ODISE..."
    git clone https://github.com/NVlabs/ODISE.git odise
    cd odise
    git apply ../odise.patch
    cd ..

    pip install --no-build-isolation ./odise/
    
    pip install --no-build-isolation "xformers>=0.0.16"


    # install back kornia target version since the required stable-diffusion-sdkit requires kornia=0.6.0 and downgrades it 
    pip install --upgrade kornia==0.8.2
fi

cd $ROOT_DIR
