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


cd $ROOT_DIR/thirdparty

if [ ! -d "rf_detr" ]; then
    echo "Cloning rf_detr..."
    git clone https://github.com/roboflow/rf-detr.git rf_detr

    cd rf_detr
    git checkout fd1295b8ccacba0fad2b4a40c8a35b67bc62c335
    git apply ../rf_detr.patch

    # Pin transformers/peft to compatible APIs used by rf_detr
    pip install --upgrade "transformers>=4.41.2" "peft==0.10.0"

    # install rf_detr as a package
    pip install -e .

    cd ..
fi

cd $ROOT_DIR
