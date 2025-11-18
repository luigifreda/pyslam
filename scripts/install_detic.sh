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


if [ ! -d "detic" ]; then
    echo "Cloning detic..."
    git clone https://github.com/facebookresearch/Detic.git detic --recurse-submodules
    
    # patch Detic
    cd detic
    git apply ../detic.patch

    # patch CenterNet2
    cd third_party/CenterNet2/
    git apply ../centernet2.patch  # this patch is contained in detic.patch
    cd ../../

    cd ..

    # install detic requirements (opencv removed)
    pip install mss timm dataclasses ftfy regex fasttext scikit-learn lvis nltk
    pip install git+https://github.com/openai/CLIP.git
fi

if [ ! -d "detic/models" ]; then
    mkdir -p detic/models
fi

if [ ! -f "detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth" ]; then
    cd detic
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
fi

cd $ROOT_DIR
