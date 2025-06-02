#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

cd "$ROOT_DIR" 

# ====================================================

#set -e

PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
echo "PYTHON_ENV: $PYTHON_ENV"

print_blue '================================================================'
print_blue "Installing python packages for semanatics ..."

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter

# detect and configure CUDA 
. cuda_config.sh



# HACK
pip3 install "$ROOT_DIR"/thirdparty/opencv-python/opencv*.whl --force-reinstall

# semantic mapping packages
# Torchvision will be installed already
# Install transformers with specific version due to break of last version
pip3 install transformers==4.38.2 --constraint "$ROOT_DIR/constraints.txt" # originally suggested version 4.51.0  
 # Install f3rm for the dense CLIP model
pip3 install f3rm  --constraint "$ROOT_DIR/constraints.txt"
pip3 install timm==1.0.15 --constraint "$ROOT_DIR/constraints.txt"