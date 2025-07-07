#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

cd "$ROOT_DIR" 

# ====================================================

#set -e

print_blue '================================================================'
print_blue "Installing python packages for semanatics ..."

# Check if conda is installed
if command -v conda &> /dev/null; then
    CONDA_INSTALLED=true
else
    CONDA_INSTALLED=false
fi

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
else
    PIXI_ACTIVATED=false
fi


PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
echo "PYTHON_ENV: $PYTHON_ENV"

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter

# detect and configure CUDA 
. "$ROOT_DIR"/cuda_config.sh


# # HACK
if [ -f "$ROOT_DIR"/thirdparty/opencv-python/opencv*.whl ]; then
    pip3 install "$ROOT_DIR"/thirdparty/opencv-python/opencv*.whl --force-reinstall
fi

# semantic mapping packages
# Torchvision will be installed already
# Install transformers with specific version due to break of last version
pip install transformers==4.38.2 --constraint "$ROOT_DIR/constraints.txt" # originally suggested version 4.51.0  
 # Install f3rm for the dense CLIP model
pip install f3rm  --constraint "$ROOT_DIR/constraints.txt"
pip install timm==1.0.15 --constraint "$ROOT_DIR/constraints.txt"
#pip install protobuf==3.20.3 --force-reinstall

# HACK
shopt -s nullglob
FILES=("$ROOT_DIR"/thirdparty/opencv-python/opencv*.whl)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .whl files found in $ROOT_DIR/thirdparty/opencv-python"
else
    print_blue "Installing opencv-python file: $FILES"
    pip install "$FILES" --force-reinstall
    pip install "numpy<2"
fi
