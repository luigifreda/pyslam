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
print_blue "Installing python packages for semantics ..."

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


CONSTRAINTS=""
if [ -f "$ROOT_DIR/constraints.txt" ]; then
    CONSTRAINTS="--constraint $ROOT_DIR/constraints.txt"
fi

# semantic mapping packages
# Torchvision will be installed already
# Install transformers with specific version due to break of last version
pip install transformers==4.38.2 $CONSTRAINTS # originally suggested version 4.51.0

# Install f3rm for the dense CLIP model
pip install "fpsample<0.3.0"  # required by f3rm without upgrading pybind11 version
pip install f3rm $CONSTRAINTS

pip install timm==1.0.15 $CONSTRAINTS
#pip install protobuf==3.20.3 --force-reinstall


print_blue '================================================================'
print_blue "Installing python packages for detectron2 ..."
# Install detectron2
$SCRIPT_DIR_/install_detectron2.sh

print_blue '================================================================'
print_blue "Installing python packages for EOV-Seg ..."
# Install EOV-Seg
$SCRIPT_DIR_/install_eov_seg.sh

print_blue '================================================================'
print_blue "Installing python packages for Detic ..."
# Install Detic
$SCRIPT_DIR_/install_detic.sh

print_blue '================================================================'
print_blue "Installing python packages for ODISE ..."
# Install ODISE
$SCRIPT_DIR_/install_odise.sh

# HACK
shopt -s nullglob
FILES=("$ROOT_DIR"/thirdparty/opencv-python/opencv*.whl)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .whl files found in $ROOT_DIR/thirdparty/opencv-python"
else
    print_blue "Installing opencv-python file: $FILES"
    pip install "$FILES" --force-reinstall
fi

# Install supported numpy version <2 to avoid conflicts
pip install "numpy<2"
