#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

cd "$ROOT_DIR" 

# ====================================================

#set -x
#set -e

PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
echo "PYTHON_ENV: $PYTHON_ENV"

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

print_blue '================================================'
print_blue "Configuring and installing torch packages ..."

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter

# detect and configure CUDA 
. "$ROOT_DIR"/cuda_config.sh


if [ "$OSTYPE" == darwin* ]; then
    pip install torch==2.1           # torch==2.2.0 causes some segmentation faults on mac
    pip install torchvision==0.16         
else

    # previous versions
    #TARGET_TORCH_VERSION="2.2.0"
    #TARGET_TORCHVISION_VERSION="0.17"

    # current target versions
    TARGET_TORCH_VERSION="2.9.1"
    TARGET_TORCHVISION_VERSION="0.24.1"

    TORCH_CUDA_VERSION=0
    if [ "$CUDA_VERSION" != "0" ]; then
        #TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        TORCH_CUDA_VERSION=$(get_installed_torch_cuda_ver)
    fi

    INSTALL_CUDA_SPECIFIC_TORCH=false
    if [[ "$CUDA_VERSION" != "0" && "$TORCH_CUDA_VERSION" != "$CUDA_VERSION" && "$CONDA_INSTALLED" != true ]]; then
        INSTALL_CUDA_SPECIFIC_TORCH=true
    fi

    print_blue "CUDA_VERSION: $CUDA_VERSION, TORCH_CUDA_VERSION: $TORCH_CUDA_VERSION"
    print_blue "INSTALL_CUDA_SPECIFIC_TORCH: $INSTALL_CUDA_SPECIFIC_TORCH"

    if $INSTALL_CUDA_SPECIFIC_TORCH; then
        print_green "System CUDA_VERSION is $CUDA_VERSION but the detected TORCH CUDA version is $TORCH_CUDA_VERSION."
        print_green "Installing torch==$TARGET_TORCH_VERSION+cu${CUDA_VERSION_STRING_COMPACT} and torchvision==$TARGET_TORCHVISION_VERSION+cu${CUDA_VERSION_STRING_COMPACT}"
        pip3 install torch=="$TARGET_TORCH_VERSION+cu${CUDA_VERSION_STRING_COMPACT}" \
            torchvision=="$TARGET_TORCHVISION_VERSION+cu${CUDA_VERSION_STRING_COMPACT}" \
            --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_STRING_COMPACT}   
        
        # check if last command was ok  (in the case we don't find the CUDA-specific torch version)
        if [[ $? -ne 0 ]]; then
            print_yellow "WARNING: Failed to install CUDA-specific torch and torchvision. Installing default versions."
            pip install torch==$TARGET_TORCH_VERSION
            pip install torchvision==$TARGET_TORCHVISION_VERSION
        fi
    else
        print_green "Installing torch==$TARGET_TORCH_VERSION and torchvision==$TARGET_TORCHVISION_VERSION"
        pip install torch==$TARGET_TORCH_VERSION
        pip install torchvision==$TARGET_TORCHVISION_VERSION
    fi             
fi 

pip install "numpy<2"

