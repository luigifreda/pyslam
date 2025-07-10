#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."
SCRIPTS_DIR="$SCRIPT_DIR_"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

cd "$ROOT_DIR" 

# ====================================================

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
print_blue "Configuring and installing python packages ..."

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter

# detect and configure CUDA 
. "$ROOT_DIR"/cuda_config.sh


pip3 install --upgrade pip 
pip3 install --upgrade setuptools wheel

pip3 install setuptools==80.8.0

pip3 install ninja

# Install opencv_python from source with non-free modules enabled 
INSTALL_OPENCV_FROM_SOURCE=1
if [ $INSTALL_OPENCV_FROM_SOURCE -eq 1 ]; then
    #NOTE: This procedures is preferable since it avoids issues with Qt linking/configuration
    print_green "Installing opencv_python from source with non-free modules enabled"
    $SCRIPTS_DIR/install_opencv_python.sh
else
    PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
    MAKEFLAGS_OPTION="-j$(nproc)"
    CMAKE_ARGS_OPTION="-DOPENCV_ENABLE_NONFREE=ON" # install nonfree modules

    MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-python -vvv $PRE_OPTION
    MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-contrib-python -vvv $PRE_OPTION
fi

# pip3 packages 
install_pip_package numpy==1.23.5
install_pip_package numpy-quaternion==2023.0.4
install_pip_package gdown  # to download from google drive

install_pip_package PyQt5-sip==12.15.0    # NOTE: This is required by pyqt5. The the next versions of PyQt5-sip require python 3.9.
install_pip_package pyqt5==5.15.11        # version 5.15.11 working under mac
install_pip_package pyqtgraph==0.13.3  

install_pip_package scikit-image==0.21.0 # mac

install_pip_package tensorflow==2.13
install_pip_package tensorflow_hub  # required by VPR
install_pip_package tf_slim==1.1.0

install_pip_package kornia==0.7.3
install_pip_package kornia_moons==0.2.9
install_pip_package importlib_metadata==8.0.0

install_pip_package hjson    # for reading hjson files (https://hjson.github.io/)
install_pip_package jinja2

install_pip_package timm             # ml-depth-pro
if [ "$OSTYPE" == darwin* ]; then
    pip3 install pillow_heif==0.17.0 # ml-depth-pro
else
    install_pip_package pillow_heif # ml-depth-pro
fi

if [ "$OSTYPE" == darwin* ]; then
    install_pip_package torch==2.1           # torch==2.2.0 causes some segmentation faults on mac
    install_pip_package torchvision==0.16         
else
    TORCH_CUDA_VERSION=0
    if [ "$CUDA_VERSION" != "0" ]; then
        TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    fi
    # if [ "$CUDA_VERSION" == "11.8" ]; then
    #     # See also docs/TROUBLESHOOTING.md
    #     # This is to avoid the RuntimeError: "The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.1). Please make sure to use the same CUDA versions."
    #     print_green "Installing torch==2.2.0+cu118 and torchvision==0.17+cu118"
    #     pip3 install torch==2.2.0+cu118 torchvision==0.17+cu118 --index-url https://download.pytorch.org/whl/cu118
    # fi
    INSTALL_CUDA_SPECIFIC_TORCH=false
    if [[ "$CUDA_VERSION" != "0" && "$TORCH_CUDA_VERSION" != "$CUDA_VERSION" && "$CONDA_INSTALLED" != true ]]; then
        INSTALL_CUDA_SPECIFIC_TORCH=true
    fi

    print_blue "CUDA_VERSION: $CUDA_VERSION, TORCH_CUDA_VERSION: $TORCH_CUDA_VERSION"

    if $INSTALL_CUDA_SPECIFIC_TORCH; then
        print_green "System CUDA_VERSION is $CUDA_VERSION but the detected TORCH CUDA version is $TORCH_CUDA_VERSION. Installing torch==2.2.0+cu${CUDA_VERSION_STRING_COMPACT} and torchvision==0.17+cu${CUDA_VERSION_STRING_COMPACT}"
        pip3 install torch=="2.2.0+cu${CUDA_VERSION_STRING_COMPACT}" torchvision=="0.17+cu${CUDA_VERSION_STRING_COMPACT}" --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_STRING_COMPACT}   
        # check if last command was ok  (in the case we don't find the CUDA-specific torch version)
        if [[ $? -ne 0 ]]; then
            print_yellow "WARNING: Failed to install CUDA-specific torch and torchvision. Installing default versions."
            install_pip_package torch==2.2.0
            install_pip_package torchvision==0.17
        fi
    else
        install_pip_package torch==2.2.0
        install_pip_package torchvision==0.17
    fi             
fi 


install_pip_package "rerun-sdk>=0.17.0"

install_pip_package ujson

install_pip_package protobuf==3.20.*    # for delf NN

install_pip_package einops                       # for VLAD
install_pip_package fast-pytorch-kmeans #==0.1.6 # for VLAD

install_pip_package pyflann-py3 # for loop closure database
install_pip_package faiss-cpu # for loop closure database
if [ "$CUDA_VERSION" != "0" ]; then
    install_pip_package faiss-gpu  # for loop closure database on GPU
fi 

if [[ "$OSTYPE" != "darwin"* ]]; then
    install_pip_package open3d
fi

# crestereo
if [[ "$OSTYPE" != "darwin"* ]]; then
    # Unfortunately, megengine is not supported on macOS with arm architecture
    pip3 install --upgrade cryptography pyOpenSSL
    python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html # This brings issues when launched in parallel processes
    #pip3 install megengine  # This brings issues with non-supported CUDA architecture on my machine
fi

if [[ "$OSTYPE" != "darwin"* ]]; then
    # for ROS
    pip3 install pycryptodomex
    pip3 install gnupg
    pip3 install rospkg
fi

pip3 install evo      #==1.11.0
pip3 install trimesh  # for utils_dust3r.py 

# pip3 install jax # not used at the moment 
# if [ "$CUDA_VERSION" != "0" ]; then
#     if [[ "$CUDA_VERSION" =~ ^11\. ]]; then
#         pip3 install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#     fi
#     if [[ "$CUDA_VERSION" =~ ^12\. ]]; then
#         pip3 install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#     fi
# fi

# MonoGS
if [ "$CUDA_VERSION" != "0" ]; then
    # We need cuda for MonoGS

    #MAKEFLAGS_OPTION="-j$(nproc)"
    
    pip3 install munch
    pip3 install wandb
    pip3 install plyfile
    pip3 install glfw
    pip3 install torchmetrics
    pip3 install -U imgviz
    pip3 install PyOpenGL
    pip3 install PyGLM
    pip3 install lpips
    pip3 install rich
    pip3 install ruff
    
    #pip3 install lycon  # removed since it creates install issues under ubuntu 24.04
    
    #sudo apt install -y cuda-command-line-tools-$CUDA_VERSION_STRING_WITH_HYPHENS cuda-libraries-$CUDA_VERSION_STRING_WITH_HYPHENS  # moved to install_system_packages.sh

    cd "$ROOT_DIR"    
    #pip3 install ./thirdparty/lietorch --verbose                              # to clean: $ rm -rf thirdparty/lietorch/build thirdparty/lietorch/*.egg-info

    ./thirdparty/lietorch/build.sh                                             # building with cmake to enable parallel threads (for some reasons, enabling parallel threads in pip3 install fails)
    
    pip3 install ./thirdparty/monogs/submodules/simple-knn                     # to clean: $ rm -rf thirdparty/monogs/submodules/simple-knn/build thirdparty/monogs/submodules/simple-knn/*.egg-info
    pip3 install ./thirdparty/monogs/submodules/diff-gaussian-rasterization    # to clean: $ rm -rf thirdparty/monogs/submodules/diff-gaussian-rasterization/build thirdparty/monogs/submodules/diff-gaussian-rasterization/*.egg-info
else
    print_yellow "Skipping MonoGS since CUDA_VERSION is 0"
fi


# mast3r and mvdust3r
if [ "$CUDA_VERSION" != "0" ]; then
    # We need cuda for mast3r and mvdust3r

    pip3 install gradio 
    pip3 install roma 

    #sudo apt install -y libcusparse-dev-$CUDA_VERSION_STRING_WITH_HYPHENS libcusolver-dev-$CUDA_VERSION_STRING_WITH_HYPHENS # moved to install_system_packages.sh

    #MAKEFLAGS_OPTION="-j$(nproc)"
    
    # to install from source (to avoid linking issue with CUDA)
    pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"    
else 
    print_yellow "Skipping mast3r and mvdust3r since CUDA_VERSION is 0"
fi


# HACK: Moved the install of the semantic tools at the end of the install process to avoid some conflict issues among the deps
# $SCRIPTS_DIR/install_pip3_semantics.sh

