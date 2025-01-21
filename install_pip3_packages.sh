#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing python packages ..."


# detect CUDA VERSION
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(get_cuda_version)
    echo CUDA_VERSION: $CUDA_VERSION

    export CUDA_VERSION_STRING="cuda-"${CUDA_VERSION}  # must be an installed CUDA path in "/usr/local"; 
                                                       # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
    if [ ! -d /usr/local/$CUDA_VERSION_STRING ]; then
        CUDA_VERSION_STRING="cuda"  # use last installed CUDA path in standard path as a fallback 
    fi     
    echo CUDA_VERSION_STRING: $CUDA_VERSION_STRING
    export PATH=/usr/local/$CUDA_VERSION_STRING/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION_STRING/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}    
fi


# N.B.: python3 is required!

pip3 install --upgrade pip 
pip3 install --upgrade setuptools wheel


# PIP_MAC_OPTIONS=""
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     PIP_MAC_OPTIONS=" --no-binary :all: "
# fi


# pip3 packages 
install_pip_package pygame==2.6.0 
install_pip_package matplotlib==3.7.5 
install_pip_package pyopengl==3.1.7 
install_pip_package pillow==10.4.0
install_pip_package pybind11==2.13.1 
install_pip_package numpy==1.23.5
install_pip_package pyyaml==6.0.1 
install_pip_package termcolor==2.4.0 
install_pip_package yacs==0.1.8
install_pip_package gdown  # to download from google drive
install_pip_package ordered-set==4.1.0 # from https://pypi.org/project/ordered-set/

install_pip_package numpy-quaternion==2023.0.4
install_pip_package psutil

install_pip_package PyQt5-sip==12.15.0    # NOTE: This is required by pyqt5. The the next versions of PyQt5-sip require python 3.9.
install_pip_package pyqt5==5.15.11        # version 5.15.11 working under mac
install_pip_package pyqtgraph==0.13.3  

INSTALL_OPENCV_FROM_SOURCE=1

# Install opencv_python from source with non-free modules enabled 
if [ $INSTALL_OPENCV_FROM_SOURCE -eq 1 ]; then
    #NOTE: This procedures is preferable since it avoids issues with Qt linking/configuration
    print_green "Installing opencv_python from source with non-free modules enabled"
    ./install_opencv_python.sh
else
    PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
    MAKEFLAGS_OPTION="-j$(nproc)"
    CMAKE_ARGS_OPTION="-DOPENCV_ENABLE_NONFREE=ON" # install nonfree modules

    MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-python -vvv $PRE_OPTION
    MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-contrib-python -vvv $PRE_OPTION
fi

install_pip_package tqdm==4.66.4 
install_pip_package scipy==1.10.1
#install_pip_package scikit-image==0.16.2 # ubuntu 
install_pip_package scikit-image==0.21.0 # mac
install_pip_package seaborn==0.13.2

install_pip_package tensorflow==2.13
install_pip_package tf_slim==1.1.0

install_pip_package kornia==0.7.3
install_pip_package kornia_moons==0.2.9
install_pip_package importlib_metadata==8.0.0

install_pip_package timm        # ml-depth-pro
if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install pillow_heif==0.17.0 # ml-depth-pro
else
    install_pip_package pillow_heif # ml-depth-pro
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    install_pip_package torch==2.1           # torch==2.2.0 causes some segmentation faults on mac
    install_pip_package torchvision==0.16         
else 
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        # See also docs/TROUBLESHOOTING.md
        # This is to avoid the RuntimeError: "The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.1). Please make sure to use the same CUDA versions."
        print_green "Installing torch==2.2.0+cu118 and torchvision==0.17+cu118"
        pip install torch==2.2.0+cu118 torchvision==0.17+cu118 --index-url https://download.pytorch.org/whl/cu118
    else
        install_pip_package torch==2.2.0
        install_pip_package torchvision==0.17
    fi         
fi 

pip install "rerun-sdk>=0.17.0"

install_pip_package ujson
install_pip_package tensorflow_hub  # required for VPR

if command -v nvidia-smi &> /dev/null; then
    install_pip_package faiss-gpu 
else 
    install_pip_package faiss-cpu
fi 

pip install protobuf==3.20.*    # for delf NN
pip install ujson

pip install einops                       # for VLAD
pip install fast-pytorch-kmeans #==0.1.6 # for VLAD
 
pip install pyflann-py3 # for loop closure database
pip install faiss-cpu   # for loop closure database (there is also faiss-gpu)

if [[ "$OSTYPE" != "darwin"* ]]; then
    pip install open3d
fi

# crestereo
if [[ "$OSTYPE" != "darwin"* ]]; then
    # Unfortunately, megengine is not supported on macOS with arm architecture
    pip install --upgrade cryptography pyOpenSSL
    python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html # This brings issues when launched in parallel processes
    #pip install megengine  # This brings issues with non-supported CUDA architecture on my machine
fi

pip install gdown  # to download from google drive


# MonoGS
if command -v nvidia-smi &> /dev/null; then
    # We need cuda for MonoGS

    pip install munch
    pip install wandb
    pip install plyfile
    pip install glfw
    pip install trimesh
    pip install evo    #==1.11.0
    pip install torchmetrics
    pip install imgviz
    pip install PyOpenGL
    pip install PyGLM
    pip install lpips
    pip install rich
    pip install ruff
    
    #pip install lycon  # removed since it creates install issues under ubuntu 24.04
    
    #pip install git+https://github.com/princeton-vl/lietorch.git
    if [[ ! -d "$ROOT_DIR/thirdparty/lietorch" ]]; then
        cd $ROOT_DIR/thirdparty
        git clone --recursive https://github.com/princeton-vl/lietorch.git lietorch
        cd lietorch
        git checkout 0fa9ce8ffca86d985eca9e189a99690d6f3d4df6
        git apply ../lietorch.patch  # added fixes for building under ubuntu 22.04 and 24.04
        cd $ROOT_DIR
    fi
    pip install ./thirdparty/lietorch --verbose                               # to clean: $ rm -rf thirdparty/lietorch/build thirdparty/lietorch/*.egg-info
    pip install ./thirdparty/monogs/submodules/simple-knn                     # to clean: $ rm -rf thirdparty/monogs/submodules/simple-knn/build thirdparty/monogs/submodules/simple-knn/*.egg-info
    pip install ./thirdparty/monogs/submodules/diff-gaussian-rasterization    # to clean: $ rm -rf thirdparty/monogs/submodules/diff-gaussian-rasterization/build thirdparty/monogs/submodules/diff-gaussian-rasterization/*.egg-info
fi