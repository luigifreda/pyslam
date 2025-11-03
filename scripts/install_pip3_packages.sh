#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

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


# Install opencv_python from source with non-free modules enabled 
INSTALL_OPENCV_FROM_SOURCE=1
if [ $INSTALL_OPENCV_FROM_SOURCE -eq 1 ]; then
    #NOTE: This procedures is preferable since it avoids issues with Qt linking/configuration
    print_green "Installing opencv_python from source with non-free modules enabled"
    #$SCRIPTS_DIR/install_opencv_python.sh
    $SCRIPTS_DIR/install_opencv_local.sh
else
    PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
    MAKEFLAGS_OPTION="-j$(nproc)"
    CMAKE_ARGS_OPTION="-DOPENCV_ENABLE_NONFREE=ON" # install nonfree modules

    MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip install $PIP_MAC_OPTIONS opencv-python -vvv $PRE_OPTION
    MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip install $PIP_MAC_OPTIONS opencv-contrib-python -vvv $PRE_OPTION
fi

# Install torch and related packages
$SCRIPTS_DIR/install_pip3_torch.sh

if [[ "$OSTYPE" != "darwin"* ]]; then

    # for crestereo
    # Unfortunately, megengine is not supported on macOS with arm architecture
    pip install --upgrade cryptography pyOpenSSL
    python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
    #pip install megengine  # This brings issues with non-supported CUDA architecture on my machine
    # NOTE: if you do not succeed in installing megengine in your system with the pre-built wheels, you can try to install it from source with the following command:
    #   $SCRIPTS_DIR/install_megengine.sh    
    # Megengine supports `DepthEstimatorCrestereoMegengine`. However, there is an equivalent 
    #`DepthEstimatorCrestereoPytorch` that is fully working.

    # for ROS
    pip install pycryptodomex
    pip install gnupg
    pip install rospkg
fi

if [ "$CUDA_VERSION" != "0" ]; then
    $SCRIPTS_DIR/install_faiss.sh # for loop closure database on GPU

    # MonoGS required packages
    ./thirdparty/lietorch/build.sh   # building with cmake to enable parallel threads (for some reasons, enabling parallel threads in pip install fails)
   
   # NOTE: --no-build-isolation is used to avoid the build isolation issue: Pip’s build isolation prevents access to the already-installed torch
    pip install --no-build-isolation ./thirdparty/monogs/submodules/simple-knn                     # to clean: $ rm -rf thirdparty/monogs/submodules/simple-knn/build thirdparty/monogs/submodules/simple-knn/*.egg-info
    pip install --no-build-isolation ./thirdparty/monogs/submodules/diff-gaussian-rasterization    # to clean: $ rm -rf thirdparty/monogs/submodules/diff-gaussian-rasterization/build thirdparty/monogs/submodules/diff-gaussian-rasterization/*.egg-info
else
    print_yellow "Skipping MonoGS since CUDA_VERSION is 0"
fi 

# Install tesorflow and related packages
pip install tensorflow==2.13
pip install tensorflow_hub  # required by VPR
pip install tf_slim==1.1.0
pip install protobuf==3.20.3 --force-reinstall # delf

pip3 install -U imgviz

pip install "numpy<2"

# HACK: Moved the install of the semantic tools at the end of the install process to avoid some conflict issues among the deps
# $SCRIPTS_DIR/install_pip3_semantics.sh

