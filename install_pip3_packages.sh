#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing python packages ..."

# N.B.: python3 is required!

pip3 install --upgrade pip setuptools wheel

# pip3 packages 
install_pip_packages pygame matplotlib pyopengl Pillow pybind11 
install_pip_package numpy==1.18.3
install_pip_package scipy==1.4.1
install_pip_package scikit-image==0.16.2
install_pip_packages pyyaml termcolor tqdm yacs

# install_pip_package opencv-python  
# #pip3 uninstall opencv-contrib-python  # better to clean it before installing the right version 
# install_pip_package opencv-contrib-python #==3.4.2.16 

# faster way to install opencv packages 
PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
MAKEFLAGS_OPTION="-j$(nproc)"

MAKEFLAGS="$MAKEFLAGS_OPTION" pip3 install opencv-python -vvv $PRE_OPTION
MAKEFLAGS="$MAKEFLAGS_OPTION" pip3 install opencv-contrib-python -vvv $PRE_OPTION 

install_pip_package torch==1.4.0 
install_pip_package torchvision==0.5.0
install_pip_package ordered-set # from https://pypi.org/project/ordered-set/

# The following line might not work
#install_pip_package tensorflow-gpu==1.14.0  # 1.14.0 works with all the modules contained in pyslam2
# Thanks Utkarsh for the fix https://github.com/luigifreda/pyslam/issues/92 
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl
pip3 install --upgrade $TF_BINARY_URL


# it may be required if you have errors with pillow
#pip3 uninstall pillow 
#pip3 install pillow==6.2.2

 
