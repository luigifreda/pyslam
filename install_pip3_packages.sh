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

install_pip_package psutil

# Install opencv_python from source with non-free modules enabled 
PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
MAKEFLAGS_OPTION="-j$(nproc)"
CMAKE_ARGS_OPTION="-DOPENCV_ENABLE_NONFREE=ON" # install nonfree modules

MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-python -vvv $PRE_OPTION
MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-contrib-python -vvv $PRE_OPTION

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

if [[ "$OSTYPE" == "darwin"* ]]; then
    install_pip_package torch==2.1           # torch==2.2.0 causes some segmentation faults on mac
    install_pip_package torchvision==0.16         
else 
    install_pip_package torch==2.2.0
    install_pip_package torchvision==0.17               
fi 

install_pip_package rerun-sdk #==0.17.0

pip install protobuf==3.20.*    # for delf NN


 
