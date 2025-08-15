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


STARTING_DIR=`pwd`
cd "$ROOT_DIR" 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing system packages ..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
    CONDA_INSTALLED=false
fi


# NOTE: in order to detect macOS use:  
if [[ "$OSTYPE" == darwin* ]]; then 
    ## MacOS

    echo "Installing macOs packages with brew..."

    # Check if brew is installed
    if ! command -v brew &> /dev/null; then
        # brew is not installed 
        print_red "❌ ERROR: brew could not be found!"
        print_red "1. Install Homebrew: https://brew.sh/"
        exit 1
    fi
        
    # install required packages
    brew update 
    brew install wget 
    brew install doxygen 
    brew install eigen 
    #brew install opencv 
    brew install glew 
    brew install pkg-config 
    brew install suite-sparse 
    brew install pyenv
    brew install zlib bzip2 unzip minizip
    brew install rsync
    brew install readline
    brew install libomp   # need to add -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include
    brew install boost    # for serialization
    brew install tmux
    brew install flann
    brew install catch2
    brew install assimp
    brew install liblzf
    brew install openssl
    brew install libpng
    brew install openblas
    brew install vtk; brew link vtk
    #brew install numpy
    #brew install open3d     # we are going to build open3d from source for different issues 
    brew install x265 libjpeg libde265 libheif   # for pillow-heif

else 
    ## Linux 

    echo "Installing linux packages with apt-get..."    

    sudo apt-get update

    install_package rsync 
    install_package wget
    install_package unzip 

    # system packages 
    install_packages build-essential cmake
    install_package python3-sdl2 
    install_package python3-tk
    install_package python3-dev

    install_package libsuitesparse-dev

    install_package libprotobuf-dev 

    install_packages libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev 

    install_package libglew-dev 
    install_package libgoogle-glog-dev

    install_package libeigen3-dev # pangolin installation 
    install_package libopencv-dev # orbslam2_features compilation
    install_package libgtk2.0-dev # needed by opencv when built from source 

    install_package pkg-config
    install_package python3-gi
    install_package cmake 
    install_package build-essential 

    install_packages liblz4-dev libzstd-dev
    install_package  libhdf5-dev    # needed when building h5py wheel from src is required (arm64)

    install_package libboost-all-dev
    install_package tmux # for launching tmux sessions

    if [[ $version == *"24.04"* ]] ; then
        install_package libqt5gui5t64 # for qt5 support     
    else
        install_package libqt5gui5 # for qt5 support 
    fi 

    install_package libomp-dev

    # detect CUDA VERSION
    . "$ROOT_DIR"/cuda_config.sh
    if [ "$CUDA_VERSION" != "0" ]; then
        # if CUDA is installed then install the required packages
        sudo apt install -y cuda-command-line-tools-$CUDA_VERSION_STRING_WITH_HYPHENS cuda-libraries-$CUDA_VERSION_STRING_WITH_HYPHENS
        sudo apt install -y libcusparse-dev-$CUDA_VERSION_STRING_WITH_HYPHENS libcusolver-dev-$CUDA_VERSION_STRING_WITH_HYPHENS
    fi

    if [ "$CONDA_INSTALLED" = true ]; then
        $SCRIPTS_DIR/install_gcc11_if_needed.sh # this won't change your system gcc version!
    fi
fi

print_blue "System package configuration and installation... Done!" 

cd "$STARTING_DIR"