#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing system packages ..."


# NOTE: in order to detect macOS use:  
if [[ "$OSTYPE" == "darwin"* ]]; then 
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
    brew install zlib bzip2
    brew install rsync
    brew install readline
    brew install pyenv
    brew install libomp   # need to add -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include
    brew install boost    # for serialization
    brew install tmux
    brew install flann
    brew install catch2
    #brew install numpy
    #brew install open3d     # We built open3d from source for different issues 
    brew install x265 libjpeg libde265 libheif   # for pillow-heif

else 
    ## Linux 

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
    install_package  libglew-dev 

    install_package libeigen3-dev # pangolin installation 
    install_package libopencv-dev # orbslam2_features compilation
    install_package libgtk2.0-dev # needed by opencv when built from source 

    install_package pkg-config
    install_package python3-gi
    install_package cmake 
    install_package build-essential 

    install_packages liblz4-dev libzstd-dev
    install_package  libhdf5-dev    # needed when building h5py wheel from src is required (arm64)

    install_packages libboost-serialization-dev libboost-system-dev libboost-filesystem-dev
    install_package  tmux # for launching tmux sessions

    install_package libqt5gui5 # for qt5 support 

    install_package libomp-dev

fi

print_blue "System package configuration and installation... Done!" 