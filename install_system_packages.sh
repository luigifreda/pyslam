#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing system packages ..."

sudo apt-get update

install_package rsync 
install_package unzip 

# system packages 
install_packages build-essential cmake
install_package python3-sdl2 
install_package python3-tk

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

echo "... Done!" 