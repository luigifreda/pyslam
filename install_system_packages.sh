#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing system packages ..."


install_package rsync 
install_package unzip 

# system packages 
install_package build-essential cmake
install_package python3-sdl2 
install_package python3-tk

install_package libsuitesparse-dev

install_package libprotobuf-dev 

install_package libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev
install_package libglew-dev 

install_package libeigen3-dev # pangolin installation 
install_package libopencv-dev # orbslam2_features compilation

#install_package libgtk2.0-dev 
install_package pkg-config
install_package python3-gi
install_package cmake 
install_package build-essential 
