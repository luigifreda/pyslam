#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring and installing system packages ..."

# N.B.: python3 is required!

# system packages 
install_package python3-sdl2 
install_package python3-tk

install_package libprotobuf-dev 

install_package libeigen3-dev # pangolin installation 
install_package libopencv-dev # orbslam2_features compilation
