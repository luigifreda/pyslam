#!/usr/bin/env bash


# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

print_blue '================================================'
print_blue "Configuring and installing python packages ..."

# N.B.: it's required the use of python3 

install_pip_packages pygame numpy matplotlib pyopengl Pillow pybind11 scikit-image pyyaml termcolor
install_pip_packages opencv-python opencv-contrib-python 
install_pip_packages torch 
install_pip_packages torchvision
install_package python3-sdl2 
install_package python3-tk

# it may be required if you have errors with pillow
#pip3 uninstall pillow 
#pip3 install pillow==6.2.2

print_blue '================================================'
print_blue "Checking and downloading git modules ..."
set_git_modules   