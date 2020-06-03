#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

# set up git submodules  
./install_basic.sh 0 # the '0' is an option for skipping pip3 packages installation  

. pyenv-conda-create.sh 

# build and install thirdparty 
./install_thirdparty.sh 