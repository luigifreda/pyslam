#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

# set up git submodules  
./install_basic.sh 

# build and install thirdparty 
./install_thirdparty.sh 