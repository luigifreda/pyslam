#!/usr/bin/env bash

#N.B: this install script allows you to run main_vo.py and the test scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

echo `pwd`

# install system packages 
./install_system_packages.sh

# install pip3 packages 
 # N.B.: install_pip3_packages script can be skipped if you intend to use a virtual python environment 
./install_pip3_packages.sh 

# set up git submodules  
./install_git_modules.sh 

# build and install cpp stuff 
./install_cpp.sh 
