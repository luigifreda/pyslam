#!/usr/bin/env bash

#N.B: this install script allows you to run main_vo.py and the test scripts 
# echo "usage: ./${0##*/} <INSTALL_PIP3_PACKAGES>"   # the argument is optional 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

export INSTALL_PIP3_PACKAGES=1   # install pip3 packages by default 
if [ $# -eq 1 ]; then
    # check optional argument 
    INSTALL_PIP3_PACKAGES=$1
    echo INSTALL_PIP3_PACKAGES: $INSTALL_PIP3_PACKAGES
fi
# ====================================================

set -e

echo `pwd`

# install system packages 
./install_system_packages.sh

# install pip3 packages 
# N.B.: install_pip3_packages script can be skipped if you intend to use a virtual python environment 
if [ $INSTALL_PIP3_PACKAGES -eq 1 ]; then
    echo 'installing pip3 packages'
    ./install_pip3_packages.sh 
fi 

# set up git submodules  
./install_git_modules.sh 

# build and install cpp stuff 
./install_cpp.sh 
