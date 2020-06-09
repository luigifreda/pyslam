#!/usr/bin/env bash

#N.B: this install script allows you to run main_vo.py and the test scripts 
# echo "usage: ./${0##*/} <INSTALL_PIP3_PACKAGES> <INSTALL_CPP>"   # the arguments are optional 

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

export INSTALL_PIP3_PACKAGES=1   # install pip3 packages by default 
if [ $# -ge 1 ]; then
    # check optional argument 
    INSTALL_PIP3_PACKAGES=$1
    echo INSTALL_PIP3_PACKAGES: $INSTALL_PIP3_PACKAGES
fi

export INSTALL_CPP=1   # install cpp by default 
if [ $# -ge 2 ]; then
    # check optional argument 
    INSTALL_CPP=$2
    echo INSTALL_CPP: $INSTALL_CPP
fi

# ====================================================

echo `pwd`

# install system packages 
. install_system_packages.sh     # use . in order to inherit python env configuration 

# install pip3 packages 
# N.B.: install_pip3_packages script can be skipped if you intend to use a virtual python environment 
if [ $INSTALL_PIP3_PACKAGES -eq 1 ]; then
    echo 'installing pip3 packages'
    ./install_pip3_packages.sh   
fi 

# set up git submodules  
./install_git_modules.sh 

# build and install cpp stuff 
# N.B.: install_cpp script can be skipped here if you intend to use a virtual python environment 
#       but it must be then called within your virtual python environment in order to properly install libs 
if [ $INSTALL_CPP -eq 1 ]; then
    ./install_cpp.sh                
fi 
