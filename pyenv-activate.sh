#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

# ====================================================

#set -e

# Check if conda is installed
if command -v conda &> /dev/null; then
    CONDA_INSTALLED=true
else
    CONDA_INSTALLED=false
fi

# check that conda is activated 
if [ "$CONDA_INSTALLED" = true ]; then
    print_blue "Activating pySLAM environment by using conda"
    . pyenv-conda-activate.sh
else
    print_blue "Activating pySLAM environment by using venv"
    . pyenv-venv-activate.sh
fi


cd "$STARTING_DIR"