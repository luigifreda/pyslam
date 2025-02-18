#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

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
    print_blue "Creating pySLAM environment by using conda"
    . pyenv-conda-create.sh
else
    print_blue "Creating pySLAM environment by using venv"
    . pyenv-venv-create.sh
fi
