#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

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
    . "$ROOT_DIR"/pyenv-conda-activate.sh
else
    print_blue "Activating pySLAM environment by using venv"
    . "$ROOT_DIR"/pyenv-venv-activate.sh
fi
