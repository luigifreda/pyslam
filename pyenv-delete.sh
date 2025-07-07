#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_"
SCRIPTS_DIR="$ROOT_DIR/scripts"

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

# check if pixi was installed 
if [ -d "$ROOT_DIR/.pixi" ]; then
    PIXI_INSTALLED=true
else
    PIXI_INSTALLED=false
fi

if [ "$PIXI_INSTALLED" = true ]; then
    print_blue "Deleting pySLAM environment by using pixi"
    . $SCRIPTS_DIR/pyenv-pixi-delete.sh
    exit 0
fi

# check that conda is activated 
if [ "$CONDA_INSTALLED" = true ]; then
    print_blue "Deleting pySLAM environment by using conda"
    . $SCRIPTS_DIR/pyenv-conda-delete.sh
else
    print_blue "Deleting pySLAM environment by using venv"
    . $SCRIPTS_DIR/pyenv-venv-delete.sh
fi

cd "$STARTING_DIR"