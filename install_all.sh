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

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

#set -e

# if we are not under docker
if [ ! -f /.dockerenv  ]; then 
    echo "Insert your sudo password (if prompted)..."
    keep_sudo_alive
fi
# Ensure the sudo background process is killed when the script exits
trap stop_sudo_alive EXIT

# Check if conda is installed
if command -v conda &> /dev/null; then
    CONDA_INSTALLED=true
else
    CONDA_INSTALLED=false
fi

if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
else
    PIXI_ACTIVATED=false
fi

print_blue '================================================'
print_blue "Installing pySLAM"
print_blue '================================================'

# check that conda is activated 
if [ "$CONDA_INSTALLED" = true ]; then
    print_blue "Installing pySLAM by using conda"
    . "$SCRIPTS_DIR"/install_all_conda.sh
elif [ "$PIXI_ACTIVATED" = true ]; then
    print_blue "Installing pySLAM by using pixi"
    . "$SCRIPTS_DIR"/install_all_pixi.sh
else
    print_blue "Installing pySLAM by using venv"
    . "$SCRIPTS_DIR"/install_all_venv.sh
fi

print_blue '================================================'
print_blue "Building and installing pyslam C++ core"
print_blue '================================================'
"$ROOT_DIR"/build_cpp.sh

cd "$STARTING_DIR"