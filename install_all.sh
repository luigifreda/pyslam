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

# if we are not under docker
if [ ! -f /.dockerenv  ]; then 
  # Provide the password to sudo once at the start (hopefully...)
  echo "Insert your sudo password" 
  sudo -S -v    # not working properly under mac
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    CONDA_INSTALLED=true
else
    CONDA_INSTALLED=false
fi

# check that conda is activated 
if [ "$CONDA_INSTALLED" = true ]; then
    print_blue "Installing pySLAM by using conda"
    . "$ROOT_DIR"/install_all_conda.sh
else
    print_blue "Installing pySLAM by using venv"
    . "$ROOT_DIR"/install_all_venv.sh
fi
