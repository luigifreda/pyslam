#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

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
    . install_all_conda.sh
else
    print_blue "Installing pySLAM by using venv"
    . install_all_venv.sh
fi
