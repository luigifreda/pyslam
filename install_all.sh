#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

set -e


echo This script is deprecated. Read the instructions in docs/PYTHON-VIRTUAL-ENVS.md and use the command: . install_all_venv.sh
exit 1


# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

# install basic modules and set up git submodules  
. install_basic.sh          # use . in order to inherit python env configuration and possible other env vars 

# build and install thirdparty 
. install_thirdparty.sh     # use . in order to inherit python env configuration and possible other env vars 