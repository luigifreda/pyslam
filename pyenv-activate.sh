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

# ====================================================
# some useful environment variables to remove unwanted warnings

export TF_CPP_MIN_LOG_LEVEL=3 # 0=all messages are logged (default), 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL

# Ignore *all* warnings coming from NumPy getlimits
export PYTHONWARNINGS="ignore:::numpy.core.getlimits"
# Add pkg_resources deprecation message too
export PYTHONWARNINGS="$PYTHONWARNINGS,ignore:pkg_resources is deprecated"

# Disable MIT-SHM extension for Qt/PyQt multi-threading
# export QT_X11_NO_MITSHM=1

# optional:
# export PYTHONWARNINGS="ignore::UserWarning" # ignore all UserWarning
# export TF_ENABLE_ONEDNN_OPTS=0
# export GLOG_minloglevel=3
# export GLOG_logtostderr=1

# ====================================================

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
    print_blue "You need to activate pySLAM environment by using pixi shell"
    #pixi shell 
elif [ "$CONDA_INSTALLED" = true ]; then
    # check that conda is activated     
    print_blue "Activating pySLAM environment by using conda"
    . $SCRIPTS_DIR/pyenv-conda-activate.sh
else
    print_blue "Activating pySLAM environment by using venv"
    . $SCRIPTS_DIR/pyenv-venv-activate.sh
fi


cd "$STARTING_DIR"