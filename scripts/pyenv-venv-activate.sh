#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=${1:-pyslam}  # default name of the virtual environment is "pyslam"

ENVS_PATH=~/.python/venvs  # path where to group virtual environments 
ENV_PATH=$ENVS_PATH/$ENV_NAME        # path of the virtual environment we are creating 

#export PYTHONPATH=$ENVS_PATH/$ENV_NAME/bin  
export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 

if [ ! -f $ENV_PATH/bin/activate ]; then
    echo "There is no $ENV_NAME virtual environment in $ENV_PATH"
    return 1
fi
. $ENV_PATH/bin/activate  

# N.B.: in order to deactivate the virtual environment run: 
# $ deactivate 

# Check if the operating system is Darwin (macOS)
if [[ $OSTYPE == 'darwin'* ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    echo "setting PYTORCH_ENABLE_MPS_FALLBACK: $PYTORCH_ENABLE_MPS_FALLBACK"
fi