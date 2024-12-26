#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "${ENV_NAME}" ]]; then
    ENV_NAME='pyslam'
fi

ENVS_PATH=~/.python/venvs  # path where to group virtual environments 
ENV_PATH=$ENVS_PATH/$ENV_NAME        # path of the virtual environment we are creating 

#export PYTHONPATH=$ENVS_PATH/$ENV_NAME/bin  
export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 

if [ ! -f $ENV_PATH/bin/activate ]; then
    echo "Are you using conda? If so, use pyenv-conda-activate.sh"
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