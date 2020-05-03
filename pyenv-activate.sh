#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "${ENV_NAME}" ]]; then
    ENV_NAME='pyslam'
fi

ENVS_PATH=~/.python/venvs  # path where to group virtual environments 
ENV_PATH=$ENVS_PATH/$ENV_NAME        # path of the virtual environment we are creating 

export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
source $ENV_PATH/bin/activate  

# N.B.: in order to deactivate the virtual environment run: 
# $ deactivate 
