#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

ENVS_PATH=$ROOT_DIR/.pixi    # path where to group virtual environments 
ENV_PATH="$ENVS_PATH"        # path of the virtual environment we are creating 
 
if [ -d "$ENV_PATH" ]; then 
    echo deleting virtual environment $ENV_NAME in $ENV_PATH
    sudo rm -R "$ENV_PATH"
else
    echo virtual environment $ENV_NAME does not exist in $ENV_PATH
fi 

