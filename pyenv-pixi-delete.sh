#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [ -z "${ENV_NAME}" ]; then
    ENV_NAME='pyslam-pixi'
fi

ENVS_PATH=.pixi/envs            # path where to group virtual environments 
ENV_PATH="$ENVS_PATH/$ENV_NAME"        # path of the virtual environment we are creating 
 
if [ -d "$ENV_PATH" ]; then 
    echo deleting virtual environment $ENV_NAME in $ENV_PATH
    sudo rm -R "$ENV_PATH"
else
    echo virtual environment $ENV_NAME does not exist in $ENV_PATH
fi 

