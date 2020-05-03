#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "${ENV_NAME}" ]]; then
    ENV_NAME='pyslam'
fi

export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
conda activate $ENV_NAME

# N.B.: in order to deactivate the virtual environment run: 
# $ conda deactivate 
 