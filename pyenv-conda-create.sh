#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME='pyslam'
fi

#conda create --name $ENV_NAME --file requirements-conda.txt -c conda-forge
# or (easier)
conda env create -f requirements-conda.yml

. pyenv-conda-activate.sh 
which pip 
pip install -r requirements-conda-pip.txt  # this should refer to */pyslam/bin/pip
