#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [ -z "${ENV_NAME}" ]; then
    ENV_NAME='pyslam'
fi

# # Check if conda is already initialized
# if ! conda info -v &> /dev/null; then
#     conda init bash 
#     source ~/.bashrc
# fi


# This variable is used to indicate that we want to use conda
export USING_CONDA_PYSLAM=1

export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
#source activate base   # from https://community.anaconda.cloud/t/unable-to-activate-environment-prompted-to-run-conda-init-before-conda-activate-but-it-doesnt-work/68677/10
conda activate base
conda activate $ENV_NAME

# N.B.: in order to deactivate the virtual environment run: 
# $ conda deactivate
