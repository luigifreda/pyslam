#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME='pyslam'
fi

# check that conda is activated 
if ! command -v conda &> /dev/null
then
    echo "conda could not be found! did you installed/activated conda?"
    exit
fi

export PYSLAM_PYTHON_VERSION="3.8.10"

#conda create --name $ENV_NAME --file requirements-conda.txt -c conda-forge
# or (easier)
#conda env create -f requirements-conda.yml
conda create -yn $ENV_NAME python=$PYSLAM_PYTHON_VERSION

# activate created env 
. pyenv-conda-activate.sh 

#which pip  # this should refer to */pyslam/bin/pip  (that is actually pip3)

# install opencv python from source with non-free modules enabled (installation order does matter here!)
. install_opencv_python.sh

# install required packages (basic packages, some unresolved conflicts may be resolved by the next steps)
pip3 install -r requirements-pip3.txt #-vvv


# To activate this environment, use
#   $ conda activate pyslam
# To deactivate an active environment, use
#   $ conda deactivate
