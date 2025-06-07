#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

# ====================================================


export ENV_NAME=$1

if [ -z "$ENV_NAME" ]; then
    ENV_NAME='pyslam'
fi

#echo "ENV_NAME: $ENV_NAME"

print_blue '================================================'
print_blue "Creating Conda Environment: $ENV_NAME"
print_blue '================================================'

# check that conda is activated 
if ! command -v conda &> /dev/null ; then
    print_red "ERROR: Conda could not be found! Did you installe/activate conda?"
    exit 1
fi

conda update conda -y

export PYSLAM_PYTHON_VERSION="3.10.12"

if conda env list | grep -E "^[[:space:]]*$ENV_NAME[[:space:]]" > /dev/null; then
    print_yellow "Conda environment $ENV_NAME already exists."
else 
    print_blue "Creating conda virtual environment $ENV_NAME with python version $PYSLAM_PYTHON_VERSION"
    conda create -yn "$ENV_NAME" python="$PYSLAM_PYTHON_VERSION"
fi

# on first run
if [ -z "$CONDA_PREFIX" ]; then
    CONDA_PREFIX=$(conda info --base)
fi
. "$CONDA_PREFIX"/bin/activate base   # from https://community.anaconda.cloud/t/unable-to-activate-environment-prompted-to-run-conda-init-before-conda-activate-but-it-doesnt-work/68677/10

# activate created env  
. pyenv-conda-activate.sh 

# Check if the current conda environment is "pyslam"
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    print_red "ERROR: The current conda environment is not '$ENV_NAME'. Please activate the '$ENV_NAME' environment and try again."
    exit 1
fi

#which pip  # this should refer to */pyslam/bin/pip  (that is actually pip3)

pip3 install --upgrade pip setuptools wheel

# install required packages (basic packages, some unresolved conflicts may be resolved by the next steps)
pip3 install -r requirements-pip3.txt #-vvv


cd "$STARTING_DIR"

# To activate this environment, use
#   $ conda activate pyslam
# To deactivate an active environment, use
#   $ conda deactivate
