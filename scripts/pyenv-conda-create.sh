#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

# ====================================================

export ENV_NAME="${1:-pyslam}"  # get the first input if any, otherwise use 'pyslam' as default name
export PYSLAM_PYTHON_VERSION="${2:-3.11.9}"  # Default Python version

# ====================================================

print_blue '================================================'
print_blue "Creating Conda Environment: $ENV_NAME"
print_blue '================================================'

# check that conda is activated 
if ! command -v conda &> /dev/null ; then
    print_red "ERROR: Conda could not be found! Did you installe/activate conda?"
    exit 1
fi

conda update conda -y


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
. "$SCRIPTS_DIR/pyenv-conda-activate.sh"

# Check if the current conda environment is "pyslam"
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    print_red "ERROR: The current conda environment is not '$ENV_NAME'. Please activate the '$ENV_NAME' environment and try again."
    exit 1
fi

pip install --upgrade pip setuptools wheel build
pip install -e .

cd "$STARTING_DIR"

# To activate this environment, use
#   $ conda activate pyslam
# To deactivate an active environment, use
#   $ conda deactivate
