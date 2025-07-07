#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

#set -e

STARTING_DIR=`pwd`
cd "$ROOT_DIR" 

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# check if we want to add a python interpreter check
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi


OpenCV_DIR="$ROOT_DIR/thirdparty/opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================
# activate pyslam python environment
#. "$ROOT_DIR"/pyenv-activate.sh

print_blue '================================================'
print_blue "Building and installing cpp ..."

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "Currently used pyenv: $CURRENT_USED_PYENV"

cd cpp 

# build utils
. build.sh $EXTERNAL_OPTIONS       # use . in order to inherit python env configuration 


cd "$STARTING_DIR"


# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON