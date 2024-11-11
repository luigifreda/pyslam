#!/usr/bin/env bash


# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e


# ====================================================
# check if we have external options
EXTERNAL_OPTION=$1
if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
fi

# check if we want to add a python interpreter check
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi
# ====================================================
# check if want to use conda or venv
if [ -z $USING_CONDA_PYSLAM ]; then
    if [[ -z "${USE_PYSLAM_ENV}" ]]; then
        USE_PYSLAM_ENV=0
    fi
    if [ $USE_PYSLAM_ENV -eq 1 ]; then
        . pyenv-activate.sh
    fi  
else 
    echo "Using conda pyslam..."
    . pyenv-conda-activate.sh
fi 

print_blue '================================================'
print_blue "Building and installing cpp ..."

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "currently used pyenv: $CURRENT_USED_PYENV"

cd cpp 

# build utils 
cd utils 
. build.sh $EXTERNAL_OPTION       # use . in order to inherit python env configuration 
cd ..

cd .. 


# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON