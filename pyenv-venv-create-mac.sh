#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

cd "$ROOT_DIR"
STARTING_DIR=`pwd`

export ENV_NAME=$1

if [ -z "${ENV_NAME}" ]; then
    ENV_NAME='pyslam'
fi

ENVS_PATH=~/.python/venvs  # path where to group virtual environments 
ENV_PATH=$ENVS_PATH/$ENV_NAME        # path of the virtual environment we are creating 

# clean previous install 
if [ -d ~/.python/venvs/pyslam/ ]; then 
   rm -Rf ~/.python/venvs/pyslam/
   echo ""
fi  

# ====================================================

# export LDFLAGS="-L/opt/homebrew/lib:$LDFLAGS"
# export CPPFLAGS="-I/opt/homebrew/include:$CPPFLAGS"
# export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig/"


# ====================================================
# create folder for virtual environment and get into it 
make_dir $ENV_PATH
cd $ENVS_PATH

export PYSLAM_PYTHON_VERSION="3.10.12"

# actually create the virtual environment 
if [ ! -d $ENV_PATH/bin ]; then 
    print_blue "Creating virtual environment $ENV_NAME with python version $PYSLAM_PYTHON_VERSION under MacOs"
    export PATH=~/.pyenv/shims:$PATH
    
    PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s -v  $PYSLAM_PYTHON_VERSION

    pyenv local $PYSLAM_PYTHON_VERSION
    python3 -m venv $ENV_NAME
fi 

# activate the environment 
cd $STARTING_DIR
export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
. $ENV_PATH/bin/activate  

# install required packages 

pip3 install --upgrade pip

#print_blue "Installing opencv"

MAKEFLAGS_OPTION="-j$(nproc)" 
# PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
# CMAKE_ARGS_OPTION="-DOPENCV_ENABLE_NONFREE=ON" # install nonfree modules

# PIP_MAC_OPTIONS=""
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     PIP_MAC_OPTIONS=" --no-binary :all: "
# fi

# This does not reall work on mac
# MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-python -vvv $PRE_OPTION
# MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install $PIP_MAC_OPTIONS opencv-contrib-python -vvv $PRE_OPTION

# install required packages (basic packages, some unresolved conflicts may be resolved by the next steps)
MAKEFLAGS="$MAKEFLAGS_OPTION" pip3 install -r requirements-pip3.txt #-vvv

# NOTE: opencv is installed outside of this script!
# install opencv python from source with non-free modules enabled (installation order does matter here!)
#. install_opencv_python.sh
# NOTE: This another way of installing opencv and it solves some dynamic linking issues. Unfortunately, it generates other ones that are more critical with multiple intercepted QT versions. 
# pip3 install numpy
# . install_opencv_local.sh


# To activate the virtual environment run: 
#   $ source pyenv-activate.sh 
# To deactivate run:
#   $ deactivate 
