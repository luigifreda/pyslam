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


# ====================================================

version=$(lsb_release -a 2>&1)  # ubuntu version 


sudo apt update 
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev unzip
sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev ffmpeg 
sudo apt-get install -y libgtk2.0-dev 
sudo apt-get install -y libglew-dev
sudo apt-get install -y libsuitesparse-dev

# https://github.com/pyenv/pyenv/issues/1889
export CUSTOM_CC_OPTIONS=""
if [[ $version == *"22.04"* ]] ; then
    sudo apt install -y clang
fi 

# install required package to create virtual environment
install_package python3-venv
. install_pyenv.sh 

# create folder for virutal environment and get into it 
make_dir $ENV_PATH
cd $ENVS_PATH


export PYSLAM_PYTHON_VERSION="3.10.12"

# actually create the virtual environment 
if [ ! -d $ENV_PATH/bin ]; then 
    export PATH="/home/$USER/.pyenv/bin:$PATH"  # this seems to be needed under docker (even if it seems redundant)
    print_blue "Creating virtual environment $ENV_NAME with python version $PYSLAM_PYTHON_VERSION under Linux"
    if [[ $version == *"22.04"* ]] ; then
        CC=clang pyenv install -v $PYSLAM_PYTHON_VERSION
    else
        pyenv install -v $PYSLAM_PYTHON_VERSION
    fi
    pyenv local $PYSLAM_PYTHON_VERSION
    python3 -m venv $ENV_NAME
fi 

# activate the environment 
cd $STARTING_DIR
export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
source $ENV_PATH/bin/activate  

pip3 install --upgrade pip setuptools wheel --no-cache-dir
if [ -d ~/.cache/pip/selfcheck ]; then
    rm -r ~/.cache/pip/selfcheck/
fi 


PRE_OPTION="--pre"   # this sometimes helps because a pre-release version of the package might have a wheel available for our version of Python.
MAKEFLAGS_OPTION="-j$(nproc)" 

#print_blue "Installing opencv"
# CMAKE_ARGS_OPTION="-DOPENCV_ENABLE_NONFREE=ON" # install nonfree modules

# MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install opencv-python -vvv $PRE_OPTION
# MAKEFLAGS="$MAKEFLAGS_OPTION" CMAKE_ARGS="$CMAKE_ARGS_OPTION" pip3 install opencv-contrib-python -vvv $PRE_OPTION

# install required packages (basic packages, some unresolved conflicts may be resolved by the next steps)
MAKEFLAGS="$MAKEFLAGS_OPTION" pip3 install -r requirements-pip3.txt #-vvv


# To activate the virtual environment run: 
#   $ source pyenv-activate.sh 
# To deactivate run:
#   $ deactivate 
