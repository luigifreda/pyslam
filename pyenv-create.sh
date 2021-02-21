#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

STARTING_DIR=`pwd`

export ENV_NAME=$1

if [[ -z "${ENV_NAME}" ]]; then
    ENV_NAME='pyslam'
fi

ENVS_PATH=~/.python/venvs  # path where to group virtual environments 
ENV_PATH=$ENVS_PATH/$ENV_NAME        # path of the virtual environment we are creating 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

# install required package to create virtual environment
install_package python3-venv
. install_pyenv.sh 

# create folder for virutal environment and get into it 
make_dir $ENV_PATH
cd $ENVS_PATH

export PYSLAM_PYTHON_VERSION="3.6.9"

# actually create the virtual environment 
if [ ! -d $ENV_PATH/bin ]; then 
    print_blue creating virtual environment $ENV_NAME with python version $PYSLAM_PYTHON_VERSION
    pyenv install -v $PYSLAM_PYTHON_VERSION
    pyenv local $PYSLAM_PYTHON_VERSION
    python3 -m venv $ENV_NAME
fi 

# activate the environment 
cd $STARTING_DIR
export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
source $ENV_PATH/bin/activate  

pip3 install --upgrade pip

# install required packages 

#source install_pip3_packages.sh 
# or 
pip3 install -r requirements-pip3.txt

# HACK to fix opencv-contrib-python version!
#pip3 uninstall opencv-contrib-python                # better to clean it before installing the right version 
#install_pip_package opencv-contrib-python #==3.4.2.16 

# N.B.: in order to activate the virtual environment run: 
# $ source pyenv-activate.sh 
# to deactivate 
# $ deactivate 
