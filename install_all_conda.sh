#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e

# check that conda is activated 
if ! command -v conda &> /dev/null
then
    echo "ERROR: conda could not be found! did you installed/activated conda?"
    return 1 
fi

# install system packages 
./install_system_packages.sh     

# set up git submodules  
./install_git_modules.sh 

# create a pyslam environment within conda and activate it (this will set the env var USING_CONDA_PYSLAM)
. pyenv-conda-create.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # some unresolved dep conflict found in requirement-pip3.txt may be managed by the following command: 
. install_pip3_packages.sh 

# build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 
