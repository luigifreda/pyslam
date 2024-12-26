#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e

# clean the old .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
  rm "$ROOT_DIR/.env"
fi

# install system packages 
./install_system_packages.sh    

# set up git submodules  
./install_git_modules.sh 

# create a pyslam environment within venv 
./pyenv-create.sh 

# activate the created python virtual environment 
. pyenv-activate.sh   

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # some unresolved dep conflict found in requirement-pip3.txt may be managed by the following command: 
. install_pip3_packages.sh 

# build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 
