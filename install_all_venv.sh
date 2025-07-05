#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

print_blue "Running install_all_venv.sh"

#set -e

# clean the old .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
  rm "$ROOT_DIR/.env"
fi

set_env_var "$ROOT_DIR/.env" USE_VENV 1

# Check if conda is installed
if command -v conda &> /dev/null; then
    print_red "ERROR: Under conda, you have to use install_all_conda.sh script!"
    exit 1
fi

# 1. install system packages 
./install_system_packages.sh    

# 2. create a pyslam environment within venv 
./pyenv-venv-create.sh  # NOTE: Keep the use of "./" seems. It seems crucial for the correct identification of the python libs for C++ projects 

# 3. activate the created python virtual environment 
. pyenv-activate.sh   

# 4. set up git submodules (we need to install gdown before this) 
./install_git_modules.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # 5. install pip packages: some unresolved dep conflicts found in requirement-pip3.txt may be managed by the following command: 
#. install_pip3_packages.sh
. install_pip3_packages2.sh 

# 6. build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# 7. build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 

# 8. install tools for semantics
# HACK: Moved the install of the semantic tools at the end of the install process to avoid some conflict issues among the deps
./install_pip3_semantics.sh  # must use "./"


cd "$STARTING_DIR"