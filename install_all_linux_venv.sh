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

# 1. install system packages 
./install_system_packages.sh    

# 2. create a pyslam environment within venv 
./pyenv-create.sh 

# 3. activate the created python virtual environment 
. pyenv-activate.sh   

# 4. set up git submodules (we need to install gdown before this) 
./install_git_modules.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # 5. install pip packages: some unresolved dep conflicts found in requirement-pip3.txt may be managed by the following command: 
. install_pip3_packages.sh 

# 6. build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# 7. build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 
