#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

print_blue "Running install_all_conda.sh"

#set -e

# clean the old .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
  rm "$ROOT_DIR/.env"
fi

set_env_var "$ROOT_DIR/.env" USE_CONDA 1

# check that conda is activated 
if ! command -v conda &> /dev/null ; then
    print_red "ERROR: conda could not be found! Did you installe/activate conda?"
    exit 1 
fi

# 1. install system packages 
./install_system_packages.sh     

# 2. create a pyslam environment within conda (this will set the env var USING_CONDA_PYSLAM)
./pyenv-conda-create.sh 

# 3. activate the created python virtual environment 
. pyenv-activate.sh   

# Check if the current conda environment is "pyslam"
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    print_red "ERROR: The current conda environment is not '$ENV_NAME'. Please activate the '$ENV_NAME' environment and try again."
    exit 1
fi

# 4. set up git submodules  
./install_git_modules.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # 5. install pip packages: some unresolved dep conflicts found in requirement-pip3.txt may be managed by the following command: 
. "$ROOT_DIR"/install_pip3_packages.sh 

# 6. build and install cpp stuff 
. "$ROOT_DIR"/install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# 7. build and install thirdparty 
. "$ROOT_DIR"/install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 
