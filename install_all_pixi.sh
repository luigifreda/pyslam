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

print_blue "Running install_all_pixi.sh"

#set -e

# clean the old .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
  rm "$ROOT_DIR/.env"
fi

set_env_var "$ROOT_DIR/.env" USE_PIXI 1

# Check if pixi shell is active
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
  echo "Inside pixi shell: $PIXI_PROJECT_NAME"
else
   print_red "ERROR: pixi shell is not active! Did you installe/activate pixi? Run: pixi shell" 
  exit 1 
fi


#pixi shell 
echo $(which python) 

# # 1. install system packages 
./install_system_packages.sh    

# # 2. set up git submodules (we need to install gdown before this) 
./install_git_modules.sh 

# 3. configure the environment for pixi
PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYTHON_LIBRARY=$(find $(dirname $PYTHON_EXECUTABLE)/../lib -name libpython$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").so | head -n 1)
echo PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE
echo PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR
echo PYTHON_LIBRARY: $PYTHON_LIBRARY

# This is needed to make pybind11 find the correct python interpreter within the pixi environment
export EXTERNAL_OPTIONS="-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
  -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
  -DPYTHON_LIBRARY=$PYTHON_LIBRARY"

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

# 4. install pip packages: some unresolved dep conflicts found in requirement-pip3.txt may be managed by the following command: 
#. install_pip3_packages.sh $EXTERNAL_OPTIONS
. install_pip3_packages2.sh 

# 5. build and install cpp stuff 
. install_cpp.sh $EXTERNAL_OPTIONS                    # use . in order to inherit python env configuration and other environment vars 

# 6. build and install thirdparty 
. install_thirdparty.sh  $EXTERNAL_OPTIONS            # use . in order to inherit python env configuration and other environment vars 

# 7. install tools for semantics
# HACK: Moved the install of the semantic tools at the end of the install process to avoid some conflict issues among the deps
./install_pip3_semantics.sh    # must use "./"

cd "$STARTING_DIR"