#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

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
$SCRIPTS_DIR/install_system_packages.sh     

# 2. create a pyslam environment within conda (this will set the env var USING_CONDA_PYSLAM)
$SCRIPTS_DIR/pyenv-conda-create.sh 

# 3. activate the created python virtual environment 
. "$ROOT_DIR"/pyenv-activate.sh   

# Check if the current conda environment is "pyslam"
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    print_red "ERROR: The current conda environment is not '$ENV_NAME'. Please activate the '$ENV_NAME' environment and try again."
    exit 1
fi

# 4. set up git submodules  
$SCRIPTS_DIR/install_git_modules.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # 5. install pip packages: some unresolved dep conflicts found in requirement-pip3.txt may be managed by the following command: 
. $SCRIPTS_DIR/install_pip3_packages.sh 

# 6. build and install cpp stuff 
. $SCRIPTS_DIR/install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# 7. build and install thirdparty 
. $SCRIPTS_DIR/install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 

# 8. install tools for semantics
# HACK: Moved the install of the semantic tools at the end of the install process to avoid some conflict issues among the deps
$SCRIPTS_DIR/install_pip3_semantics.sh  # must use "./"

# 9. outliers under macos
if [[ "$OSTYPE" == "darwin"* ]]; then
    # To solve under mac the (crash) issue mentioned in the troubleshoting document
    pip uninstall tensorflow
    #pip install "tensorflow==2.15.*"
    pip install tensorflow-macos tensorflow-metal --force-reinstall
fi 

# 9. outliers under conda
pip install "pyarrow<19"  # See https://github.com/luigifreda/pyslam/issues/193
pip install -U "protobuf>=5,<6" # For solving final issues with contextdesc
# NOTE: There can be possible issues with delf and protobuf too. To solve them, run the following command:
# cd <pyslam_root>/thirdparty/tensorflow_models/research/delf
# protoc -I=. --python_out=. delf/protos/*.proto
pip install "numpy<2"  

cd "$STARTING_DIR"