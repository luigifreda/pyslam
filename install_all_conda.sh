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


# set up git submodules  
. install_basic.sh 0 0 # the first '0' is an option for skipping pip3 packages installation (script install_pip3_packages.sh),
                       # the second '0' is for skipping the install_cpp.sh script therein (that will be called below) 

# create a pyslam environment within conda and activate it 
. pyenv-conda-create.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # some unresolved dep conflict found in requirement-pip3.txt may be managed by the following command: 
. install_pip3_packages.sh 

# build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 
