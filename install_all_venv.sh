#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e

# set up git submodules  
. install_basic.sh 0 0 # the first '0' is an option for skipping pip3 packages installation (script install_pip3_packages.sh),
                       # the second '0' is for skipping the install_cpp.sh script therein (that will be called below) 

# create a pyslam environment within venv 
. pyenv-create.sh 

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

# build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 


# N.B.:
# if your run into troubles with opencv xfeatures2D/SIFT/SURF then run the following commands into your pyslam environment 
# $ pip3 uninstall opencv-contrib-python
# $ pip3 install opencv-contrib-python==3.4.2.16

# HACK (this is actually run inside the created pyenv environment ) 
#pip uninstall opencv-contrib-python
#pip install opencv-contrib-python==3.4.2.16
