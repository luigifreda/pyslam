#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

# set up git submodules  
./install_basic.sh 0 # the '0' is an option for skipping pip3 packages installation  

# create a pyslam environment within conda and activate it 
. pyenv-conda-create.sh 

# build and install thirdparty 
./install_thirdparty.sh 


# N.B.:
# if your run into troubles with opencv xfeatures2D/SIFT/SURF then run the following commands into your pyslam environment 
# $ pip3 uninstall opencv-contrib-python
# $ pip3 install opencv-contrib-python==3.4.2.16

# HACK (this is actually run inside the created pyenv environment ) 
#pip uninstall opencv-contrib-python
#pip install opencv-contrib-python==3.4.2.16
