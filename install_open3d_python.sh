#!/usr/bin/env bash
# Author: Luigi Freda 

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

# NOTE: this is required under mac where I got unexpected segmentation fault errors
#       on open3d dynamic library loading


print_blue "Installing open3d-python from source"

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

#pip install --upgrade pip
pip uninstall -y open3d

cd thirdparty
if [ ! -d open3d ]; then
    git clone https://github.com/isl-org/Open3D.git open3d
    cd open3d
    
    # This commit 0f06a149c4fb9406fd3e432a5cb0c024f38e2f0e didn't work. It corresponds to open3d 0.18.0 -> https://github.com/isl-org/Open3D/commits/v0.18.0
    #git checkout 0f06a149c4fb9406fd3e432a5cb0c024f38e2f0e 

    # This commit worked!
    git checkout c8856fc0d4ec89f8d53591db245fd29ad946f9cb

    cd ..
fi

cd open3d
if [ ! -d build ]; then
    mkdir build 
fi 
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTION

make -j$(sysctl -n hw.physicalcpu)

# Activate the virtualenv first
# Install pip package in the current python environment
make install-pip-package

# Create Python package in build/lib
make python-package

# Create pip wheel in build/lib
# This creates a .whl file that you can install manually.
make pip-package


sudo make install