#!/usr/bin/env bash

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Installing opencv-python from source"
print_blue '================================================'

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

#pip install --upgrade pip
pip uninstall opencv-python
pip uninstall opencv-contrib-python

cd thirdparty
if [ ! -d opencv-python ]; then
    git clone --recursive https://github.com/opencv/opencv-python.git
    # This procedure worked on commit cce7c994d46406205eb39300bb7ca9c48d80185a  that corresponds to opencv 4.10.0.84 -> https://github.com/opencv/opencv-python/releases/tag/84 
    git checkout cce7c994d46406205eb39300bb7ca9c48d80185a  # uncomment this if you get some issues in building opencv_python!
fi

cd opencv-python
MY_OPENCV_PYTHON_PATH=`pwd`

export CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$MY_OPENCV_PYTHON_PATH/opencv_contrib/modules \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
$MY_OPENCV_PYTHON_PATH"

export MAKEFLAGS="-j$(nproc)"

# build and install opencv_python 
pip wheel . --verbose
pip install *.whl

# build and install opencv_contrib_python
export ENABLE_CONTRIB=1
pip wheel . --verbose
pip install *.whl

cd $STARTING_DIR