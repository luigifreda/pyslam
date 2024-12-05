#!/usr/bin/env bash
# Author: Luigi Freda 

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Installing opencv-python from source"
print_blue '================================================'

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

#pip install --upgrade pip
pip uninstall -y opencv-python
pip uninstall -y opencv-contrib-python

pip install --upgrade numpy

cd thirdparty
if [ ! -d opencv-python ]; then
    git clone --recursive https://github.com/opencv/opencv-python.git
    cd opencv-python
    # This procedure worked on commit cce7c994d46406205eb39300bb7ca9c48d80185a  that corresponds to opencv 4.10.0.84 -> https://github.com/opencv/opencv-python/releases/tag/84 
    git checkout cce7c994d46406205eb39300bb7ca9c48d80185a  # uncomment this if you get some issues in building opencv_python!
    cd ..
fi

cd opencv-python
MY_OPENCV_PYTHON_PATH=`pwd`

export MAKEFLAGS="-j$(nproc)"
export CPPFLAGS+=""
export CPATH+=""
export CPP_INCLUDE_PATH+=""
export C_INCLUDE_PATH+=""
if [ -z $USING_CONDA_PYSLAM ]; then
    PYSLAM_VENV_PATH=$HOME/.python/venvs/pyslam
    CPPFLAGS="-I$PYSLAM_VENV_PATH/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/:$CPPFLAGS" 
    CPATH="$PYSLAM_VENV_PATH/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/":$CPATH
    C_INCLUDE_PATH="$PYSLAM_VENV_PATH/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/":$C_INCLUDE_PATH
    CPP_INCLUDE_PATH="$PYSLAM_VENV_PATH/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/":$CPP_INCLUDE_PATH
else
    echo "Using conda pyslam..."
    # we want to use conda environment
    CONDA_BASE_FOLDER="$(conda info --base)"
    CPPFLAGS="-I$CONDA_BASE_FOLDER/envs/pyslam/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/:$CPPFLAGS"
    CPATH="$CONDA_BASE_FOLDER/envs/pyslam/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/":$CPATH
    C_INCLUDE_PATH="$CONDA_BASE_FOLDER/envs/pyslam/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/":$C_INCLUDE_PATH
    CPP_INCLUDE_PATH="$CONDA_BASE_FOLDER/envs/pyslam/lib/python$PYTHON_VERSION/site-packages/numpy/core/include/":$CPP_INCLUDE_PATH
fi

export CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$MY_OPENCV_PYTHON_PATH/opencv_contrib/modules \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
-DCMAKE_CXX_FLAGS=$CPPFLAGS
$MY_OPENCV_PYTHON_PATH"

# build opencv_python 
pip wheel . --verbose
# install built packages
pip install opencv*.whl --force-reinstall

# build opencv_contrib_python
export ENABLE_CONTRIB=1
pip wheel . --verbose
# install built packages
pip install opencv*.whl --force-reinstall

cd $STARTING_DIR