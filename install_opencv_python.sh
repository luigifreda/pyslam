#!/usr/bin/env bash
# Author: Luigi Freda 

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

# ====================================================

print_blue '================================================'
print_blue "Installing opencv-python from source"
print_blue '================================================'

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

# Get the current Python environment's base directory
PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")
echo "Using PYTHON_ENV: $PYTHON_ENV"


# Set include paths dynamically
export CPPFLAGS="-I$NUMPY_INCLUDE_PATH:$CPPFLAGS"
export CPATH="$NUMPY_INCLUDE_PATH:$CPATH"
export C_INCLUDE_PATH="$NUMPY_INCLUDE_PATH:$C_INCLUDE_PATH"
export CPP_INCLUDE_PATH="$NUMPY_INCLUDE_PATH:$CPP_INCLUDE_PATH"


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