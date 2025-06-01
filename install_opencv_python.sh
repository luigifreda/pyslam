#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 


STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


# ====================================================

print_blue '================================================'
print_blue "Installing opencv-python from source"
print_blue '================================================'

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# Get the current Python environment's base directory
PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")
echo "Using PYTHON_ENV: $PYTHON_ENV"

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


NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")


# Set include paths dynamically
export CPPFLAGS="-I$NUMPY_INCLUDE_PATH:$CPPFLAGS"
export CPATH="$NUMPY_INCLUDE_PATH:$CPATH"
export C_INCLUDE_PATH="$NUMPY_INCLUDE_PATH:$C_INCLUDE_PATH"
export CPP_INCLUDE_PATH="$NUMPY_INCLUDE_PATH:$CPP_INCLUDE_PATH"

# Set library paths dynamically
export LDFLAGS="-L$PYTHON_ENV/lib:$LDFLAGS"
export LIBRARY_PATH="$PYTHON_ENV/lib:$LIBRARY_PATH"

# Add NumPy library path if needed
NUMPY_LIB_PATH=$(python3 -c "import numpy; print(numpy.__path__[0] + '/core/lib')")
export LDFLAGS="-L$NUMPY_LIB_PATH:$LDFLAGS"
export LIBRARY_PATH="$NUMPY_LIB_PATH:$LIBRARY_PATH"

export CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$MY_OPENCV_PYTHON_PATH/opencv_contrib/modules \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF"

if [[ $version == *"24.04"* ]] ; then
    export CMAKE_ARGS="$CMAKE_ARGS -DBUILD_opencv_sfm=OFF" # It seems this module brings some build issues with Ubuntu 24.04
fi

export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS=$CPPFLAGS"

# build opencv_python 
pip wheel . --verbose
# install built packages
pip install opencv*.whl --force-reinstall

# build opencv_contrib_python
export ENABLE_CONTRIB=1
pip wheel . --verbose
# install built packages
# pip install opencv*.whl --force-reinstall


# Extract the installed OpenCV version and wheel path
OPENCV_WHL_PATH=$(ls opencv*.whl | head -n 1)
OPENCV_PKG_NAME=$(basename "$OPENCV_WHL_PATH" | cut -d'-' -f1)  # usually "opencv_python"

# Write pip constraint file
echo "$OPENCV_PKG_NAME @ file://$MY_OPENCV_PYTHON_PATH/$OPENCV_WHL_PATH" > "$ROOT_DIR/constraints.txt"

print_green "Created constraints.txt:"
cat "$ROOT_DIR/constraints.txt"


cd "$ROOT_DIR"


# HACK for conda issue:
# "ImportError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32"
#  See https://github.com/elerac/EasyPySpin/issues/12
if [[ "$OSTYPE" != "darwin"* ]]; then
    CV2_SO_PATH=$PYTHON_ENV/lib/python$PYTHON_VERSION/site-packages/cv2
    # Find the actual .so file (handles different naming variations)
    CV2_SO_FILE=$(ls "$CV2_SO_PATH"/cv2.*.so 2>/dev/null | head -n 1)
    # Check if a valid file was found
    if [[ -n "$CV2_SO_FILE" && -f "$CV2_SO_FILE" ]]; then
        if ldd "$CV2_SO_FILE" | grep -q "libffi"; then
            if [[ -f /lib/x86_64-linux-gnu/libffi.so ]]; then
                echo "Preloading libffi..."
                export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so
            else
                echo "WARNING: /lib/x86_64-linux-gnu/libffi.so not found; check your libffi dependencies"
            fi
        fi
    fi
fi

cd "$STARTING_DIR"