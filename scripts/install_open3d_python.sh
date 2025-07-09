#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

# NOTE: this is required under mac where I got unexpected segmentation fault errors
#       on open3d dynamic library loading

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  

print_blue "Installing open3d-python from source"

# check if HAVE_CUDA is set, if not, source the cuda_config.sh script
if [ -z "$HAVE_CUDA" ]; then
    source $ROOT_DIR/cuda_config.sh
else
    echo "HAVE_CUDA is already set to $HAVE_CUDA"
fi

if [[ $OSTYPE == "darwin"* ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

    export LDFLAGS="-L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/openblas/include -I/opt/homebrew/Cellar/minizip/1.3.1/include/minizip/ -I/opt/homebrew/include/libpng16"

    export CPLUS_INCLUDE_PATH="/opt/homebrew/opt/openblas/include:/opt/homebrew/Cellar/minizip/1.3.1/include/minizip/"
    export LIBRARY_PATH="/opt/homebrew/opt/openblas/lib"    
fi

#pip3 install --upgrade pip
pip3 uninstall -y open3d

cd thirdparty
if [ ! -d open3d ]; then
    git clone https://github.com/isl-org/Open3D.git open3d
    cd open3d
    git checkout 02674268f706be4b004bbbf3d39b95fa9de35f74

    if [[ $OSTYPE == "darwin"* ]]; then
        git apply ../open3d.patch
    fi

    cd ..
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    num_cores=$(sysctl -n hw.physicalcpu)
else 
    num_cores=$(nproc)
fi

cd open3d
if [ ! -d build ]; then
    mkdir build 
fi 

if [ ! -d install ]; then
    if [ ! -d build ]; then
        mkdir build 
    fi

    cd build

    if [ $HAVE_CUDA == 1 ]; then 
        EXTERNAL_OPTIONS+=" -DBUILD_CUDA_MODULE=ON -DBUILD_COMMON_CUDA_ARCHS=ON "
    fi

    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON \
        -DUSE_SYSTEM_ASSIMP=ON -DUSE_SYSTEM_VTK=ON -DUSE_SYSTEM_BLAS=ON -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_PNG=ON -DUSE_SYSTEM_TBB=ON \
        $EXTERNAL_OPTIONS -DCMAKE_INSTALL_PREFIX="`pwd`/../install"

    make -j$num_cores

    # Activate the virtualenv first
    # Install pip package in the current python environment
    make install-pip-package -j$num_cores

    # Create Python package in build/lib
    make python-package -j$num_cores

    # Create pip wheel in build/lib
    # This creates a .whl file that you can install manually.
    make pip-package -j$num_cores

    $SUDO make install
fi


cd "$STARTING_DIR"