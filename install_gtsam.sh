#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


print_blue '================================================'
print_blue "Installing gtsam from source"
print_blue '================================================'

PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

cd thirdparty
if [ ! -d gtsam ]; then
	git clone https://github.com/borglab/gtsam.git gtsam
    #git fetch --all --tags # to fetch tags 
    cd gtsam
    git checkout tags/4.2a9   
    git apply ../gtsam.patch
    cd .. 
fi
cd gtsam
make_buid_dir
TARGET_GTSAM_LIB="install/lib/libgtsam.so"
if [[ "$OSTYPE" == darwin* ]]; then 
    TARGET_GTSAM_LIB="install/lib/libgtsam.dylib"
fi
if [[ ! -f "$TARGET_GTSAM_LIB" ]]; then
	cd build
    # NOTE: gtsam has some issues when compiling with march=native option!
    # https://groups.google.com/g/gtsam-users/c/jdySXchYVQg
    # https://bitbucket.org/gtborg/gtsam/issues/414/compiling-with-march-native-results-in 
    GTSAM_OPTIONS="-DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_BUILD_PYTHON=ON -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES=OFF"
    GTSAM_OPTIONS+=" -DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF -DCMAKE_PYTHON_EXECUTABLE=$(which python) -DGTSAM_PYTHON_VERSION=$PYTHON_VERSION"
    if [[ "$OSTYPE" == darwin* ]]; then
        GTSAM_OPTIONS+=" -DGTSAM_WITH_TBB=OFF"
    fi 
    echo GTSAM_OPTIONS: $GTSAM_OPTIONS
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $GTSAM_OPTIONS $EXTERNAL_OPTION
	make -j $(nproc)
    make install 
fi

echo current folder: $(pwd)

echo "deploying built gtsam module"
PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")    
PYTHON_SITE_PACKAGES=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
PYTHON_SOURCE_FOLDER_GTSAM=$(pwd)/python/gtsam
PYTHON_SOURCE_FOLDER_GTSAM_UNSTABLE=$(pwd)/python/gtsam_unstable
if [[ -d "$PYTHON_SITE_PACKAGES" && -d "$PYTHON_SOURCE_FOLDER_GTSAM" ]]; then
    echo "copying built python gtsam module from $PYTHON_SOURCE_FOLDER_GTSAM to $PYTHON_SITE_PACKAGES"
    if [[ -d "$PYTHON_SITE_PACKAGES/gtsam" ]]; then
        rm -rf $PYTHON_SITE_PACKAGES/gtsam
    fi
    cp -r $PYTHON_SOURCE_FOLDER_GTSAM $PYTHON_SITE_PACKAGES
else
    echo "ERROR: failed to copy build python gtsam module from $PYTHON_SOURCE_FOLDER_GTSAM to $PYTHON_SITE_PACKAGES"  
fi 
if [[ -d "$PYTHON_SITE_PACKAGES" && -d "$PYTHON_SOURCE_FOLDER_GTSAM_UNSTABLE" ]]; then
    echo "copying built python gtsam module from $PYTHON_SOURCE_FOLDER_GTSAM_UNSTABLE to $PYTHON_SITE_PACKAGES"
    if [[ -d "$PYTHON_SITE_PACKAGES/gtsam_unstable" ]]; then
        rm -rf $PYTHON_SITE_PACKAGES/gtsam_unstable
    fi    
    cp -r $PYTHON_SOURCE_FOLDER_GTSAM_UNSTABLE $PYTHON_SITE_PACKAGES
else
    echo "ERROR: failed to copy build python gtsam module from $PYTHON_SOURCE_FOLDER_GTSAM_UNSTABLE to $PYTHON_SITE_PACKAGES"  
fi 

cd "$ROOT_DIR"

print_blue '================================================'
print_blue "Building gtsam_factors"
print_blue '================================================'

cd thirdparty
cd gtsam_factors
./build.sh

cd "$ROOT_DIR"

cd "$STARTING_DIR"