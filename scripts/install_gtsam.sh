#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"


# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
    CONDA_INSTALLED=false
fi


ubuntu_version=$(lsb_release -rs | cut -d. -f1)
gcc_version=$($CC -dumpversion | cut -d. -f1)
if [[ "$CONDA_INSTALLED" == true && "$ubuntu_version" == "20" && "$gcc_version" == 11 ]]; then
    print_blue "Setting GCC and G++ to version 9"
    export CC=/usr/bin/gcc-9
    export CXX=/usr/bin/g++-9
fi

print_blue '================================================'
print_blue "Installing gtsam from source"
print_blue '================================================'

PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

cd thirdparty
if [ ! -d gtsam_local ]; then
	git clone https://github.com/borglab/gtsam.git gtsam_local
    #git fetch --all --tags # to fetch tags 
    cd gtsam_local
    git checkout tags/4.2a9   
    git apply ../gtsam.patch
    cd .. 
fi
cd gtsam_local
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
    if [[ "$version" == *"24.04"* ]] ; then
        # Ubuntu 24.04 requires CMake 3.22 or higher
        GTSAM_OPTIONS+=" -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    fi
    GTSAM_OPTIONS+=" -DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF -DCMAKE_PYTHON_EXECUTABLE=$(which python) -DGTSAM_PYTHON_VERSION=$PYTHON_VERSION"
    if [[ "$OSTYPE" == darwin* ]]; then
        GTSAM_OPTIONS+=" -DGTSAM_WITH_TBB=OFF"
    fi 
    echo GTSAM_OPTIONS: $GTSAM_OPTIONS
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $GTSAM_OPTIONS $EXTERNAL_OPTIONS
	make -j $(nproc)
    make install 

    # Now install gtsam python package
    make python-install 
fi

echo current folder: $(pwd)

cd "$ROOT_DIR"

print_blue '================================================'
print_blue "Building gtsam_factors"
print_blue '================================================'

cd thirdparty
cd gtsam_factors
./build.sh $EXTERNAL_OPTIONS

cd "$ROOT_DIR"

cd "$STARTING_DIR"