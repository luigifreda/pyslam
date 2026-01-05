#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."
SCRIPTS_DIR="$ROOT_DIR/scripts"

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
    #echo "Conda is not installed"
    CONDA_INSTALLED=false
fi

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
    echo "Pixi environment detected: $PIXI_PROJECT_NAME"

    source "$SCRIPTS_DIR/pixi_python_config.sh"
else
    PIXI_ACTIVATED=false
fi

# ====================================================

ubuntu_version=$(lsb_release -rs | cut -d. -f1)

# Check if CC is set and available, otherwise use default gcc
if command -v "$CC" &> /dev/null; then
    gcc_version=$($CC -dumpversion | cut -d. -f1)
elif command -v gcc &> /dev/null; then
    gcc_version=$(gcc -dumpversion | cut -d. -f1)
else
    print_red "Error: No C compiler found. Please install gcc or a equivalent compiler."
fi
echo "gcc_version: $gcc_version"

if [[ "$CONDA_INSTALLED" == true && "$ubuntu_version" == "20" && "$gcc_version" == 11 ]]; then
    print_blue "Setting GCC and G++ to version 9"
    export CC=/usr/bin/gcc-9
    export CXX=/usr/bin/g++-9
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++"

    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

# print_blue '================================================'
print_blue "Installing qhull"
# print_blue '================================================'

PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

cd "$ROOT_DIR"

cd thirdparty
if [ ! -d qhull ]; then
    git clone https://github.com/qhull/qhull.git qhull
    #git fetch --all --tags # to fetch tags 
    cd qhull
    git fetch --all --tags
    git checkout v8.0.2
    cd .. 
fi
cd qhull
make_buid_dir
if [[ ! -d install ]]; then
	cd build
    # Set C++17 standard for qhull build (qhull has C++ components in libqhullcpp)
    # Since qhull's CMakeLists.txt doesn't use CMAKE_CXX_STANDARD, we need to add -std=c++17 directly to CMAKE_CXX_FLAGS
    # This ensures the C++ compiler uses C++17 standard (works for both GCC and Clang)
    # BUILD_STATIC_LIBS=ON is needed to build libqhullcpp (C++ interface library)
    # Even though we want shared libraries, we need static libs for the C++ interface
    QHULL_OPTIONS="-DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON"
    # Add -std=c++17 to CMAKE_CXX_FLAGS to ensure it's actually used
    # Note: This sets the initial value; qhull's CMakeLists.txt doesn't modify CMAKE_CXX_FLAGS, so this is safe
    CXX_STD_FLAG="-DCMAKE_CXX_FLAGS=-std=c++17"
    # Put CXX_STD_FLAG after EXTERNAL_OPTIONS so it takes precedence
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $QHULL_OPTIONS $EXTERNAL_OPTIONS $MAC_OPTIONS $CXX_STD_FLAG
    
    # Quick verification: Check CMakeCache.txt for C++17 flag
    print_blue "Verifying C++17 configuration..."
    if grep -q "std=c++17" CMakeCache.txt 2>/dev/null; then
        print_blue "✓ Found -std=c++17 in CMakeCache.txt"
        CXX_FLAGS=$(grep "^CMAKE_CXX_FLAGS:" CMakeCache.txt 2>/dev/null | cut -d'=' -f2- | head -1)
        print_blue "  CMAKE_CXX_FLAGS: $CXX_FLAGS"
    else
        print_red "✗ Warning: -std=c++17 not found in CMakeCache.txt"
        print_red "  Run './scripts/verify_qhull_cpp17.sh' after build for detailed verification"
    fi
    
	make -j 8
    make install 
fi


cd "$ROOT_DIR"

cd "$STARTING_DIR"