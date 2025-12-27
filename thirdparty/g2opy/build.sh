#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install -y libsuitesparse-dev libeigen3-dev python3-dev
fi   

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
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
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# Allow overriding Python executable (e.g., conda env); resolved early so we can query its arch
PYTHON_EXE=${Python3_EXECUTABLE:-$(which python3)}
echo "PYTHON_EXE: $PYTHON_EXE"

# Check if we're in a pixi environment
PIXI_ENV_PREFIX=""
if [[ "$PYTHON_EXE" == *".pixi"* ]]; then
    # Extract pixi environment prefix from Python path
    # e.g., /path/to/project/.pixi/envs/default/bin/python3 -> /path/to/project/.pixi/envs/default
    PIXI_ENV_PREFIX=$(dirname $(dirname $(dirname "$PYTHON_EXE")))
    echo "Detected pixi environment at: $PIXI_ENV_PREFIX"
fi

if [ "$CONDA_INSTALLED" = true ]; then
    # NOTE: these are the "system" packages that are needed within conda to build opencv from source
    conda install -y -c conda-forge suitesparse 
fi

# Initialize MAC_OPTIONS and LINUX_OPTIONS to empty strings
MAC_OPTIONS=""
LINUX_OPTIONS=""

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    TARGET_ARCH=${CMAKE_OSX_ARCHITECTURES:-${OSX_ARCH:-${PYTHON_ARCH:-$(uname -m)}}}
    echo "TARGET_ARCH: $TARGET_ARCH"

    # Respect user-requested architecture (e.g., build x86_64 when running under Rosetta)
    MAC_OPTIONS="\
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_OSX_ARCHITECTURES=${TARGET_ARCH} \
    -DOPENGL_opengl_LIBRARY= -DOPENGL_glx_LIBRARY="

    if [ "$CONDA_INSTALLED" = true ]; then
        if [ -z "$CONDA_PREFIX" ]; then
            CONDA_PREFIX=$(conda info --base)
        fi
        echo "CONDA_PREFIX: $CONDA_PREFIX"

        export CMAKE_PREFIX_PATH=$CONDA_PREFIX
        export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
        export CMAKE_IGNORE_PATH=/opt/homebrew
        export CSPARSE_INCLUDE_DIR=$CONDA_PREFIX/include/suitesparse
        export CSPARSE_LIBRARY=$CONDA_PREFIX/lib/libcxsparse.dylib
        export CHOLMOD_INCLUDE_DIR=$CONDA_PREFIX/include/suitesparse
        export CHOLMOD_LIBRARY=$CONDA_PREFIX/lib/libcholmod.dylib

        MAC_OPTIONS+="\
        -DCHOLMOD_INCLUDE_DIR=$CHOLMOD_INCLUDE_DIR -DCHOLMOD_LIBRARY=$CHOLMOD_LIBRARY \
        -DCSPARSE_INCLUDE_DIR=$CSPARSE_INCLUDE_DIR -DCSPARSE_LIBRARY=$CSPARSE_LIBRARY"
    fi


    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

# Handle Linux pixi/conda environments
if [[ "$OSTYPE" == "linux-gnu"* ]] && [[ -n "$PIXI_ENV_PREFIX" ]]; then
    # Check if CSPARSE exists in pixi environment
    if [ -d "$PIXI_ENV_PREFIX/include/suitesparse" ] && ([ -f "$PIXI_ENV_PREFIX/lib/libcxsparse.so" ] || [ -f "$PIXI_ENV_PREFIX/lib/libcxsparse.a" ]); then
        export CSPARSE_INCLUDE_DIR=$PIXI_ENV_PREFIX/include/suitesparse
        # Try to find the library (could be .so or .a)
        if [ -f "$PIXI_ENV_PREFIX/lib/libcxsparse.so" ]; then
            export CSPARSE_LIBRARY=$PIXI_ENV_PREFIX/lib/libcxsparse.so
        elif [ -f "$PIXI_ENV_PREFIX/lib/libcxsparse.a" ]; then
            export CSPARSE_LIBRARY=$PIXI_ENV_PREFIX/lib/libcxsparse.a
        fi
        
        LINUX_OPTIONS+=" -DCSPARSE_INCLUDE_DIR=$CSPARSE_INCLUDE_DIR -DCSPARSE_LIBRARY=$CSPARSE_LIBRARY"
        echo "Using pixi CSPARSE: $CSPARSE_INCLUDE_DIR, $CSPARSE_LIBRARY"
    fi
    
    # Also set CHOLMOD if needed (though it seems to be found correctly already)
    if [ -d "$PIXI_ENV_PREFIX/include/suitesparse" ] && ([ -f "$PIXI_ENV_PREFIX/lib/libcholmod.so" ] || [ -f "$PIXI_ENV_PREFIX/lib/libcholmod.a" ]); then
        export CHOLMOD_INCLUDE_DIR=$PIXI_ENV_PREFIX/include/suitesparse
        if [ -f "$PIXI_ENV_PREFIX/lib/libcholmod.so" ]; then
            export CHOLMOD_LIBRARY=$PIXI_ENV_PREFIX/lib/libcholmod.so
        elif [ -f "$PIXI_ENV_PREFIX/lib/libcholmod.a" ]; then
            export CHOLMOD_LIBRARY=$PIXI_ENV_PREFIX/lib/libcholmod.a
        fi
        
        LINUX_OPTIONS+=" -DCHOLMOD_INCLUDE_DIR=$CHOLMOD_INCLUDE_DIR -DCHOLMOD_LIBRARY=$CHOLMOD_LIBRARY"
        echo "Using pixi CHOLMOD: $CHOLMOD_INCLUDE_DIR, $CHOLMOD_LIBRARY"
    fi
fi

# Handle Linux conda environments
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$CONDA_INSTALLED" = true ] && [[ -z "$PIXI_ENV_PREFIX" ]]; then
    if [ -n "$CONDA_PREFIX" ]; then
        if [ -d "$CONDA_PREFIX/include/suitesparse" ] && ([ -f "$CONDA_PREFIX/lib/libcxsparse.so" ] || [ -f "$CONDA_PREFIX/lib/libcxsparse.a" ]); then
            export CSPARSE_INCLUDE_DIR=$CONDA_PREFIX/include/suitesparse
            if [ -f "$CONDA_PREFIX/lib/libcxsparse.so" ]; then
                export CSPARSE_LIBRARY=$CONDA_PREFIX/lib/libcxsparse.so
            elif [ -f "$CONDA_PREFIX/lib/libcxsparse.a" ]; then
                export CSPARSE_LIBRARY=$CONDA_PREFIX/lib/libcxsparse.a
            fi
            
            LINUX_OPTIONS+=" -DCSPARSE_INCLUDE_DIR=$CSPARSE_INCLUDE_DIR -DCSPARSE_LIBRARY=$CSPARSE_LIBRARY"
            echo "Using conda CSPARSE: $CSPARSE_INCLUDE_DIR, $CSPARSE_LIBRARY"
        fi
    fi
fi

if [[ -n "$LINUX_OPTIONS" ]]; then
    echo "Using LINUX_OPTIONS for cpp build: $LINUX_OPTIONS"
fi

EXTERNAL_OPTIONS+=" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython3_EXECUTABLE=${PYTHON_EXE}" 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

BUILD_TYPE="Release"
echo "BUILD_TYPE: $BUILD_TYPE"

make_dir build
cd build
cmake .. $EXTERNAL_OPTIONS $MAC_OPTIONS $LINUX_OPTIONS -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j 8

cd ..
