#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME="${1:-pyslam}"  # get the first input if any, otherwise use 'pyslam' as default name


# # Check if conda is already initialized
# if ! conda info -v &> /dev/null; then
#     conda init bash 
#     source ~/.bashrc
# fi


SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used


# This variable is used to indicate that we want to use conda
export USING_CONDA_PYSLAM=1

export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 

if [ -z "$CONDA_PREFIX" ]; then
    CONDA_PREFIX=$(conda info --base)
fi
. "$CONDA_PREFIX"/bin/activate base   # from https://community.anaconda.cloud/t/unable-to-activate-environment-prompted-to-run-conda-init-before-conda-activate-but-it-doesnt-work/68677/10
conda activate $ENV_NAME

# N.B.: in order to deactivate the virtual environment run: 
# $ conda deactivate


PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

# HACK for "ImportError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32"
# See https://github.com/elerac/EasyPySpin/issues/12
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


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

ubuntu_version=$(lsb_release -rs | cut -d. -f1)
gcc_version=$(gcc -dumpversion | cut -d. -f1)
#echo "Ubuntu version: $ubuntu_version, GCC version: $gcc_version"

# for conda potentially needed under Ubuntu 22.04 and 24.04
if [[ "$ubuntu_version" == "22" || "$ubuntu_version" == "24" ]]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1
    echo "Preloading libstdc++..."
fi 

if [[ "$ubuntu_version" == "22" ]]; then
    # find the following libraries in $CONDA_PREFIX/lib and add them individually to LD_LIBRARY_PATH
    LIBRARIES_TO_PRELOAD="libjpeg.so libgio-2.0.so"
    export LD_PRELOAD=$(python3 $SCRIPT_DIR_/set_conda_preload.py $LIBRARIES_TO_PRELOAD):$LD_PRELOAD
else 
    # export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
    # echo "Setting CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" # This solves some issues but it may cause other issues (wrong linking to conda libraries under ubuntu 22.04) 
    echo "Setting LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
fi


# Check if the Ubuntu version is 20.04 and GCC version is less than 11
# If so, set GCC to version 11 to avoid linking errors due undefined reference to new GLIBCXX stuff
if [[ "$ubuntu_version" == "20" && "$gcc_version" -lt 11 ]]; then
    if [[ -f /usr/bin/gcc-11 ]]; then
        echo "Setting GCC to version 11"
        export CC=/usr/bin/gcc-11
        export CXX=/usr/bin/g++-11
    fi
fi