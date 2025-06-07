#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring CUDA ..."


# detect CUDA VERSION
export CUDA_VERSION=0


HAVE_CUDA=0
if command -v nvidia-smi &> /dev/null; then
    HAVE_CUDA=1
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    HAVE_CUDA=1   # this branch is needed at docker build time where nvidia-smi is not available
fi

echo "HAVE_CUDA=${HAVE_CUDA}"

if [ $HAVE_CUDA -eq 1 ]; then
    CUDA_VERSION=$(get_cuda_version)
    echo CUDA_VERSION: $CUDA_VERSION

    if [ "$CUDA_VERSION" != "0" ]; then
        export CUDA_FOLDER_STRING="cuda-"${CUDA_VERSION}  # must be an installed CUDA path in "/usr/local"; 
                                                        # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
        if [ ! -d /usr/local/$CUDA_FOLDER_STRING ]; then
            CUDA_FOLDER_STRING="cuda"  # use last installed CUDA path in standard path as a fallback 
        fi     
        #echo CUDA_FOLDER_STRING: $CUDA_FOLDER_STRING
        export CUDA_HOME="/usr/local/$CUDA_FOLDER_STRING"
        export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
        echo CUDA_HOME: $CUDA_HOME
        export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}    
        export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
        export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH     

        export CUDA_VERSION_STRING_WITH_HYPHENS=$(replace_dot_with_hyphen $CUDA_VERSION)
        echo CUDA_VERSION_STRING_WITH_HYPHENS: $CUDA_VERSION_STRING_WITH_HYPHENS   

        export CUDA_VERSION_STRING_COMPACT=$(remove_dots $CUDA_VERSION)
        echo CUDA_VERSION_STRING_COMPACT: $CUDA_VERSION_STRING_COMPACT               
    fi
fi

# check the nvidia toolkit is available and install it 
if [ "$CUDA_VERSION" != "0" ]; then
    print_blue "Checking the nvidia toolkit ..."
    sudo apt-get install -y cuda-toolkit-$CUDA_VERSION_STRING_WITH_HYPHENS
    INSTALL_CUDA_TOOLKIT_STATUS=$?  # Store the exit status of the last command (apt-get install)
    if [ $INSTALL_CUDA_TOOLKIT_STATUS -ne 0 ]; then
        print_yellow "Installation of cuda-toolkit-$CUDA_VERSION_STRING_WITH_HYPHENS failed!"
        print_yellow "Something can go wrong in the install process. Please:" 
        print_yellow "1. Check you have an available cuda-toolkit versions with:$ apt-cache search cuda-toolkit"
        print_yellow "2. Before trying again, manually add NVIDIAs repository. For further details, see this link:" 
        print_yellow "   https://developer.nvidia.com/cuda-toolkit-archive"
        #print_red "Exiting..."
        #exit 1  # Exit immediately with a critical error
        sleep 2
    fi

    if [ $INSTALL_CUDA_TOOLKIT_STATUS -ne 0 ]; then
        print_blue "Checking the package nvidia-cuda-dev ..."
        # If the installation of cuda-toolkit failed, then install nvidia-cuda-dev to avoid the issue "Fatal error: cuda.h: No such file or directory"
        # This problem seems to be critical under CUDA 12.8
        sudo apt-get install -y nvidia-cuda-dev 
    fi

else
    print_yellow "Skipping nvidia toolkit install since CUDA_VERSION is 0"
fi