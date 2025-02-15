#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used


# ====================================================
# import the bash utils 
. $SCRIPT_DIR/bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring CUDA ..."


# detect CUDA VERSION
export CUDA_VERSION=0
if command -v nvidia-smi &> /dev/null; then
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
    if [ $? -ne 0 ]; then
        print_red "❌ Installation failed! Try these alternatives:"
        print_red "1. Check available versions: apt-cache search nvidia-cuda-toolkit"
        print_red "2. Before trying again, manually add NVIDIAs repository. For further details, see this link:" 
        print_red "   https://developer.nvidia.com/cuda-toolkit-archive"
        print_red "Exiting..."
        exit 1  # Exit immediately with a critical error
    fi
else
    print_yellow "Skipping nvidia toolkit install since CUDA_VERSION is 0"
fi