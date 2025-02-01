#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used


# ====================================================
# import the bash utils 
. $SCRIPT_DIR/bash_utils.sh 

# ====================================================

#set -e

print_blue '================================================'
print_blue "Configuring CUDA ..."


# detect CUDA VERSION
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(get_cuda_version)
    echo CUDA_VERSION: $CUDA_VERSION

    export CUDA_VERSION_STRING="cuda-"${CUDA_VERSION}  # must be an installed CUDA path in "/usr/local"; 
                                                       # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
    if [ ! -d /usr/local/$CUDA_VERSION_STRING ]; then
        CUDA_VERSION_STRING="cuda"  # use last installed CUDA path in standard path as a fallback 
    fi     
    echo CUDA_VERSION_STRING: $CUDA_VERSION_STRING
    export PATH=/usr/local/$CUDA_VERSION_STRING/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION_STRING/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}    
fi