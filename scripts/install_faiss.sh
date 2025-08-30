#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

cd "$ROOT_DIR" 

# ====================================================


# Check if faiss-gpu-cu12 exists in pip index
if pip index versions faiss-gpu-cu12 >/dev/null 2>&1; then
    echo "faiss-gpu-cu12 found in pip index."
    echo "Uninstalling faiss-cpu if present..."
    pip uninstall -y faiss faiss-cpu || true
    echo "Installing faiss-gpu-cu12..."
    pip install faiss-gpu-cu12
    echo "Done: faiss-gpu-cu12 installed."
else
    echo "faiss-gpu-cu12 not available for this Python/CUDA version."
    # check if faiss-cpu was installed
    if pip show faiss-cpu >/dev/null 2>&1; then
        echo "faiss-cpu was installed. Uninstalling it..."
        pip uninstall -y faiss-cpu
    fi
    echo "Installing faiss-cpu..."
    pip install faiss-cpu
    echo "Done: faiss-cpu installed."
fi