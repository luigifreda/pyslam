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

print_blue "Cleaning Python caches in: $ROOT_DIR"

echo "Removing __pycache__ folders"
# Remove all __pycache__ folders
find "$ROOT_DIR/pyslam" -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

#echo "Removing .pyc and .pyo files"
# Optionally, remove .pyc and .pyo files
#find "$ROOT_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null

echo "Cleanup completed."