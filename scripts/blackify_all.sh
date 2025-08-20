#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this install script allows you to run main_slam.py and all the scripts 

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."
cd $ROOT_DIR

# ====================================================
# import the utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

BLACK_PATH=$(which black)

if [ -z "$BLACK_PATH" ]; then
    print_red "black not found"
    exit 1
fi

print_blue "blackifying pyslam"
$BLACK_PATH pyslam


print_blue "blackifying all main scripts"
MAIN_SCRIPTS=$(find . -name "main_*.py" | xargs)

for script in "${MAIN_SCRIPTS[@]}"; do
    $BLACK_PATH $script
done