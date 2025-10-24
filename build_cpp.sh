#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this script allows to build the C++ core of pySLAM

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_"
SCRIPTS_DIR="$ROOT_DIR/scripts"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

#set -e

print_blue '================================================'
print_blue "Building pySLAM C++ core"
print_blue '================================================'

cd "$ROOT_DIR/pyslam/slam/cpp"
./build.sh

cd "$STARTING_DIR"