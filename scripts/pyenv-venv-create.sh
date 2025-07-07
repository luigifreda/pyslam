#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this install script allows you to run main_slam.py and all the scripts 

set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

STARTING_DIR=`pwd`
cd "$ROOT_DIR"


case "$OSTYPE" in
  darwin*)
    echo "macOS"
    . $SCRIPTS_DIR/pyenv-venv-create-mac.sh
    ;;
  linux*)
    echo "Linux"
    . $SCRIPTS_DIR/pyenv-venv-create-linux.sh
    ;;
  *)
    echo "Unknown OS"
    ;;
esac

cd "$STARTING_DIR"