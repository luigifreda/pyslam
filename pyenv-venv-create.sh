#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

STARTING_DIR=`pwd`
cd "$ROOT_DIR"


case "$OSTYPE" in
  darwin*)
    echo "macOS"
    . pyenv-venv-create-mac.sh
    ;;
  linux*)
    echo "Linux"
    . pyenv-venv-create-linux.sh 
    ;;
  *)
    echo "Unknown OS"
    ;;
esac

cd "$STARTING_DIR"