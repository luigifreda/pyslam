#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

set -e


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