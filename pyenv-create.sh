#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

set -e


case "$OSTYPE" in
  darwin*)
    echo "macOS"
    . pyenv-mac-create.sh
    ;;
  linux*)
    echo "Linux"
    . pyenv-linux-create.sh
    ;;
  *)
    echo "Unknown OS"
    ;;
esac