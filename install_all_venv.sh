#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# import the utils 
. bash_utils.sh 

set -e

# if we are not under docker
if [ ! -f /.dockerenv  ]; then 
  # Provide the password to sudo once at the start (hopefully...)
  echo "Insert your sudo password" 
  sudo -S -v    # not working properly under mac
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Under conda, you have to use install_all_conda.sh script"
    exit
fi

case "$OSTYPE" in
  darwin*)
    echo "macOS"
    . install_all_venv_mac.sh
    ;;
  linux*)
    echo "Linux"
    . install_all_venv_linux.sh
    ;;
  *)
    echo "Unknown OS"
    ;;
esac