#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# import the utils 
. bash_utils.sh 

set -e

# if we are not under docker
if [ ! -f /.dockerenv  ]; then 
  # Provide the password to sudo once at the start
  echo "insert your sudo password once" 
  sudo -S -v    # not working properly under mac
fi


case "$OSTYPE" in
  darwin*)
    echo "macOS"
    . install_all_mac_venv.sh
    ;;
  linux*)
    echo "Linux"
    . install_all_linux_venv.sh
    ;;
  *)
    echo "Unknown OS"
    ;;
esac