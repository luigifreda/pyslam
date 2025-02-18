#!/usr/bin/env bash

# NOTE: This script is expected to be used under mac. See the file `docs/MAC.md`.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR/.."

. "$ROOT_DIR"/pyenv-activate.sh 

#export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
#OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 -O "$ROOT_DIR"/main_slam.py
python3 -O "$ROOT_DIR"/main_slam.py