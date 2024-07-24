#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR/.."

. "$ROOT_DIR"/pyenv-activate.sh 

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 -O "$ROOT_DIR"/main_slam.py
