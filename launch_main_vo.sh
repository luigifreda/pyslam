#!/usr/bin/env bash

. pyenv-activate.sh 

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 -O main_vo.py
