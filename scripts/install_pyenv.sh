#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

STARTING_DIR=`pwd`  
cd "$ROOT_DIR" 

# ====================================================

print_blue '================================================'
print_blue "Installing pyenv"
print_blue '================================================'

if ! pyenv install --list &> /dev/null; then
    echo "pyenv was not found => installing it..."

    if [ ! -d ~/.pyenv ]; then 
        git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    else
        echo "folder ~/.pyenv is already set"
    fi  

    if grep -q "PYENV_ROOT" "~/.bashrc"; then 
        echo "already found pyenv settings in ~/.bashrc"  
    else 
        echo "adding pyenv setting to bashrc"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init --path)"\nfi' >> ~/.bashrc
    fi 

    source ~/.bashrc
    export PATH=~/.pyenv/shims:$PATH
fi

cd $STARTING_DIR