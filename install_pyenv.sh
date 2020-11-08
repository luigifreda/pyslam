#!/usr/bin/env bash

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Installing pyenv"
print_blue '================================================'

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo


if ! pyenv install --list &> /dev/null; then
    echo "pyenv could not be found => install it!"

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
        echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
    fi 

    source ~/.bashrc
    export PATH=~/.pyenv/shims:$PATH
fi