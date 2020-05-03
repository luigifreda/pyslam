#!/usr/bin/env bash

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

STARTING_DIR=`pwd`

make_dir build
cd build
cmake ..
make -j 4

cd $STARTING_DIR
