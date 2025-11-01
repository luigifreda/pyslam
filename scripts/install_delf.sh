#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam


SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


#set -e

# N.B.: this must be run after having run the script install_git_modules.sh  

print_blue '================================================'
print_blue "Installing delf ..."

cd thirdparty 
make_dir protoc
cd protoc 


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PROTOC_VERSION=3.20.0  # before it was 3.3.0
    PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip
    if [ ! -f "${PROTOC_ZIP}" ]; then 
        echo 'installing protoc'
        wget https://github.com/google/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP}
        unzip "${PROTOC_ZIP}"
    fi
    PATH_TO_PROTOC=`pwd`/bin/protoc  
fi 
if [[ "$OSTYPE" == darwin* ]]; then
    PROTOC_VERSION=3.20.0  # before it was 3.7.1
    PROTOC_ZIP=protoc-${PROTOC_VERSION}-osx-x86_64.zip
    if [ ! -f "${PROTOC_ZIP}" ]; then 
        echo 'installing protoc'
        curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP}
        sudo unzip -o ${PROTOC_ZIP} -d /usr/local bin/protoc
        sudo unzip -o ${PROTOC_ZIP} -d /usr/local 'include/*'
    fi 
    PATH_TO_PROTOC=/usr/local/bin/protoc 
fi 

cd "$ROOT_DIR"

cd thirdparty/tensorflow_models/research/delf 
${PATH_TO_PROTOC} delf/protos/*.proto --python_out=.

PARAMETERS_DIR=delf/python/examples/parameters/
mkdir -p $PARAMETERS_DIR
if [ ! -f $PARAMETERS_DIR/delf_gld_20190411.tar.gz ]; then 
    echo 'downloading delf model'
    wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz -O $PARAMETERS_DIR/delf_gld_20190411.tar.gz
    tar -C $PARAMETERS_DIR -xvf $PARAMETERS_DIR/delf_gld_20190411.tar.gz
fi 


cd "$STARTING_DIR"
