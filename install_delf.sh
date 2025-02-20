#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

#echo ROOT_DIR: $ROOT_DIR
cd "$ROOT_DIR"  # from bash_utils.sh

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo


#set -e

# N.B.: this must be run after having run the script install_git_modules.sh  

print_blue '================================================'
print_blue "Installing delf ..."

cd thirdparty 
make_dir protoc
cd protoc 


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ ! -f protoc-3.3.0-linux-x86_64.zip ]; then 
        echo 'installing protoc'
        wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
        unzip protoc-3.3.0-linux-x86_64.zip
    fi
    PATH_TO_PROTOC=`pwd`/bin/protoc  
fi 
if [[ "$OSTYPE" == "darwin"* ]]; then
    PROTOC_ZIP=protoc-3.7.1-osx-x86_64.zip
    if [ ! -f $PROTOC_ZIP ]; then 
        echo 'installing protoc'
        curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
        sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
        sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
    fi 
    PATH_TO_PROTOC=/usr/local/bin/protoc 
fi 


cd $STARTING_DIR

cd thirdparty/tensorflow_models/research/delf 
${PATH_TO_PROTOC} delf/protos/*.proto --python_out=.

PARAMETERS_DIR=delf/python/examples/parameters/
mkdir -p $PARAMETERS_DIR
if [ ! -f $PARAMETERS_DIR/delf_gld_20190411.tar.gz ]; then 
    echo 'downloading delf model'
    wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz -O $PARAMETERS_DIR/delf_gld_20190411.tar.gz
    tar -C $PARAMETERS_DIR -xvf $PARAMETERS_DIR/delf_gld_20190411.tar.gz
fi 

cd $STARTING_DIR
