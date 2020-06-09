#!/usr/bin/env bash


# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo


#set -e

# N.B.: this must be run after having run the script install_git_modules.sh  

print_blue '================================================'
print_blue "Installing delf ..."

cd thirdparty 
make_dir protoc
cd protoc 

if [ ! -f protoc-3.3.0-linux-x86_64.zip ]; then 
    echo 'installing protoc'
    wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
    unzip protoc-3.3.0-linux-x86_64.zip
fi 
PATH_TO_PROTOC=`pwd`

cd $STARTING_DIR

cd thirdparty/tensorflow_models/research/delf 
${PATH_TO_PROTOC}/bin/protoc delf/protos/*.proto --python_out=.

PARAMETERS_DIR=delf/python/examples/parameters/
mkdir -p $PARAMETERS_DIR
if [ ! -f $PARAMETERS_DIR/delf_gld_20190411.tar.gz ]; then 
    echo 'downloading delf model'
    wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz -O $PARAMETERS_DIR/delf_gld_20190411.tar.gz
    tar -C $PARAMETERS_DIR -xvf $PARAMETERS_DIR/delf_gld_20190411.tar.gz
fi 

cd $STARTING_DIR