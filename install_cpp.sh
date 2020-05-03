#!/usr/bin/env bash


# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

print_blue '================================================'
print_blue "Building and installing cpp ..."

cd cpp 

# build utils 
cd utils 
./build.sh
cd ..

# ... 

