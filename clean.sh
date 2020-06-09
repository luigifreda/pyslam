#!/usr/bin/env bash

# clean thirdparty install and compiled libraries  

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

print_blue "=================================================================="
print_blue "clearning thirdparty packages and utils..."

rm -Rf thirdparty/pangolin 

rm -Rf thirdparty/g2opy

rm -Rf thirdparty/protoc                   # set by install_delf.sh 

rm -Rf thirdparty/orbslam2_features/build
rm -Rf thirdparty/orbslam2_features/lib

rm -Rf cpp/utils/build  
rm -Rf cpp/utils/lib 


# TODO
# clean downloaded models 