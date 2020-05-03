#!/usr/bin/env bash

# clean thirdparty install 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

print_blue "=================================================================="
print_blue "clearning thirdparty packages ..."

rm -Rf thirdparty/pangolin thirdparty/g2opy