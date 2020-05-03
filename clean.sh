#!/usr/bin/env bash

# clean thirdparty install and utils 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

set -e

print_blue "=================================================================="
print_blue "clearning thirdparty packages and utils..."

rm -Rf thirdparty/pangolin 
rm -Rf thirdparty/g2opy
rm -Rf cpp/utils/build 