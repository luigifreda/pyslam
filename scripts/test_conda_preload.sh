#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

# Example: request jpeg + tiff
export LD_PRELOAD=$(python3 set_conda_preload.py libjpeg.so libtiff.so)

# Check result
echo LD_PRELOAD: $LD_PRELOAD