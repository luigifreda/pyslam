#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# clean thirdparty install and compiled libraries  

# get the first input if any 
HARD_CLEAN=0
if [ "$1" == "--hard" ]; then
    HARD_CLEAN=1
fi


# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

set -e

print_blue "=================================================================="
print_blue "Cleaning thirdparty packages and utils..."

#echo ROOT_DIR: $ROOT_DIR
cd "$ROOT_DIR"  # from bash_utils.sh

#rm -Rf thirdparty/pangolin 
rm -Rf thirdparty/pangolin/build
rm -Rf thirdparty/pangolin/bin
rm -f thirdparty/pangolin/pypangolin.cpython-*.so 

#rm -Rf thirdparty/g2opy
rm -Rf thirdparty/g2opy/build
rm -Rf thirdparty/g2opy/lib
rm -Rf thirdparty/g2opy/bin

rm -Rf thirdparty/protoc                   # set by install_delf.sh 

if [ -d "thirdparty/raft_stereo" ]; then
    rm -Rf thirdparty/raft_stereo
fi 
if [ -d "thirdparty/ml_depth_pro" ]; then
    rm -Rf thirdparty/ml_depth_pro
fi 
if [ -d "thirdparty/depth_anything_v2" ]; then
    rm -Rf thirdparty/depth_anything_v2
fi
if [ -d "thirdparty/depth_anything_v3" ]; then
    rm -Rf thirdparty/depth_anything_v3
fi
if [ -d "thirdparty/crestereo" ]; then
    rm -Rf thirdparty/crestereo
fi
if [ -d "thirdparty/crestereo_pytorch" ]; then
    rm -Rf thirdparty/crestereo_pytorch
fi

rm -Rf thirdparty/orbslam2_features/build
rm -Rf thirdparty/orbslam2_features/lib

rm -Rf cpp/build  
rm -Rf cpp/lib 

if [ -d "thirdparty/opencv-python" ]; then
    rm -Rf "thirdparty/opencv-python"
fi

if [ -d "thirdparty/open3d" ]; then
    rm -Rf thirdparty/open3d
fi

if [ -d "thirdparty/opencv" ]; then
    rm -Rf thirdparty/opencv
fi

if [ -f constraints.txt ]; then
    rm constraints.txt
fi

cd thirdparty/pydbow3
./clean.sh
cd "$ROOT_DIR"

cd thirdparty/pydbow2
./clean.sh
cd "$ROOT_DIR"

cd thirdparty/pyibow
./clean.sh
cd "$ROOT_DIR"

if [ -d "thirdparty/gtsam_local" ]; then
    rm -Rf thirdparty/gtsam_local
fi
cd "$ROOT_DIR"

if [ -d "thirdparty/gtsam_factors" ]; then
    cd thirdparty/gtsam_factors
    ./clean.sh
    cd "$ROOT_DIR"
fi

if [ -d "thirdparty/lietorch" ]; then
    cd thirdparty/lietorch
    ./clean.sh
    cd "$ROOT_DIR"
fi

if [ -d "thirdparty/monogs/submodules/simple-knn/build" ]; then
    rm -rf thirdparty/monogs/submodules/simple-knn/build thirdparty/monogs/submodules/simple-knn/*.egg-info
fi

if [ -d "thirdparty/monogs/submodules/diff-gaussian-rasterization/build" ]; then
    rm -rf thirdparty/monogs/submodules/diff-gaussian-rasterization/build thirdparty/monogs/submodules/diff-gaussian-rasterization/*.egg-info
fi

if [ -d "thirdparty/mast3r" ]; then
    rm -rf thirdparty/mast3r
fi

if [ -d "thirdparty/mvdust3r" ]; then
    rm -rf thirdparty/mvdust3r
fi

if [ -d "thirdparty/ros2_pybindings" ]; then
    cd thirdparty/ros2_pybindings
    ./clean.sh
    cd "$ROOT_DIR"
fi

if [ -d "thirdparty/vggt" ]; then
    rm -rf thirdparty/vggt
fi

if [ -d "thirdparty/detectron2" ]; then
    rm -rf thirdparty/detectron2
fi

if [ -d "thirdparty/detic" ]; then
    rm -rf thirdparty/detic
fi

if [ -d "thirdparty/eov_segmentation" ]; then
    rm -rf thirdparty/eov_segmentation
fi

if [ -d "thirdparty/odise" ]; then
    rm -rf thirdparty/odise
fi

if [ -d "$ROOT_DIR/.pyslam.egg-info" ]; then
    echo "Removing pyslam.egg-info directory"
    rm -rf "$ROOT_DIR/.pyslam.egg-info"
fi

if [ -f "$ROOT_DIR/.env" ]; then
    echo "Removing $ROOT_DIR/.env file"
    rm "$ROOT_DIR/.env"
fi

if [ -d "thirdparty/megengine" ]; then
    echo "Removing thirdparty/megengine directory"
    sudo rm -rf thirdparty/megengine
fi

if [ -d "thirdparty/json" ]; then
    echo "Removing thirdparty/json directory"
    sudo rm -rf thirdparty/json
fi

# clean downloaded models, reset submodules and clean repo itself
if [ $HARD_CLEAN -eq 1 ]; then
    print_blue "Resetting git submodules"
    git submodule foreach 'git reset --hard; git clean -xfd'
    print_blue "Cleaning repo itself"
    git clean -xfd
fi