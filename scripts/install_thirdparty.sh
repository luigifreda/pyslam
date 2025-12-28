#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

STARTING_DIR=`pwd`  
cd "$ROOT_DIR" 

# ====================================================

print_blue '================================================'
print_blue "Building Thirdparty"
print_blue '================================================'

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

# ====================================================
# detect and configure CUDA 
. "$ROOT_DIR"/cuda_config.sh

# ====================================================
# activate pyslam python environment
#. "$ROOT_DIR"/pyenv-activate.sh

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
    CONDA_INSTALLED=false
fi

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# check if we want to add a python interpreter check
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi

OpenCV_DIR="$ROOT_DIR/thirdparty/opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

export CONDA_OPTIONS=""
if [ "$CONDA_INSTALLED" = true ]; then
    CONDA_OPTIONS="-DOPENGL_opengl_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenGL.so \
        -DOPENGL_glx_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLX.so"
    echo "Using CONDA_OPTIONS for build: $CONDA_OPTIONS"
fi

EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS $CONDA_OPTIONS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "Currently used python virtual environment: $CURRENT_USED_PYENV"


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/json ..."
# Must be installed before building slam cpp
$SCRIPTS_DIR/install_json_nlohmann.sh $EXTERNAL_OPTIONS

cd "$ROOT_DIR"

exit 0

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/orbslam2_features ..."
cd thirdparty/orbslam2_features
./build.sh $EXTERNAL_OPTIONS
cd "$ROOT_DIR"

print_blue '================================================'
print_blue "Configuring and building thirdparty/pangolin ..."

cd thirdparty/pangolin
./build.sh $EXTERNAL_OPTIONS

cd "$ROOT_DIR"


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/g2o ..."

cd thirdparty/g2opy
./build.sh $EXTERNAL_OPTIONS 

cd $ROOT_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow3 ..."

cd thirdparty/pydbow3
./build.sh $EXTERNAL_OPTIONS

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow2 ..."

cd thirdparty/pydbow2
./build.sh $EXTERNAL_OPTIONS

cd $ROOT_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pyibow ..."

cd thirdparty/pyibow
./build.sh $EXTERNAL_OPTIONS

cd $ROOT_DIR


if [[ "$OSTYPE" == darwin* ]]; then
    print_blue "=================================================================="
    print_blue "Configuring and building thirdparty/open3d ..."

    # NOTE: Under mac I got segmentation faults when trying to use open3d python bindings
    #       This happends when trying to load the open3d dynamic library.
    $SCRIPTS_DIR/install_open3d_python.sh $EXTERNAL_OPTIONS

    cd $ROOT_DIR
fi 

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/gtsam ..."
$SCRIPTS_DIR/install_gtsam.sh $EXTERNAL_OPTIONS

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/ros2_pybindings ..."
cd thirdparty/ros2_pybindings
./build.sh $EXTERNAL_OPTIONS

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/ml_depth_pro ..."

cd thirdparty
if [ ! -d ml_depth_pro ]; then
    git clone https://github.com/apple/ml-depth-pro.git ml_depth_pro
    cd ml_depth_pro
    #git checkout b2cd0d51daa95e49277a9f642f7fd736b7f9e91d # use this commit if you hit any problems

    git apply ../ml_depth_pro.patch

    source get_pretrained_models.sh   # Files will be downloaded to `ml_depth_pro/checkpoints` directory. 
fi

cd $ROOT_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/depth_anything_v2 ..."

cd thirdparty
if [ ! -d depth_anything_v2 ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2
    cd depth_anything_v2
    #git checkout 31dc97708961675ce6b3a8d8ffa729170a4aa273 # use this commit if you hit any problems

    git apply ../depth_anything_v2.patch

    ./download_metric_models.py
fi

cd $ROOT_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/depth_anything_v3 ..."
$SCRIPTS_DIR/install_depth_anything_v3.sh

cd $ROOT_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/raft_stereo ..."

cd thirdparty
if [ ! -d raft_stereo ]; then
    git clone https://github.com/princeton-vl/RAFT-Stereo.git raft_stereo
    cd raft_stereo
    #git checkout 6068c1a26f84f8132de10f60b2bc0ce61568e085 # use this commit if you hit any problems

    git apply ../raft_stereo.patch
    
    ./download_models.sh
fi

cd $ROOT_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/crestereo ..."

cd thirdparty
if [ ! -d crestereo ]; then
    git clone https://github.com/megvii-research/CREStereo.git crestereo
    cd crestereo
    #git checkout ad3a1613bdedd88b93247e5f002cb7c80799762d # use this commit if you hit any problems

    git apply ../crestereo.patch
    
    ./download_models.py
fi

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/crestereo_pytorch ..."

cd thirdparty
if [ ! -d crestereo_pytorch ]; then
    git clone https://github.com/ibaiGorordo/CREStereo-Pytorch.git crestereo_pytorch
    cd crestereo_pytorch
    #git checkout b6c7a9fe8dc2e9e56ba7b96f4677312309282d15 # use this commit if you hit any problems

    git apply ../crestereo_pytorch.patch
    
    ./download_models.py
fi

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/mast3r ..."

if [ "$CUDA_VERSION" != "0" ]; then
    # we need CUDA

    cd thirdparty
    if [ ! -d mast3r ]; then
        git clone --recursive https://github.com/naver/mast3r mast3r
        git checkout e06b0093ddacfd8267cdafe5387954a650af0d3b
        cd mast3r
        git apply ../mast3r.patch
        # DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
        cd dust3r 
        git apply ../../mast3r-dust3r.patch
        cd croco 
        git apply ../../../mast3r-dust3r-croco.patch
        cd models/curope/
        if [ "$CUDA_VERSION" != "0" ]; then        
            python setup.py build_ext --inplace
        fi
        cd ../../../../    
        make_dir checkpoints
        cd checkpoints
        if [ ! -f MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth ]; then
            wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
        fi
    fi 
else
    print_yellow "MASt3R requires CUDA. Skipping..."
fi 

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/mvdust3r ..."

if [ "$CUDA_VERSION" != "0" ]; then
    # we need CUDA

    cd thirdparty
    if [ ! -d mvdust3r ]; then
        git clone https://github.com/facebookresearch/mvdust3r.git mvdust3r
        git checkout 430ca6630b07567cfb2447a4dcee9747b132d5c7
        cd mvdust3r
        git apply ../mvdust3r.patch
        # DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
        cd croco/models/curope/
        if [ "$CUDA_VERSION" != "0" ]; then        
            python setup.py build_ext --inplace
        fi
        cd ../../../
        make_dir checkpoints
        cd checkpoints
        cp $ROOT_DIR/thirdparty/mvdust3r_scripts/download_models.py .
        python download_models.py
    fi

else 
    print_yellow "MASt3R requires CUDA. Skipping..."
fi

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/vggt ..."

if [ "$CUDA_VERSION" != "0" ]; then
    # we need CUDA

    cd thirdparty
    if [ ! -d vggt ]; then
        git clone https://github.com/facebookresearch/vggt.git vggt
    fi 
else
    print_yellow "VGGT requires CUDA. Skipping..."
fi 

cd $ROOT_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/vggt_robust ..."

if [ "$CUDA_VERSION" != "0" ]; then
    # we need CUDA

    cd thirdparty
    if [ ! -d vggt_robust ]; then
        git clone https://github.com/cvlab-kaist/RobustVGGT.git vggt_robust
        cd vggt_robust
        git checkout 0763ed6484b1e91a2b8bd5072d317745743492cc
        cd ..
    fi 
else
    print_yellow "VGGT Robust requires CUDA. Skipping..."
fi 

cd $ROOT_DIR

print_blue "=================================================================="

echo "...done with thirdparty"

cd "$STARTING_DIR"


# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON