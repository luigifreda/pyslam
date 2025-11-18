#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

# Setup environment for detectron2 installation
$SCRIPT_DIR_/install_detectron2.sh

cd $ROOT_DIR/thirdparty

echo "Installing panopticapi..."
pip install git+https://github.com/cocodataset/panopticapi.git

echo "Installing cityscapesScripts..."
pip install git+https://github.com/mcordts/cityscapesScripts.git


if [ ! -d "eov_segmentation" ]; then
    echo "Cloning eov_segmentation..."
    git clone https://github.com/nhw649/EOV-Seg.git eov_segmentation
    cd eov_segmentation
    git apply ../eov_segmentation.patch
    cd ..
fi

echo "Installing eov_segmentation requirements..."
pip install cython scipy shapely timm h5py submitit scikit-image
pip install open-clip-torch==2.24.0

# download the convnext-l.pth checkpoint for panoptic segmentation
if [ ! -d "eov_segmentation/checkpoints" ]; then
    mkdir -p eov_segmentation/checkpoints
fi 

if [ ! -f "eov_segmentation/checkpoints/convnext-l.pth" ]; then
    cd eov_segmentation/checkpoints
    # https://drive.google.com/file/d/1dVfHpzmCOlV6hLfUpd3nHXz62wdB7RY2/view?pli=1
    gdrive_download "1dVfHpzmCOlV6hLfUpd3nHXz62wdB7RY2" "convnext-l.pth"
fi 

cd $ROOT_DIR
