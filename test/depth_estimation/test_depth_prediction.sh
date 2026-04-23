#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

dataset=$SCRIPT_DIR/../data
python $SCRIPT_DIR/test_depth_prediction_from_image_or_stereo_pair.py --image "$dataset/kitti06-12-color.png" \
    --image_right "$dataset/kitti06-12-R-color.png" \
    --environment outdoor \
    --balance-stereo-light 