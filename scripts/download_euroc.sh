#!/usr/bin/env bash

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."
cd "$ROOT_DIR"

mkdir -p datasets/euroc
cd datasets/euroc

wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip
unzip MH_02_easy.zip -d mh02
rm -rf MH_02_easy.zip