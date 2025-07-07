#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME="${1:-pyslam}"  # get the first input if any, otherwise use 'pyslam' as default name

conda remove --name $ENV_NAME --all -y

