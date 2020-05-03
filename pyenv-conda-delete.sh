#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME='pyslam'
fi

conda remove --name $ENV_NAME --all

