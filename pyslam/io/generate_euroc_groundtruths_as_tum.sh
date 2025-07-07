#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir 
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
cd $SCRIPT_DIR # this brings us in the actual used folder (not the symbolic one)


TIME_SCALE_FACTOR=1e+9 # used to multiply the timestamps and bring them to TUM timestamp scale 

EUROC_DATASETS_PATH=$1
if [[ -n "$EUROC_DATASETS_PATH" ]]; then
	echo ...
else
    echo missing datasets base path... using default 
	EUROC_DATASETS_PATH="$HOME/Work/datasets/rgbd_datasets/euroc"
fi

echo EUROC_DATASETS_PATH: $EUROC_DATASETS_PATH
if [ ! -d "$EUROC_DATASETS_PATH" ]; then	
	echo "EUROC_DATASETS_PATH does not exist"
	exit 1
fi


DATASETS=( \
MH01 \
MH02 \
MH03 \
MH04 \
MH05 \
V101 \
V102 \
V103 \
V201 \
V202 \
V203 \
)

for dataset in "${DATASETS[@]}"
do
	echo "============================================================="
	echo "processing $dataset"
	DATASET_PATH="$EUROC_DATASETS_PATH/$dataset/mav0/state_groundtruth_estimate0/"

	if [ -d $DATASET_PATH ] ; then
		cd $DATASET_PATH
		evo_traj euroc data.csv --save_as_tum
		# multiply the timestamps for a scale factor
		#$SCRIPT_DIR/multiply_timestamps.py data.tum data_t2.tum $TIME_SCALE_FACTOR 
	else
		echo "dataset not found"
	fi
done



