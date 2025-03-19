#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR/.."
ROOT_DIR=$(realpath $ROOT_DIR)
LOGS_DIR="$ROOT_DIR"/logs

echo "ROOT_DIR: $ROOT_DIR"

. "$ROOT_DIR"/pyenv-activate.sh 

if [ ! -f "$LOGS_DIR/local_mapping.log" ]; then 
    touch "$LOGS_DIR/local_mapping.log"
fi 
if [ ! -f "$LOGS_DIR/loop_closing.log" ]; then 
    touch "$LOGS_DIR/loop_closing.log"
fi
if [ ! -f "$LOGS_DIR/loop_detecting.log" ]; then 
    touch "$LOGS_DIR/loop_detecting.log"
fi
if [ ! -f "$LOGS_DIR/gba.log" ]; then 
    touch "$LOGS_DIR/gba.log"
fi
if [ ! -f "$LOGS_DIR/volumetric_integrator.log" ]; then 
    touch "$LOGS_DIR/volumetric_integrator.log"
fi

# launch SLAM and check the parallel logs
COMMAND_STRING='[" . '$ROOT_DIR'/pyenv-activate.sh; '$ROOT_DIR'/main_slam.py", "tail -f '$LOGS_DIR'/local_mapping.log", "tail -f '$LOGS_DIR'/loop_closing.log '$LOGS_DIR'/loop_detecting.log", "tail -f '$LOGS_DIR'/gba.log", "tail -f '$LOGS_DIR'/volumetric_integrator.log"]'
echo COMMAND_STRING: $COMMAND_STRING

#set -x 

$SCRIPT_DIR/tmux_split_json.py "$COMMAND_STRING"


# to kill it 
#  $ tmux kill-server
#  or
#  $ tmux kill-session -t pyslam_session 