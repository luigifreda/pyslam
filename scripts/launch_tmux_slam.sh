#!/usr/bin/env bash

# NOTE: This script is expected to be used under mac. See the file `MAC.md`.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR/.."
ROOT_DIR=$(realpath $ROOT_DIR)
LOGS_DIR="$ROOT_DIR"/logs

. "$ROOT_DIR"/pyenv-activate.sh 

if ! -f "$LOGS_DIR/local_mapping.log"; then 
    touch "$LOGS_DIR/local_mapping.log"
fi 
if ! -f "$LOGS_DIR/loop_closing.log"; then 
    touch "$LOGS_DIR/loop_closing.log"
fi
if ! -f "$LOGS_DIR/loop_detecting.log"; then 
    touch "$LOGS_DIR/loop_detecting.log"
fi
if ! -f "$LOGS_DIR/gba.log"; then 
    touch "$LOGS_DIR/gba.log"
fi

COMMAND_STRING='["'$ROOT_DIR'/main_slam.py", "tail -f '$LOGS_DIR'/local_mapping.log", "tail -f '$LOGS_DIR'/loop_closing.log '$LOGS_DIR'/loop_detecting.log", "tail -f '$LOGS_DIR'/gba.log", "tail -f '$LOGS_DIR'/volumetric_integrator.log"]'
echo COMMAND_STRING: $COMMAND_STRING

#set -x 

$SCRIPT_DIR/tmux_split_json.py "$COMMAND_STRING"


# to kill it 
#  $ tmux kill-server
#  or
#  $ tmux kill-session -t pyslam_session 