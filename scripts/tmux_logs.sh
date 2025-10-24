#!/usr/bin/env bash

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR_/.."
ROOT_DIR=$(realpath $ROOT_DIR)
LOGS_DIR="$ROOT_DIR"/logs
SCRIPTS_DIR="$ROOT_DIR"/scripts

echo ROOT_DIR: $ROOT_DIR

. "$ROOT_DIR"/pyenv-activate.sh 

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Define required log files
LOG_FILES=(
    local_mapping.log
    loop_closing.log
    loop_detecting.log
    gba.log
    volumetric_integrator.log
    semantic_mapping.log
)

# Create missing log files
for logfile in "${LOG_FILES[@]}"; do
    LOG_PATH="$LOGS_DIR/$logfile"
    if [ ! -f "$LOG_PATH" ]; then
        touch "$LOG_PATH"
    fi
done

# Check the parallel SLAM logs 
# Construct JSON-style command string
COMMAND_STRING=$(cat <<EOF
[
  "tail -f $LOGS_DIR/local_mapping.log",
  "tail -f $LOGS_DIR/loop_closing.log $LOGS_DIR/loop_detecting.log",
  "tail -f $LOGS_DIR/gba.log",
  "tail -f $LOGS_DIR/semantic_mapping.log",
  "tail -f $LOGS_DIR/volumetric_integrator.log"
]
EOF
)

echo "COMMAND_STRING: $COMMAND_STRING"

#set -x 

$SCRIPTS_DIR/tmux_split_json.py "$COMMAND_STRING"


# to kill it 
#  $ tmux kill-server
#  or
#  $ tmux kill-session -t pyslam_session 