#!/usr/bin/env python3

import subprocess
import sys
import json


def run_tmux_in_subshells(commands):
    n_shells = len(commands)

    # Create a new tmux session
    session_name = "pyslam_session"
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

    # Create the required number of sub-shells as rows (vertical split)
    for i in range(1, n_shells):
        subprocess.run(["tmux", "split-window", "-t", session_name, "-v"])

    # Ensure all panes are arranged in rows
    subprocess.run(["tmux", "select-layout", "-t", session_name, "even-vertical"])

    # Send the command to each pane (sub-shell)
    for i, command in enumerate(commands):
        pane_id = f"{session_name}.{i}"
        subprocess.run(["tmux", "send-keys", "-t", pane_id, command, "C-m"])

    # Attach to the session
    subprocess.run(["tmux", "attach-session", "-t", session_name])



#example usage:
#  $ ./tmux_split_json.py '["htop", "ls -la", "watch -n 1 date", "ping google.com"]'
# to kill it 
#  $ tmux kill-server
#  or
#  $ tmux kill-session -t pyslam_session 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <json_commands>")
        sys.exit(1)

    # Read commands from the input JSON variable
    try:
        commands = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print("Invalid JSON input")
        sys.exit(1)

    # Check that commands is a list
    if not isinstance(commands, list):
        print("The JSON input should be a list of commands")
        sys.exit(1)

    run_tmux_in_subshells(commands)