#!/usr/bin/env python3

import subprocess
import sys
import json
import signal
import os


# current file path 
kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)

kMaxNumRowsPerColumn = 3
kSessionName = "pyslam_session"
kCustomTmuxConfigFile = kScriptFolder + "/tmux.conf"


def run_tmux_in_subshells(commands, tmux_config=None):
    session_name = "pyslam_session"
    n_shells = len(commands)

    # Use a custom tmux configuration if provided
    tmux_base_cmd = ["tmux"]
    if tmux_config:
        tmux_base_cmd += ["-f", tmux_config]

    # Create a new tmux session
    subprocess.run(tmux_base_cmd + ["new-session", "-d", "-s", session_name])

    for i in range(1, n_shells):
        if i % 3 == 0:  # After 3 panes in a column, create a new column
            subprocess.run(tmux_base_cmd + ["split-window", "-t", session_name, "-h"])  # Horizontal split
        else:
            subprocess.run(tmux_base_cmd + ["split-window", "-t", session_name, "-v"])  # Vertical split

        subprocess.run(tmux_base_cmd + ["select-layout", "-t", session_name, "tiled"])  # Arrange panes

    # Enable mouse support dynamically if not using a config file
    if not tmux_config:
        subprocess.run(tmux_base_cmd + ["set-option", "-t", session_name, "mouse", "on"])

    # Send the command to each pane
    for i, command in enumerate(commands):
        pane_id = f"{session_name}.{i}"
        subprocess.run(tmux_base_cmd + ["send-keys", "-t", pane_id, command, "C-m"])

    # Attach to the session
    subprocess.run(tmux_base_cmd + ["attach-session", "-t", session_name])


def cleanup_tmux(session_name):
    """Kill the tmux session."""
    print("\nCleaning up tmux session...")
    subprocess.run(["tmux", "kill-session", "-t", session_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def signal_handler(sig, frame):
    """Handle SIGINT (CTRL+C) and exit."""
    cleanup_tmux(kSessionName)
    sys.exit(0)


#example usage:
#  $ ./tmux_split_json.py '["htop", "ls -la", "watch -n 1 date", "ping google.com"]'
# to kill it press in sequence [CTRL+A] + [CTRL+C]
# or run one of the following commands in another shell:
#  $ tmux kill-server
#  or
#  $ tmux kill-session -t pyslam_session 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <json_commands>")
        sys.exit(1)

    # Register the signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

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

    # Optional custom tmux config file
    tmux_config = kCustomTmuxConfigFile

    run_tmux_in_subshells(commands, tmux_config)