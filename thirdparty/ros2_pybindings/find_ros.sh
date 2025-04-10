#!/usr/bin/env bash

# Automatically find ROS paths. Will export ROS1_INSTALL_PATH and ROS2_INSTALL_PATH

# Known ROS distros. Here you can just add the distros you have/prefer.
ROS1_DISTROS=("noetic") 
ROS2_DISTROS=("foxy" "galactic" "humble" "iron" "rolling" "eloquent" "dashing" "crystal" "ardent")

ROS1_INSTALL_PATH=""
ROS2_INSTALL_PATH=""

# Helper: checks for first existing distro in a given list
find_first_ros_path() {
    local -n distros=$1
    for distro in "${distros[@]}"; do
        local path="/opt/ros/$distro"
        if [ -f "$path/setup.bash" ]; then
            echo "$path"
            return
        fi
    done
}

ROS1_INSTALL_PATH=$(find_first_ros_path ROS1_DISTROS)
ROS2_INSTALL_PATH=$(find_first_ros_path ROS2_DISTROS)

# Export or fallback
if [ -n "$ROS1_INSTALL_PATH" ]; then
    export ROS1_INSTALL_PATH
    echo "✅ Found ROS 1 at: $ROS1_INSTALL_PATH"
else
    echo "⚠️ No ROS 1 installation found."
fi

if [ -n "$ROS2_INSTALL_PATH" ]; then
    export ROS2_INSTALL_PATH
    echo "✅ Found ROS 2 at: $ROS2_INSTALL_PATH"
else
    echo "⚠️ No ROS 2 installation found."
fi