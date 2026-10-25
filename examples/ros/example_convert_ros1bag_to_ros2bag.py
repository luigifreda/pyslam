"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import subprocess
import time
import os
import signal
import argparse

# --- Environment setup ---
ros1_env = "/opt/ros/noetic/setup.bash"
ros2_env = "/opt/ros/foxy/setup.bash"

# Global process handles
bridge_proc = None
ros2_rec_proc = None
ros1_play_proc = None
is_terminated = False

def source_cmds(*sources):
    return ' && '.join([f'source {src}' for src in sources])


def launch_ros1_core():
    print("[INFO] Launching ROS1 core...")
    cmd = f'bash -c "{source_cmds(ros1_env)} && roscore"'
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

def wait_for_ros1_core(timeout=30):
    print("[INFO] Waiting for ROS1 core to start...")
    start_time = time.time()
    while True:
        try:
            subprocess.check_output(
                f'bash -c "source {ros1_env} && rosnode list"',
                shell=True,
                stderr=subprocess.DEVNULL,
                text=True
            )
            print("[INFO] ROS1 core is running.")
            break
        except subprocess.CalledProcessError:
            if time.time() - start_time > timeout:
                print("[ERROR] Timeout waiting for ROS1 core.")
                raise TimeoutError("ROS1 core did not start within timeout.")
            time.sleep(1)

def launch_ros1_bridge(ros1_bridge_package="ros1_bridge", ros1_bridge_executable="dynamic_bridge"):
    print("[INFO] Launching ros1_bridge dynamic_bridge...")
    cmd = f'bash -c "{source_cmds(ros1_env, ros2_env)} && ros2 run {ros1_bridge_package} {ros1_bridge_executable} --bridge-all-topics"'
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

def play_ros1_bag(ros1_bag_path):
    print(f"[INFO] Playing ROS1 bag: {ros1_bag_path}")
    cmd = f'bash -c "{source_cmds(ros1_env)} && rosbag play {ros1_bag_path}"'
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

def record_ros2_bag(ros2_bag_output_file, topics_to_record):
    print(f"[INFO] Recording ROS2 bag to: {ros2_bag_output_file}")
    path = os.path.dirname(ros2_bag_output_file)
    out_file = os.path.basename(ros2_bag_output_file)
    if not os.path.exists(path):
        os.makedirs(path)
    if topics_to_record:
        topics = ' '.join(topics_to_record)
        cmd = f'bash -c "{source_cmds(ros2_env)} && cd {path} && ros2 bag record {topics} -o {out_file}"'
    else:
        cmd = f'bash -c "{source_cmds(ros2_env)} && cd {path} && ros2 bag record -a -o {out_file}"'
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

def terminate_all():
    global bridge_proc, ros2_rec_proc, ros1_play_proc
    print("[INFO] Terminating all subprocesses...")
    for proc in [ros1_play_proc, ros2_rec_proc, bridge_proc]:
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception as e:
                print(f"[WARN] Error killing process: {e}")

def signal_handler(sig, frame):
    global is_terminated
    if not is_terminated:
        print("\n[INFO] Caught Ctrl+C! Cleaning up...")
        is_terminated = True
        terminate_all()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ros1bag", type=str, help="Path to the ROS1 bag file.",
                        default='/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room.bag')
    parser.add_argument("--ros2bag", type=str, help="Path to the ROS2 bag output directory.",
                        default='/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room.ros2.bag')
    parser.add_argument("--topics", nargs='+', default=[], help="List of ROS2 topics to record.")
    args = parser.parse_args()

    ros1_bag_path = args.ros1bag
    ros2_bag_output_file = args.ros2bag
    topics_to_record = args.topics

    try:
        launch_ros1_core()
        wait_for_ros1_core()
        
        bridge_proc = launch_ros1_bridge()
        time.sleep(5)

        ros2_rec_proc = record_ros2_bag(ros2_bag_output_file, topics_to_record)
        time.sleep(3)

        ros1_play_proc = play_ros1_bag(ros1_bag_path)
        ros1_play_proc.wait()

        print("[INFO] rosbag play finished.")

        if ros2_rec_proc:
            os.killpg(os.getpgid(ros2_rec_proc.pid), signal.SIGINT)
            ros2_rec_proc.wait()

    except KeyboardInterrupt:
        signal_handler(None, None)

    finally:
        if not is_terminated:
            terminate_all()
