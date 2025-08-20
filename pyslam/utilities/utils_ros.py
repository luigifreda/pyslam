import os
import sys
import time
import subprocess
import os
from pathlib import Path


# Known ROS distros
ROS1_DISTROS = ["noetic"]
ROS2_DISTROS = [
    "foxy",
    "galactic",
    "humble",
    "iron",
    "rolling",
    "eloquent",
    "dashing",
    "crystal",
    "ardent",
]


def find_first_ros_path(distros):
    """Find the first ROS installation with a setup.bash file."""
    for distro in distros:
        path = Path(f"/opt/ros/{distro}")
        if (path / "setup.bash").is_file():
            return str(path)
    return ""


def setup_ros_env():
    # Discover installations
    ros1_install_path = find_first_ros_path(ROS1_DISTROS)
    ros2_install_path = find_first_ros_path(ROS2_DISTROS)

    # Export or fallback
    if ros1_install_path:
        os.environ["ROS1_INSTALL_PATH"] = ros1_install_path
        print(f"✅ Found ROS 1 at: {ros1_install_path}")
    else:
        print("⚠️ No ROS 1 installation found.")

    if ros2_install_path:
        os.environ["ROS2_INSTALL_PATH"] = ros2_install_path
        print(f"✅ Found ROS 2 at: {ros2_install_path}")
    else:
        print("⚠️ No ROS 2 installation found.")
    return ros1_install_path, ros2_install_path


def source_cmds(*sources):
    return " && ".join([f"source {src}" for src in sources])


def launch_ros1_core(ros1_env):
    print("[INFO] Launching ROS1 core...")
    cmd = f'bash -c "{source_cmds(ros1_env)} && roscore"'
    roscore_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    return roscore_proc


def wait_for_ros1_core(ros1_env, timeout=30):
    print("[INFO] Waiting for ROS1 core to start...")
    start_time = time.time()
    while True:
        try:
            subprocess.check_output(
                f'bash -c "{source_cmds(ros1_env)} && rosnode list"',
                shell=True,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            print("[INFO] ROS1 core is running.")
            break
        except subprocess.CalledProcessError:
            if time.time() - start_time > timeout:
                print("[ERROR] Timeout waiting for ROS1 core.")
                raise TimeoutError("ROS1 core did not start within timeout.")
            time.sleep(1)


def launch_ros1_bridge(
    ros1_bridge_package="ros1_bridge",
    ros1_bridge_executable="dynamic_bridge",
    ros1_env=None,
    ros2_env=None,
    bridge_all=True,
):
    print("[INFO] Launching ros1_bridge dynamic_bridge...")
    bridge_flag = "--bridge-all-topics" if bridge_all else ""
    cmd = f'bash -c "{source_cmds(ros1_env, ros2_env)} && ros2 run {ros1_bridge_package} {ros1_bridge_executable} {bridge_flag}"'
    ros1_bridge_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


def play_ros2_bag(ros2_bag_path, ros2_env):
    print(f"[INFO] Playing ROS2 bag: {ros2_bag_path}")
    cmd = f'bash -c "{source_cmds(ros2_env)} && ros2 bag play {ros2_bag_path}"'
    ros2_bag_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
