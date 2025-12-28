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

import os
import sys
import time
import cv2
import numpy as np
import subprocess
import os
import signal
import psutil  # Add this import at the top of your script if not already there

from pyslam.utilities.logging import Printer
import traceback

try:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, Imu
    from message_filters import Subscriber, ApproximateTimeSynchronizer
except ImportError:
    Printer.red("ROS1 not installed or setup.bash not sourced!")
    print(traceback.format_exc())
    sys.exit(1)


from .dataset_types import SensorType, DatasetEnvironmentType, DatasetType
from .dataset import Dataset


ros1_env = "/opt/ros/noetic/setup.bash"
ros2_env = "/opt/ros/foxy/setup.bash"


def kill_processes_matching_cmd(keywords, time_sleep=2.0):
    """
    Kill processes whose command line contains all the specified keywords (substring match).

    Args:
        keywords (list of str): List of strings that must all appear in the cmdline string.
    """
    print(f"[INFO] Searching for processes matching: {keywords}")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue
            cmd_str = " ".join(cmdline)
            if all(word in cmd_str for word in keywords):
                Printer.yellow(f"[WARN] Killing process {proc.pid} matching: {cmd_str}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(time_sleep)


class Ros2BagPlayerManager:
    def __init__(self, ros1_env, ros2_env, ros2_bag_path, rate_playback, bridge_all=True):
        self.ros1_env = ros1_env
        self.ros2_env = ros2_env
        self.ros2_bag_path = ros2_bag_path
        self.rate_playback = rate_playback
        self.bridge_all = bridge_all

        self.roscore_proc = None
        self.ros1_bridge_proc = None
        self.ros2_bag_proc = None

    def source_cmds(self, *sources):
        return " && ".join([f"source {src}" for src in sources])

    def launch_ros1_core(self):
        print("[INFO] Launching ROS1 core...")

        # Kill any previously ros master processes
        kill_processes_matching_cmd(["rosmaster"])

        cmd = f'bash -c "{self.source_cmds(self.ros1_env)} && roscore"'
        self.roscore_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        Printer.orange(f"running command {cmd}")

    def wait_for_ros1_core(self, timeout=30):
        print("[INFO] Waiting for ROS1 core to start...")
        start_time = time.time()
        while True:
            try:
                subprocess.check_output(
                    f'bash -c "{self.source_cmds(self.ros1_env)} && rosnode list"',
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
        self, ros1_bridge_package="ros1_bridge", ros1_bridge_executable="dynamic_bridge"
    ):
        print("[INFO] Launching ros1_bridge dynamic_bridge...")

        # Kill any previously running ros2 bag play processes
        kill_processes_matching_cmd(["ros2", "run", ros1_bridge_package])

        bridge_flag = "--bridge-all-topics" if self.bridge_all else ""
        cmd = f'bash -c "{self.source_cmds(self.ros1_env, self.ros2_env)} && ros2 run {ros1_bridge_package} {ros1_bridge_executable} {bridge_flag}"'
        self.ros1_bridge_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        Printer.orange(f"running command {cmd}")

    def play_ros2_bag(self):
        print(f"[INFO] Playing ROS2 bag: {self.ros2_bag_path}")

        # Kill any previously running ros2 bag play processes
        kill_processes_matching_cmd(["ros2", "bag", "play"])

        rate_flag = f"--rate {self.rate_playback}" if self.rate_playback is not None else ""
        cmd = f'bash -c "{self.source_cmds(self.ros2_env)} && ros2 bag play {self.ros2_bag_path} {rate_flag}"'
        self.ros2_bag_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        Printer.orange(f"running command {cmd}")

    def launch_all(self):
        self.launch_ros1_core()
        self.wait_for_ros1_core()
        self.launch_ros1_bridge()
        time.sleep(5)  # Let bridge spin up
        self.play_ros2_bag()
        time.sleep(5)

    def terminate_all(self):
        print("[INFO] Cleaning up all subprocesses...")
        for proc in [self.ros2_bag_proc, self.ros1_bridge_proc, self.roscore_proc]:
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[WARN] Failed to kill process: {e}")

    def __enter__(self):
        self.launch_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_all()


class Ros2bagDataset(Dataset):
    def __init__(
        self,
        path,
        name,
        sensor_type=SensorType.RGBD,
        associations=None,
        start_frame_id=0,
        type=DatasetType.ROS2BAG,
        environment_type=DatasetEnvironmentType.INDOOR,
        fps=30,
        rate_playback=1.0,
        config=None,
    ):

        super().__init__(path, name, sensor_type, fps, associations, start_frame_id, type)
        ros_settings = config.ros_settings
        assert ros_settings is not None
        self.ros_settings = ros_settings

        self.color_image_topic = ros_settings["topics"].get("color_image")
        self.depth_image_topic = ros_settings["topics"].get("depth_image")
        self.right_color_image_topic = ros_settings["topics"].get("right_color_image")
        self.imu_topic = ros_settings["topics"].get("imu")
        self.depth_factor = float(ros_settings.get("depth_factor", 1.0))

        ros2_bag_path = ros_settings["bag_path"]

        self._manager = Ros2BagPlayerManager(
            ros1_env=ros1_env,
            ros2_env=ros2_env,
            ros2_bag_path=ros2_bag_path,
            rate_playback=rate_playback,
        )

        self._manager.launch_all()
        time.sleep(3)

        if not rospy.core.is_initialized():
            rospy.init_node("ros2bag_dataset_player", anonymous=True, disable_signals=True)

        self.bridge = CvBridge()
        self.color_img = None
        self.depth_img = None
        self.right_color_img = None
        self.imu_msg = None

        self._timestamp = None
        self._next_timestamp = None
        self.count = 0
        self.is_ok = True
        self.fps = fps
        self.Ts = 1.0 / fps
        self.rate_playback = rate_playback
        self.environment_type = environment_type
        self.scale_viewer_3d = 0.1 if sensor_type != SensorType.MONOCULAR else 0.05

        queue_size = int(ros_settings.get("queue_size", 100))
        sync_slop = float(ros_settings.get("sync_slop", 0.05))
        self.subscribe(queue_size=queue_size, sync_slop=sync_slop)

        self.cam_stereo_settings = config.cam_stereo_settings
        self.debug_rectification = False

        if self.sensor_type == SensorType.STEREO:
            Printer.yellow("[Ros2bagDataset] Automatically rectifying stereo images")
            if not self.cam_stereo_settings:
                sys.exit("Missing stereo camera settings!")

            width = config.cam_settings["Camera.width"]
            height = config.cam_settings["Camera.height"]

            K_l = self.cam_stereo_settings["left"]["K"]
            D_l = self.cam_stereo_settings["left"]["D"]
            R_l = self.cam_stereo_settings["left"]["R"]
            P_l = self.cam_stereo_settings["left"]["P"]

            K_r = self.cam_stereo_settings["right"]["K"]
            D_r = self.cam_stereo_settings["right"]["D"]
            R_r = self.cam_stereo_settings["right"]["R"]
            P_r = self.cam_stereo_settings["right"]["P"]

            self.M1l, self.M2l = cv2.initUndistortRectifyMap(
                K_l, D_l, R_l, P_l[:3, :3], (width, height), cv2.CV_32FC1
            )
            self.M1r, self.M2r = cv2.initUndistortRectifyMap(
                K_r, D_r, R_r, P_r[:3, :3], (width, height), cv2.CV_32FC1
            )

    def subscribe(self, queue_size=100, sync_slop=0.05):
        self.subs = []

        if (
            self.sensor_type == SensorType.STEREO
            and self.color_image_topic
            and self.right_color_image_topic
        ):
            sub_left = Subscriber(self.color_image_topic, Image)
            sub_right = Subscriber(self.right_color_image_topic, Image)
            ats = ApproximateTimeSynchronizer(
                [sub_left, sub_right], queue_size=queue_size, slop=sync_slop
            )
            ats.registerCallback(self._stereo_cb)
            self.subs.extend([sub_left, sub_right])
            self.sync = ats

        elif (
            self.sensor_type == SensorType.RGBD
            and self.color_image_topic
            and self.depth_image_topic
        ):
            sub_color = Subscriber(self.color_image_topic, Image)
            sub_depth = Subscriber(self.depth_image_topic, Image)
            ats = ApproximateTimeSynchronizer(
                [sub_color, sub_depth], queue_size=queue_size, slop=sync_slop
            )
            ats.registerCallback(self._rgbd_cb)
            self.subs.extend([sub_color, sub_depth])
            self.sync = ats

        if self.imu_topic:
            self.subs.append(
                rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=queue_size)
            )

    def _stereo_cb(self, msg_left, msg_right):
        self.color_img = self.bridge.imgmsg_to_cv2(msg_left, desired_encoding="bgr8")
        self.right_color_img = self.bridge.imgmsg_to_cv2(msg_right, desired_encoding="bgr8")
        self._timestamp = msg_left.header.stamp.to_sec()

    def _rgbd_cb(self, msg_color, msg_depth):
        self.color_img = self.bridge.imgmsg_to_cv2(msg_color, desired_encoding="bgr8")
        self.depth_img = self.depth_factor * self.bridge.imgmsg_to_cv2(
            msg_depth, desired_encoding="passthrough"
        )
        self._timestamp = msg_color.header.stamp.to_sec()

    def _imu_cb(self, msg):
        self.imu_msg = msg

    def read(self):
        rate = rospy.Rate(self.fps)
        retries = 30
        while self.color_img is None and retries > 0:
            rate.sleep()
            retries -= 1

        self.count += 1
        self.is_ok = self.color_img is not None
        return self.is_ok

    def getImage(self, frame_id):
        if not self.read():
            return None
        img = self.color_img
        if self.sensor_type == SensorType.STEREO:
            if self.debug_rectification:
                imgs = img
            img = cv2.remap(img, self.M1l, self.M2l, cv2.INTER_LINEAR)
            if self.debug_rectification:
                imgs = np.concatenate((imgs, img), axis=1)
                cv2.imshow("left raw and rectified images", imgs)
                cv2.waitKey(1)
        return img

    def getImageRight(self, frame_id):
        if self.sensor_type != SensorType.STEREO:
            return None
        if not self.read():
            return None
        img = self.right_color_img
        if self.debug_rectification:
            imgs = img
        img = cv2.remap(img, self.M1r, self.M2r, cv2.INTER_LINEAR)
        if self.debug_rectification:
            imgs = np.concatenate((imgs, img), axis=1)
            cv2.imshow("right raw and rectified images", imgs)
            cv2.waitKey(1)
        return img

    def getDepth(self, frame_id):
        if self.sensor_type != SensorType.RGBD:
            return None
        if not self.read():
            return None
        return self.depth_img

    def shutdown(self):
        print("[INFO] Shutting down Ros2bagDataset")
        self._manager.terminate_all()

    def __del__(self):
        self.shutdown()
