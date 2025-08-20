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

import numpy as np
import os
import cv2
from pyslam.utilities.utils_sys import Printer
import traceback

try:
    import ros2_pybindings
    from cv_bridge import CvBridge
except:
    Printer.red("Check ROS2 is installed and sourced, and ros2_pybindings was correctly built")
    print(traceback.format_exc())
    sys.exit(1)

from collections import defaultdict

from .dataset_types import SensorType, DatasetEnvironmentType, DatasetType
from .dataset import Dataset


# class Ros2BagReader:
#     def __init__(self, bag_path, topic_types, sync_slop=0.05):
#         self.bag_path = bag_path
#         self.sync_slop = sync_slop
#         self.topic_types = topic_types
#         self.topics = list(topic_types.keys())
#         self.bridge = CvBridge()

#         self.reader = SequentialReader()
#         storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
#         converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
#         self.reader.open(storage_options, converter_options)

#         self.reader.set_filter({'topics': self.topics})

#         self.topic_msg_type_map = {
#             topic: self._import_msg_class(msg_type)
#             for topic, msg_type in topic_types.items()
#         }

#         self.cached_msgs = defaultdict(list)
#         self.last_stamp = None

#     def _import_msg_class(self, full_type):
#         module_name, msg_name = full_type.split('/')
#         exec(f"from {module_name}.msg import {msg_name}", globals())
#         return eval(msg_name)

#     def read_step(self):
#         if not self.reader.has_next():
#             return None

#         topic, data, t = self.reader.read_next()
#         msg_type = self.topic_msg_type_map[topic]
#         msg = deserialize_message(data, msg_type)

#         try:
#             stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
#         except AttributeError:
#             stamp = t / 1e9

#         self.cached_msgs[topic].append((stamp, msg))

#         # Try to find a set of messages within sync_slop
#         if all(self.cached_msgs.values()):
#             ref_stamp = min([v[0][0] for v in self.cached_msgs.values()])
#             synced = []
#             for topic in self.topics:
#                 candidates = [m for m in self.cached_msgs[topic] if abs(m[0] - ref_stamp) <= self.sync_slop]
#                 if not candidates:
#                     return None  # No match
#                 synced.append(candidates[0])
#                 self.cached_msgs[topic].remove(candidates[0])
#             return ref_stamp, dict(zip(self.topics, [m[1] for m in synced]))

#         return None


def decode_ros2_depth_image(msg):
    assert msg.encoding == "32FC1", f"Unsupported encoding: {msg.encoding}"
    dtype = np.float32
    height = msg.height
    width = msg.width
    step = msg.step  # bytes per row
    expected_step = width * 4  # 4 bytes per float32

    if step != expected_step:
        raise ValueError(f"Unexpected step size: {step} != {expected_step}")

    # Convert list of ints to raw bytes
    data_bytes = bytes(msg.data)

    # Interpret buffer as float32 and reshape
    data = np.frombuffer(data_bytes, dtype=dtype)
    image = data.reshape((height, width))

    return image


class Ros2bagDataset(Dataset):
    def __init__(
        self,
        path,
        name,
        sensor_type=SensorType.RGBD,
        associations=None,
        start_frame_id=0,
        type=DatasetType.ROS1BAG,
        environment_type=DatasetEnvironmentType.INDOOR,
        fps=30,
        config=None,
    ):
        super().__init__(path, name, sensor_type, fps, associations, start_frame_id, type)
        ros_settings = config.ros_settings
        assert ros_settings is not None
        self.ros_settings = ros_settings

        assert "bag_path" in ros_settings
        self.bag_path = ros_settings["bag_path"]
        if not os.path.exists(self.bag_path):
            raise ValueError(f"[Ros2bagDataset] Bag path {self.bag_path} does not exist")

        self.topics_dict = ros_settings["topics"]
        self.topics_list = list(self.topics_dict.values())

        print(f"[Ros2bagDataset]: Bag: {self.bag_path}, Topics: {self.topics_dict}")

        self.sync_queue_size = int(self.ros_settings["sync_queue_size"])
        self.sync_slop = float(self.ros_settings["sync_slop"])
        self.depth_factor = float(self.ros_settings.get("depth_factor", 1.0))

        print(
            f"[Ros2bagDataset] sync_queue_size: {self.sync_queue_size}, sync_slop: {self.sync_slop}"
        )

        self.color_image_topic = self.topics_dict.get("color_image")
        self.depth_image_topic = self.topics_dict.get("depth_image")
        self.right_color_image_topic = self.topics_dict.get("right_color_image")
        self.imu_topic = self.topics_dict.get("imu")

        assert self.color_image_topic is not None, "Color image topic not found"
        if self.sensor_type == SensorType.STEREO:
            assert self.right_color_image_topic is not None, "Right color image topic not found"
        if self.sensor_type == SensorType.RGBD:
            assert self.depth_image_topic is not None, "Depth image topic not found"

        self.reader = ros2_pybindings.Ros2BagSyncReaderATS(
            bag_path=self.bag_path,
            topics=self.topics_list,
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.topic_timestamps = self.reader.topic_timestamps
        self.topic_counts = self.reader.topic_counts
        self.bridge = CvBridge()

        self.environment_type = environment_type
        self.Ts = 1.0 / fps
        self.scale_viewer_3d = 0.1 if sensor_type != SensorType.MONOCULAR else 0.05

        print("Processing ROS2 bag")
        self.num_frames = self.topic_counts[self.color_image_topic]
        self.max_frame_id = self.num_frames
        print(f"Number of frames: {self.num_frames}")

        num_timestamps = len(self.topic_timestamps[self.color_image_topic])
        print(f"Number of timestamps: {num_timestamps}")
        assert num_timestamps == self.num_frames

        self.cam_stereo_settings = config.cam_stereo_settings
        if self.sensor_type == SensorType.STEREO:
            Printer.yellow("[Ros2bagDataset] automatically rectifying the stereo images")
            if self.cam_stereo_settings is None:
                sys.exit("ERROR: Missing stereo settings in YAML config!")
            width = config.cam_settings["Camera.width"]
            height = config.cam_settings["Camera.height"]

            K_l, D_l, R_l, P_l = self.cam_stereo_settings["left"].values()
            K_r, D_r, R_r, P_r = self.cam_stereo_settings["right"].values()

            self.M1l, self.M2l = cv2.initUndistortRectifyMap(
                K_l, D_l, R_l, P_l[:3, :3], (width, height), cv2.CV_32FC1
            )
            self.M1r, self.M2r = cv2.initUndistortRectifyMap(
                K_r, D_r, R_r, P_r[:3, :3], (width, height), cv2.CV_32FC1
            )
        self.debug_rectification = False

        self.count = -1
        self.color_img = None
        self.depth_img = None
        self.right_color_img = None

    def read(self):
        result = None
        while not result and not self.reader.is_eof():
            result = self.reader.read_step()
        if result is not None:
            ts, synced = result
            self._timestamp = ts
            if self.color_image_topic in synced:
                self.color_img = self.bridge.imgmsg_to_cv2(
                    synced[self.color_image_topic], desired_encoding="bgr8"
                )
                print(f"[read] color image shape: {self.color_img.shape}")
            if self.depth_image_topic in synced:
                try:
                    depth_msg = synced[self.depth_image_topic]
                    # print(f"Depth encoding: {depth_msg.encoding}")
                    # print(f"Width: {depth_msg.width}, Height: {depth_msg.height}, Step: {depth_msg.step}, Endianness: {depth_msg.is_bigendian}, Encoding: {depth_msg.encoding}, expected step: {depth_msg.width * 4}")
                    # self.depth_img = self.depth_factor * self.bridge.imgmsg_to_cv2(synced[self.depth_image_topic], desired_encoding="passthrough")
                    # self.depth_img = self.bridge.imgmsg_to_cv2(synced[self.depth_image_topic], desired_encoding="passthrough")
                    self.depth_img = self.depth_factor * decode_ros2_depth_image(depth_msg)
                    if False:
                        valid_depths = self.depth_img[np.isfinite(self.depth_img)]
                        print("Depth range:", np.min(valid_depths), np.max(valid_depths))
                        print(
                            "Depth nans:",
                            np.isnan(self.depth_img).sum(),
                            " zeros:",
                            np.count_nonzero(self.depth_img == 0),
                        )
                    print(
                        f"[read] depth image shape: {self.depth_img.shape}, type: {self.depth_img.dtype}"
                    )
                except Exception as e:
                    print(f"Error reading depth image: {e}")
                    print(traceback.format_exc())
            if self.right_color_image_topic in synced:
                self.right_color_img = self.bridge.imgmsg_to_cv2(
                    synced[self.right_color_image_topic], desired_encoding="bgr8"
                )
                print(f"[read] left color image shape: {self.right_color_img.shape}")
            self.count += 1
            self.is_ok = True
        else:
            self._timestamp = None
            self.color_img = None
            self.depth_img = None
            self.right_color_img = None
            self.is_ok = False

    def getImage(self, frame_id):
        if frame_id < self.max_frame_id:
            while self.count < frame_id and self.is_ok:
                self.read()
            img = self.color_img
            if self.sensor_type == SensorType.STEREO and img is not None:
                img = cv2.remap(img, self.M1l, self.M2l, cv2.INTER_LINEAR)
            self.is_ok = img is not None
            self._next_timestamp = (
                self.topic_timestamps[self.color_image_topic][frame_id + 1]
                if frame_id + 1 < self.max_frame_id
                else self._timestamp + self.Ts
            )
            return img
        self.is_ok = False
        self._timestamp = None
        return None

    def getImageRight(self, frame_id):
        if self.sensor_type == SensorType.MONOCULAR:
            return None
        if frame_id < self.max_frame_id:
            while self.count < frame_id and self.is_ok:
                self.read()
            img = self.right_color_img
            if self.sensor_type == SensorType.STEREO and img is not None:
                img = cv2.remap(img, self.M1r, self.M2r, cv2.INTER_LINEAR)
            self.is_ok = img is not None
            self._next_timestamp = (
                self.topic_timestamps[self.right_color_image_topic][frame_id + 1]
                if frame_id + 1 < self.max_frame_id
                else self._timestamp + self.Ts
            )
            return img
        self.is_ok = False
        self._timestamp = None
        return None

    def getDepth(self, frame_id):
        if self.sensor_type != SensorType.RGBD:
            return None
        frame_id += self.start_frame_id
        if frame_id < self.max_frame_id:
            while self.count < frame_id and self.is_ok:
                self.read()
            img = self.depth_img
            self.is_ok = img is not None
            self._next_timestamp = (
                self.topic_timestamps[self.depth_image_topic][frame_id + 1]
                if frame_id + 1 < self.max_frame_id
                else self._timestamp + self.Ts
            )
            return img
        self.is_ok = False
        self._timestamp = None
        return None
