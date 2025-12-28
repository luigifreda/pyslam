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
from pyslam.utilities.logging import Printer
import traceback

try:
    import rosbag
    import rospy
    from cv_bridge import CvBridge

    if not rospy.core.is_initialized():
        rospy.Time.now = lambda: rospy.Time.from_sec(0)
    from message_filters import ApproximateTimeSynchronizer, SimpleFilter
except:
    Printer.red("ROS1 not installed or setup.bash not sourced!")
    Printer.red("Please, double check you have source your ROS1 environment")
    print(traceback.format_exc())
    sys.exit(1)

from collections import defaultdict

from .dataset_types import SensorType, DatasetEnvironmentType, DatasetType
from .dataset import Dataset


class Ros1bagSyncReaderATS:
    def __init__(self, bag_path, topics, queue_size=100, slop=0.05):
        # slop: delay (in seconds) with which messages can be synchronized
        self.bag_path = bag_path
        self.topics = topics
        self.queue_size = queue_size
        self.slop = slop
        self.filters = {topic: SimpleFilter() for topic in topics}
        self._sync = ApproximateTimeSynchronizer(
            list(self.filters.values()), queue_size=self.queue_size, slop=self.slop
        )
        self._synced_msgs = []
        self._bag = rosbag.Bag(self.bag_path, "r")
        self._bag_iter = self._bag.read_messages(topics=self.topics)

        def cb(*msgs):
            self._synced_msgs.append(msgs)

        self._sync.registerCallback(cb)

        self.topic_counts = None
        self.topic_timestamps = None
        self.get_topic_timestamps_and_counts()

    def reset(self):
        self._synced_msgs = []
        self._bag = rosbag.Bag(self.bag_path, "r")
        self._bag_iter = self._bag.read_messages(topics=self.topics)

    def get_topic_timestamps_and_counts(self):
        # Initialize structures
        timestamps = {topic: [] for topic in self.topics}
        counts = {topic: 0 for topic in self.topics}

        with rosbag.Bag(self.bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=self.topics):
                try:
                    stamp = msg.header.stamp.to_sec()
                except AttributeError:
                    stamp = t.to_sec()
                timestamps[topic].append(stamp)
                counts[topic] += 1

        self.topic_timestamps = timestamps
        self.topic_counts = counts

    # to be used in a loop
    # for ts, synced in reader.read():
    #     ...
    def read(self):
        for topic, msg, t in self._bag.read_messages(topics=self.topics):
            self.filters[topic].signalMessage(msg)

        # Now that everything has been fed in, yield synced results
        for msg_tuple in self._synced_msgs:
            # Timestamps are close, take the first one as reference
            stamp = msg_tuple[0].header.stamp.to_sec()
            yield stamp, dict(zip(self.topics, msg_tuple))

    # single step read
    def read_step(self):
        for topic, msg, t in self._bag_iter:
            self.filters[topic].signalMessage(msg)
            if self._synced_msgs:
                msg_tuple = self._synced_msgs.pop(0)
                stamp = msg_tuple[0].header.stamp.to_sec()
                return stamp, dict(zip(self.topics, msg_tuple))
        return None  # No more data

    def read_all_messages_of_topic(self, topic, with_timestamps=False):
        if topic not in self.topics:
            raise ValueError(f"Topic '{topic}' is not in the initialized topic list.")

        messages = []
        with rosbag.Bag(self.bag_path, "r") as bag:
            for _, msg, t in bag.read_messages(topics=[topic]):
                if with_timestamps:
                    try:
                        stamp = msg.header.stamp.to_sec()
                    except AttributeError:
                        stamp = t.to_sec()
                    messages.append((stamp, msg))
                else:
                    messages.append(msg)
        return messages


class Ros1bagDataset(Dataset):
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
        super().__init__(path, name, sensor_type, 30, associations, start_frame_id, type)
        ros_settings = config.ros_settings
        assert ros_settings is not None
        self.ros_settings = ros_settings

        assert "bag_path" in ros_settings
        self.bag_path = ros_settings["bag_path"]
        if not os.path.exists(self.bag_path):
            raise ValueError(f"[Ros1bagDataset] Bag path {self.bag_path} does not exist")
        self.topics_dict = ros_settings["topics"]
        self.topics_list = list(self.topics_dict.values())
        self.convert_color_rgb2bgr = ros_settings.get("convert_color_rgb2bgr", False)

        print(f"[Ros1bagDataset]: Bag: {self.bag_path}, Topics: {self.topics_dict}")

        self.sync_queue_size = int(self.ros_settings["sync_queue_size"])
        self.sync_slop = float(self.ros_settings["sync_slop"])
        self.depth_factor = (
            float(self.ros_settings["depth_factor"]) if "depth_factor" in self.ros_settings else 1.0
        )

        print(
            f"[Ros1bagDataset] sync_queue_size: {self.sync_queue_size}, sync_slop: {self.sync_slop}"
        )

        self.color_image_topic = (
            self.topics_dict["color_image"] if "color_image" in self.topics_dict else None
        )
        self.depth_image_topic = (
            self.topics_dict["depth_image"] if "depth_image" in self.topics_dict else None
        )
        self.right_color_image_topic = (
            self.topics_dict["right_color_image"]
            if "right_color_image" in self.topics_dict
            else None
        )
        self.imu_topic = self.topics_dict["imu"] if "imu" in self.topics_dict else None

        assert self.color_image_topic is not None, "Color image topic not found"
        if self.sensor_type == SensorType.STEREO:
            assert self.right_color_image_topic is not None, "Right color image topic not found"
        if self.sensor_type == SensorType.RGBD:
            assert self.depth_image_topic is not None, "Depth image topic not found"

        self.reader = Ros1bagSyncReaderATS(
            bag_path=self.bag_path,
            topics=self.topics_list,
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.topic_timestamps = self.reader.topic_timestamps
        self.bridge = CvBridge()

        self.environment_type = environment_type

        self.fps = fps
        self.Ts = 1.0 / fps

        self.scale_viewer_3d = 0.1
        if sensor_type == SensorType.MONOCULAR:
            self.scale_viewer_3d = 0.05
        print("Processing ROS1 bag")
        self.num_frames = self.reader.topic_counts[self.topics_dict["color_image"]]
        self.max_frame_id = self.num_frames
        print(f"Number of frames: {self.num_frames}")

        num_timestamps = len(self.topic_timestamps[self.topics_dict["color_image"]])
        print(f"Number of timestamps: {num_timestamps}")
        assert num_timestamps == self.num_frames

        # in case of stereo mode, we rectify the stereo images
        self.cam_stereo_settings = config.cam_stereo_settings
        if self.sensor_type == SensorType.STEREO:
            Printer.yellow("[Ros1bagDataset] automatically rectifying the stereo images")
            if self.cam_stereo_settings is None:
                sys.exit("ERROR: we are missing stereo settings in Euroc YAML settings!")
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
                K_l, D_l, R_l, P_l[0:3, 0:3], (width, height), cv2.CV_32FC1
            )
            self.M1r, self.M2r = cv2.initUndistortRectifyMap(
                K_r, D_r, R_r, P_r[0:3, 0:3], (width, height), cv2.CV_32FC1
            )
        self.debug_rectification = False  # DEBUGGING

        self.count = -1
        self.color_img = None
        self.depth_img = None
        self.right_color_img = None

    def read(self):
        result = self.reader.read_step()
        if result is not None:
            ts, synced = result
            self._timestamp = ts
            if self.color_image_topic:
                color_img_msg = synced[self.color_image_topic]
                self.color_img = self.bridge.imgmsg_to_cv2(color_img_msg, desired_encoding="bgr8")
                if self.convert_color_rgb2bgr:
                    self.color_img = cv2.cvtColor(self.color_img, cv2.COLOR_RGB2BGR)
                print(f"[read] color image shape: {self.color_img.shape}")
            if self.depth_image_topic:
                # depth_msg = synced[self.depth_image_topic]
                # print(f"Depth encoding: {depth_msg.encoding}")
                # print(f"Width: {depth_msg.width}, Height: {depth_msg.height}, Step: {depth_msg.step}, Endianness: {depth_msg.is_bigendian}, Encoding: {depth_msg.encoding}, expected step: {depth_msg.width * 4}")
                depth_msg = synced[self.depth_image_topic]
                self.depth_img = self.depth_factor * self.bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding="passthrough"
                )
                # self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
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
            if self.right_color_image_topic:
                left_color_img_msg = synced[self.right_color_image_topic]
                self.right_color_img = self.bridge.imgmsg_to_cv2(
                    left_color_img_msg, desired_encoding="bgr8"
                )
                if self.convert_color_rgb2bgr:
                    self.right_color_img = cv2.cvtColor(self.right_color_img, cv2.COLOR_RGB2BGR)
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
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            while self.count < frame_id and self.is_ok:
                print(f"[getImage] frame_id: {frame_id}, count: {self.count}")
                self.read()
            img = self.color_img
            if self.sensor_type == SensorType.STEREO:
                # rectify image
                if self.debug_rectification:
                    imgs = img
                img = cv2.remap(img, self.M1l, self.M2l, cv2.INTER_LINEAR)
                if self.debug_rectification:
                    imgs = np.concatenate((imgs, img), axis=1)
                    cv2.imshow("left raw and rectified images", imgs)
                    cv2.waitKey(1)
            self.is_ok = img is not None
            color_timestamps_len = len(self.topic_timestamps[self.color_image_topic])
            if frame_id + 1 < color_timestamps_len:
                self._next_timestamp = self.topic_timestamps[self.color_image_topic][frame_id + 1]
            elif frame_id < color_timestamps_len:
                # Use current frame timestamp + Ts as fallback
                current_ts = self.topic_timestamps[self.color_image_topic][frame_id]
                self._next_timestamp = current_ts + self.Ts
            elif self._timestamp is not None:
                self._next_timestamp = self._timestamp + self.Ts
            else:
                # Last resort: use last timestamp if available
                self._next_timestamp = (
                    self.topic_timestamps[self.color_image_topic][-1] + self.Ts
                    if color_timestamps_len > 0
                    else None
                )
        else:
            self.is_ok = False
            self._timestamp = None
        return img

    def getImageRight(self, frame_id):
        if self.sensor_type == SensorType.MONOCULAR:
            return None  # force a monocular camera if required (to get a monocular tracking even if depth is available)
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            while self.count < frame_id and self.is_ok:
                print(f"[getImageRight] frame_id: {frame_id}, count: {self.count}")
                self.read()
            img = self.right_color_img
            if self.sensor_type == SensorType.STEREO:
                # rectify image
                if self.debug_rectification:
                    imgs = img
                img = cv2.remap(img, self.M1r, self.M2r, cv2.INTER_LINEAR)
                if self.debug_rectification:
                    imgs = np.concatenate((imgs, img), axis=1)
                    cv2.imshow("right raw and rectified images", imgs)
                    cv2.waitKey(1)
            self.is_ok = img is not None
            right_timestamps_len = len(self.topic_timestamps[self.right_color_image_topic])
            if frame_id + 1 < right_timestamps_len:
                self._next_timestamp = self.topic_timestamps[self.right_color_image_topic][
                    frame_id + 1
                ]
            elif frame_id < right_timestamps_len:
                # Use current frame timestamp + Ts as fallback
                current_ts = self.topic_timestamps[self.right_color_image_topic][frame_id]
                self._next_timestamp = current_ts + self.Ts
            elif self._timestamp is not None:
                self._next_timestamp = self._timestamp + self.Ts
            else:
                # Last resort: use last timestamp if available
                self._next_timestamp = (
                    self.topic_timestamps[self.right_color_image_topic][-1] + self.Ts
                    if right_timestamps_len > 0
                    else None
                )
        else:
            self.is_ok = False
            self._timestamp = None
        return img

    def getDepth(self, frame_id):
        if self.sensor_type != SensorType.RGBD:
            return None  # force a monocular camera if required (to get a monocular tracking even if depth is available)
        frame_id += self.start_frame_id
        img = None
        if frame_id < self.max_frame_id:
            while self.count < frame_id and self.is_ok:
                print(f"[getDepth] frame_id: {frame_id}, count: {self.count}")
                self.read()
            img = self.depth_img
            self.is_ok = img is not None
            depth_timestamps_len = len(self.topic_timestamps[self.depth_image_topic])
            if frame_id + 1 < depth_timestamps_len:
                self._next_timestamp = self.topic_timestamps[self.depth_image_topic][frame_id + 1]
            elif frame_id < depth_timestamps_len:
                # Use current frame timestamp + Ts as fallback
                current_ts = self.topic_timestamps[self.depth_image_topic][frame_id]
                self._next_timestamp = current_ts + self.Ts
            elif self._timestamp is not None:
                self._next_timestamp = self._timestamp + self.Ts
            else:
                # Last resort: use last timestamp if available
                self._next_timestamp = (
                    self.topic_timestamps[self.depth_image_topic][-1] + self.Ts
                    if depth_timestamps_len > 0
                    else None
                )
        else:
            self.is_ok = False
            self._timestamp = None
        return img
