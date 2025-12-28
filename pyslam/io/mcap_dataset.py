"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
import cv2
import traceback
from datetime import datetime

from pyslam.utilities.logging import Printer

from .mcap.mcap_reader import McapReader
from .mcap.ros2_decoder import Ros2Decoder
from .mcap.mcap_message_syncer import McapMessageSyncer, McapMessageAsyncSyncer
from .dataset_types import SensorType, DatasetEnvironmentType, DatasetType
from .dataset import Dataset


class McapDataset(Dataset):
    def __init__(
        self,
        path,
        name,
        sensor_type=SensorType.RGBD,
        associations=None,
        start_frame_id=0,
        type=DatasetType.MCAP,
        environment_type=DatasetEnvironmentType.INDOOR,
        fps=30,
        config=None,
    ):
        super().__init__(path, name, sensor_type, fps, associations, start_frame_id, type)
        ros_settings = config.ros_settings
        assert ros_settings is not None
        self.ros_settings = ros_settings

        # assert "bag_path" in ros_settings
        self.mcap_path = os.path.join(path, name)
        if not os.path.exists(self.mcap_path):
            raise ValueError(f"[McapDataset] MCAP path {self.mcap_path} does not exist")

        self.topics_dict = ros_settings["topics"]
        self.topics_list = list(self.topics_dict.values())
        self.convert_color_rgb2bgr = ros_settings.get("convert_color_rgb2bgr", False)

        print(f"[McapDataset]: MCAP: {self.mcap_path}, Topics: {self.topics_dict}")

        self.sync_queue_size = int(self.ros_settings.get("sync_queue_size", 10))
        self.sync_slop = float(self.ros_settings.get("sync_slop", 0.05))
        self.depth_factor = float(self.ros_settings.get("depth_factor", 1.0))

        print(f"[McapDataset] sync_queue_size: {self.sync_queue_size}, sync_slop: {self.sync_slop}")

        self.color_image_topic = self.topics_dict.get("color_image")
        self.depth_image_topic = self.topics_dict.get("depth_image")
        self.right_color_image_topic = self.topics_dict.get("right_color_image")
        self.imu_topic = self.topics_dict.get("imu")

        assert self.color_image_topic is not None, "Color image topic not found"
        if self.sensor_type == SensorType.STEREO:
            assert self.right_color_image_topic is not None, "Right color image topic not found"
        if self.sensor_type == SensorType.RGBD:
            assert self.depth_image_topic is not None, "Depth image topic not found"

        # Initialize MCAP reader
        self.reader = McapReader(self.mcap_path, selected_topics=self.topics_list)
        self.decoder = Ros2Decoder()

        # Initialize message synchronizer
        # syncer_type = McapMessageSyncer
        syncer_type = McapMessageAsyncSyncer
        self.syncer = syncer_type(
            reader=self.reader,
            topics=self.topics_list,
            sync_queue_size=self.sync_queue_size,
            sync_slop=self.sync_slop,
        )

        # Get summary to extract topic information
        summary = self.reader.get_summary()
        print(f"[McapDataset] summary: {summary}")

        # Extract timestamps and counts for each topic
        self.topic_timestamps = {}
        self.topic_counts = {}

        # Collect all messages and their timestamps
        for topic_name in self.topics_list:
            timestamps = []
            for path, m in self.reader.iter_ros2(selected_topics=[topic_name]):
                # MCAP timestamps can be in nanoseconds (int) or datetime object
                timestamp_ns = m.log_time
                if isinstance(timestamp_ns, datetime):
                    # Convert datetime to nanoseconds
                    timestamp_ns = int(timestamp_ns.timestamp() * 1e9)
                # Convert nanoseconds to seconds
                timestamp_s = timestamp_ns / 1e9
                timestamps.append(timestamp_s)
            self.topic_timestamps[topic_name] = sorted(timestamps)
            self.topic_counts[topic_name] = len(timestamps)

        self.environment_type = environment_type
        self.Ts = 1.0 / fps
        self.scale_viewer_3d = 0.1 if sensor_type != SensorType.MONOCULAR else 0.05

        print("Processing MCAP file(s)")
        self.num_frames = self.topic_counts[self.color_image_topic]
        self.max_frame_id = self.num_frames
        print(f"Number of frames: {self.num_frames}")

        num_timestamps = len(self.topic_timestamps[self.color_image_topic])
        print(f"Number of timestamps: {num_timestamps}")
        assert num_timestamps == self.num_frames

        self.cam_stereo_settings = config.cam_stereo_settings
        if self.sensor_type == SensorType.STEREO:
            Printer.yellow("[McapDataset] automatically rectifying the stereo images")
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
        """Read next synchronized set of messages."""
        # Determine required topics based on sensor type
        required_topics = [self.color_image_topic]
        if self.sensor_type == SensorType.STEREO:
            required_topics.append(self.right_color_image_topic)
        if self.sensor_type == SensorType.RGBD:
            required_topics.append(self.depth_image_topic)

        # Get synchronized messages
        # For async syncer, use a small timeout to allow background thread to process
        # For sync syncer, timeout=None means non-blocking (immediate return)
        timeout = 0.1 if isinstance(self.syncer, McapMessageAsyncSyncer) else None
        result = self.syncer.get_next_synced(required_topics=required_topics, timeout=timeout)
        if result is None and not self.syncer.is_eof():
            # Try once more with a longer timeout for async syncer
            if isinstance(self.syncer, McapMessageAsyncSyncer):
                result = self.syncer.get_next_synced(required_topics=required_topics, timeout=1.0)
            else:
                # For sync syncer, try again (non-blocking)
                result = self.syncer.get_next_synced(required_topics=required_topics)

        if result is not None:
            ts, synced = result
            self._timestamp = ts

            if self.color_image_topic in synced:
                try:
                    msg = synced[self.color_image_topic]
                    img = self.decoder.decode_image(msg.ros_msg)
                    # Convert RGB to BGR if needed (OpenCV uses BGR)
                    # ROS2 typically uses rgb8 encoding for color images
                    encoding = msg.ros_msg.encoding.lower()
                    if img.ndim == 3 and img.shape[2] == 3:
                        if encoding in ("rgb8", "rgb") and self.convert_color_rgb2bgr:
                            # Convert RGB to BGR for OpenCV compatibility
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # If encoding is bgr8, no conversion needed
                    self.color_img = img
                    print(f"[read] color image shape: {self.color_img.shape}")
                except Exception as e:
                    print(f"Error reading color image: {e}")
                    print(traceback.format_exc())
                    self.color_img = None

            if self.depth_image_topic in synced:
                try:
                    depth_msg = synced[self.depth_image_topic]
                    # Use Ros2Decoder.decode_image() which handles depth images (32FC1, 16UC1)
                    decoded_depth = self.decoder.decode_image(depth_msg.ros_msg)
                    # Preserve original dtype when depth_factor is 1.0
                    if self.depth_factor == 1.0:
                        self.depth_img = decoded_depth
                    else:
                        # Apply depth_factor (will convert to float32 if original was uint16)
                        self.depth_img = self.depth_factor * decoded_depth
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
                    self.depth_img = None

            if self.right_color_image_topic in synced:
                try:
                    msg = synced[self.right_color_image_topic]
                    img = self.decoder.decode_image(msg.ros_msg)
                    # Convert RGB to BGR if needed
                    encoding = msg.ros_msg.encoding.lower()
                    if img.ndim == 3 and img.shape[2] == 3:
                        if encoding in ("rgb8", "rgb") and self.convert_color_rgb2bgr:
                            # Convert RGB to BGR for OpenCV compatibility
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # If encoding is bgr8, no conversion needed
                    self.right_color_img = img
                    print(f"[read] right color image shape: {self.right_color_img.shape}")
                except Exception as e:
                    print(f"Error reading right color image: {e}")
                    print(traceback.format_exc())
                    self.right_color_img = None

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
