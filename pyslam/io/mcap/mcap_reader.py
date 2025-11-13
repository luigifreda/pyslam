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
import glob
import re
import argparse

import numpy as np

import cv2

from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
from mcap_ros1.reader import read_ros1_messages


from pyslam.io.mcap.ros2_decoder import Ros2Decoder
from pyslam.io.mcap.mcap_summary import McapSummary, TopicInfo, extract_mcap_summary


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def natural_sort_key(s):
    """Sort string using human order (e.g., file2 < file10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# -------------------------------------------------------------
# Reader
# -------------------------------------------------------------
class McapReader:
    """
    Given a path (file or directory), automatically detects a sequence
    of MCAP files, sorts them, and iterates through their content
    sequentially.
    """

    def __init__(self, path, selected_topics=None, detect_sequence=True):
        if detect_sequence:
            self.files = self._detect_mcap_sequence(path)
        else:
            self.files = [path]
        self.selected_topics = selected_topics
        self.ros2_decoder = Ros2Decoder()
        print("Found MCAP files:", self.files)

    def _detect_mcap_sequence(self, path):
        """Detect a sequence of MCAP files in a directory or a single file."""
        if os.path.isdir(path):
            # Get all .mcap files inside the directory
            files = glob.glob(os.path.join(path, "*.mcap"))
        else:
            # Path is a file – find similar files in the same directory
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            root, _ = os.path.splitext(basename)

            # Match files with similar prefix, e.g.:
            #   log.mcap, log.1.mcap, log.2.mcap
            pattern1 = glob.glob(os.path.join(dirname, f"{root}*.mcap"))

            # Extract base name before trailing digits for sequence detection
            # This handles files like: file_0.mcap, file_1.mcap, file_2.mcap
            match = re.match(r"^(.*?)(\d+)$", root)
            if match:
                base_name = match.group(1)
                pattern2_path = os.path.join(dirname, f"{base_name}*.mcap")
                pattern2 = glob.glob(pattern2_path)
            else:
                # If no trailing digits, try splitting by any digit sequence
                prefix = re.split(r"\d+", root)[0]
                pattern2_path = os.path.join(dirname, f"{prefix}*.mcap")
                pattern2 = glob.glob(pattern2_path)

            files = set(pattern1 + pattern2)

        files = sorted(files, key=natural_sort_key)

        if not files:
            raise FileNotFoundError(f"No MCAP files found starting from: {path}")

        return list(files)

    def get_summary(self) -> McapSummary:
        """
        Get summary information about the MCAP files.

        Returns:
            McapSummary object containing summary information
        """
        return extract_mcap_summary(self.files)

    # ---------------------------------------------------------
    # RAW MCAP
    # ---------------------------------------------------------
    def __iter__(self):
        """
        Iterate over raw MCAP messages across all MCAP files.

        Yields:
            (path, schema, channel, message)
        """
        for path in self.files:
            with open(path, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages():
                    yield path, schema, channel, message

    # ---------------------------------------------------------
    # ROS2-decoded messages
    # ---------------------------------------------------------
    def iter_ros2(
        self,
        start_time=None,
        end_time=None,
        log_time_order=True,
        reverse=False,
        selected_topics=None,
    ):
        """
        Iterate over decoded ROS2 messages across all MCAP files.

        Yields:
            (path, mcap_ros2_msg)
        """
        if selected_topics is None:
            selected_topics = self.selected_topics
        for path in self.files:
            for m in read_ros2_messages(
                path,
                topics=selected_topics,
                start_time=start_time,
                end_time=end_time,
                log_time_order=log_time_order,
                reverse=reverse,
            ):
                yield path, m

    # ---------------------------------------------------------
    # ROS1-decoded messages
    # ---------------------------------------------------------
    def iter_ros1(
        self,
        start_time=None,
        end_time=None,
        log_time_order=True,
        reverse=False,
        selected_topics=None,
    ):
        """
        Iterate over decoded ROS1 messages across all MCAP files.

        Yields:
            (path, mcap_ros1_msg)
        """
        if selected_topics is None:
            selected_topics = self.selected_topics
        for path in self.files:
            for m in read_ros1_messages(
                path,
                topics=selected_topics,
                start_time=start_time,
                end_time=end_time,
                log_time_order=log_time_order,
                reverse=reverse,
            ):
                yield path, m

    # ---------------------------------------------------------
    # ROS2-decoded + NumPy helpers for common message types
    # ---------------------------------------------------------
    def iter_ros2_decoded(self, start_time=None, end_time=None, log_time_order=True, reverse=False):
        """
        Iterate over ROS2 messages and attach decoded/NumPy forms
        for common message types:

          * tf2_msgs/TFMessage        -> 'tf'      : list[dict]
          * sensor_msgs/PointCloud2   -> 'points'  : (N,3) float32 array
          * sensor_msgs/Image         -> 'image'   : np.ndarray
          * sensor_msgs/CompressedImage -> 'image' : np.ndarray (if cv2)
          * sensor_msgs/LaserScan     -> 'scan'    : dict

        Yields:
            (path, mcap_ros2_msg, decoded_dict)
        """
        for path, m in self.iter_ros2(
            start_time=start_time,
            end_time=end_time,
            log_time_order=log_time_order,
            reverse=reverse,
        ):
            schema_name = getattr(m.schema, "name", "")
            decoded = {}

            # TF
            if "TFMessage" in schema_name:
                decoded["tf"] = self.ros2_decoder.decode_tf_message(m.ros_msg)

            # PointCloud2
            elif "PointCloud2" in schema_name:
                decoded["points"] = self.ros2_decoder.decode_pointcloud2(m.ros_msg)

            # Image
            elif "sensor_msgs/msg/Image" in schema_name or schema_name.endswith("/Image"):
                decoded["image"] = self.ros2_decoder.decode_image(m.ros_msg)

            # CompressedImage
            elif "CompressedImage" in schema_name:
                decoded["image"] = self.ros2_decoder.decode_compressed_image(m.ros_msg)

            # LaserScan
            elif "LaserScan" in schema_name:
                decoded["scan"] = self.ros2_decoder.decode_laserscan(m.ros_msg)

            yield path, m, decoded
