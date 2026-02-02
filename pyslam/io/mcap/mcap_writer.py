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

import time
from io import BufferedWriter
from typing import Dict, Optional, Union, IO

import numpy as np
from mcap.writer import CompressionType
from mcap_ros2.writer import Writer as McapROS2Writer
from mcap.records import Schema

from .ros2_encoder import Ros2Encoder


class McapWriter:
    """
    Simple convenience wrapper around mcap_ros2.writer.Writer.

    Features:
      - open an MCAP file for writing
      - register ROS2 schemas (datatypes + msgdef text)
      - write arbitrary ROS2 dict messages
      - convenience helpers for:
          * sensor_msgs/Image from NumPy
          * sensor_msgs/PointCloud2 from (N,3) NumPy
          * tf2_msgs/TFMessage (single TransformStamped)

    Usage:
        with McapWriter("output.mcap", compression=CompressionType.ZSTD) as w:
            # Register the schemas you plan to use
            w.register_schema(IMAGE_SCHEMA_NAME, IMAGE_SCHEMA_TEXT)
            w.register_schema(POINTCLOUD2_SCHEMA_NAME, POINTCLOUD2_SCHEMA_TEXT)
            w.register_schema(TF_SCHEMA_NAME, TF_SCHEMA_TEXT)

            # Write a dummy RGB image
            img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            w.write_image_from_numpy(
                topic="/camera/color/image_raw",
                image=img,
                frame_id="camera_color_optical_frame",
                encoding="rgb8",
            )

            # Write a point cloud
            pts = np.random.randn(10000, 3).astype(np.float32)
            w.write_pointcloud_xyz(
                topic="/points_raw",
                points_xyz=pts,
                frame_id="lidar_link",
            )

            # Write a TF
            w.write_tf(
                topic="/tf",
                parent_frame="world",
                child_frame="camera_link",
                translation_xyz=(1.0, 0.0, 0.5),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
    """

    def __init__(
        self,
        path_or_file: Union[str, IO[bytes]],
        chunk_size: int = 4 * 1024 * 1024,
        compression: CompressionType = CompressionType.ZSTD,
    ):
        # Handle string path or already-open file
        if isinstance(path_or_file, str):
            self._file = open(path_or_file, "wb")
            self._owns_file = True
        else:
            self._file = path_or_file
            self._owns_file = False

        if not isinstance(self._file, BufferedWriter):
            # mcap_ros2.Writer only needs a binary file-like, so this is just for type hints
            pass

        self._writer = McapROS2Writer(
            self._file,
            chunk_size=chunk_size,
            compression=compression,
        )
        self._schemas: Dict[str, Schema] = {}
        self._ros2_encoder = Ros2Encoder()

    # ------------------------------------------------------------------ #
    # Basic schema registration
    # ------------------------------------------------------------------ #
    def register_schema(self, datatype: str, msgdef_text: str) -> Schema:
        """
        Register a ROS2 message definition and return the Schema object.

        `datatype` is like: 'sensor_msgs/Image' or 'tf2_msgs/TFMessage'.
        `msgdef_text` is the concatenated .msg text for this type + deps.
        """
        schema = self._writer.register_msgdef(datatype, msgdef_text)
        self._schemas[datatype] = schema
        return schema

    def get_schema(self, datatype: str) -> Schema:
        if datatype not in self._schemas:
            raise KeyError(
                f"Datatype '{datatype}' not registered. "
                f"Call register_schema('{datatype}', msgdef_text) first."
            )
        return self._schemas[datatype]

    # ------------------------------------------------------------------ #
    # Low-level generic message writing
    # ------------------------------------------------------------------ #
    def write_message(
        self,
        topic: str,
        datatype: str,
        message: dict,
        log_time_ns: Optional[int] = None,
        publish_time_ns: Optional[int] = None,
        sequence: int = 0,
    ):
        """
        Write a generic ROS2 message as a dict, using a previously registered schema.
        """
        schema = self.get_schema(datatype)
        if log_time_ns is None:
            log_time_ns = time.time_ns()
        if publish_time_ns is None:
            publish_time_ns = log_time_ns

        self._writer.write_message(
            topic=topic,
            schema=schema,
            message=message,
            log_time=log_time_ns,
            publish_time=publish_time_ns,
            sequence=sequence,
        )

    def write_image_from_numpy(
        self,
        topic: str,
        image: np.ndarray,
        frame_id: str = "camera",
        encoding: str = "rgb8",
        stamp_ns: Optional[int] = None,
        datatype: str = "sensor_msgs/Image",
        sequence: int = 0,
    ):
        """
        Write a sensor_msgs/Image message built from a NumPy array.
        Supports grayscale or color images, e.g. HxW or HxWxC.
        `encoding` must match the array layout (e.g. 'rgb8', 'bgr8', 'mono8').
        """
        msg = self._ros2_encoder.encode_image_from_numpy(
            topic, image, frame_id, encoding, stamp_ns, datatype, sequence
        )
        self.write_message(
            topic=topic,
            datatype=datatype,
            message=msg,
            log_time_ns=stamp_ns,
            publish_time_ns=stamp_ns,
            sequence=sequence,
        )

    def write_pointcloud(
        self,
        topic: str,
        points_xyz: np.ndarray,
        colors_rgb: Optional[np.ndarray] = None,
        frame_id: str = "lidar",
        stamp_ns: Optional[int] = None,
        datatype: str = "sensor_msgs/PointCloud2",
        sequence: int = 0,
    ):
        """
        Write a sensor_msgs/PointCloud2 with only x,y,z fields from an (N,3) float32 array.
        """
        msg = self._ros2_encoder.encode_pointcloud(
            topic, points_xyz, colors_rgb, frame_id, stamp_ns, datatype, sequence
        )
        self.write_message(
            topic=topic,
            datatype=datatype,
            message=msg,
            log_time_ns=stamp_ns,
            publish_time_ns=stamp_ns,
            sequence=sequence,
        )

    def write_tf(
        self,
        topic: str,
        parent_frame: str,
        child_frame: str,
        translation_xyz,
        quaternion_xyzw,
        stamp_ns: Optional[int] = None,
        datatype: str = "tf2_msgs/TFMessage",
        sequence: int = 0,
    ):
        """
        Write a tf2_msgs/TFMessage with a single TransformStamped.
        """
        msg = self._ros2_encoder.encode_tf(
            topic,
            parent_frame,
            child_frame,
            translation_xyz,
            quaternion_xyzw,
            stamp_ns,
            datatype,
            sequence,
        )
        self.write_message(
            topic=topic,
            datatype=datatype,
            message=msg,
            log_time_ns=stamp_ns,
            publish_time_ns=stamp_ns,
            sequence=sequence,
        )

    # ------------------------------------------------------------------ #
    # Lifetime / context-manager
    # ------------------------------------------------------------------ #
    def finish(self):
        """
        Finish writing and close the underlying MCAP stream.

        You must call this (or use a context manager) to get a valid file.
        """
        self._writer.finish()
        if self._owns_file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()
