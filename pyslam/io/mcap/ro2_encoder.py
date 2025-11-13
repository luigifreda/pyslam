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


ROS2_POINT_FIELD_DTYPES = {
    "FLOAT32": 7,
    "UINT8": 2,
    "INT16": 3,
    "UINT16": 4,
    "INT32": 5,
    "UINT32": 6,
    "FLOAT64": 8,
    "INT64": 9,
    "UINT64": 10,
}


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #


def _split_stamp_ns(stamp_ns: int):
    """
    Split a timestamp in nanoseconds into seconds and nanoseconds.
    """
    sec = stamp_ns // 1_000_000_000
    nanosec = stamp_ns % 1_000_000_000
    return int(sec), int(nanosec)


# ------------------------------------------------------------------ #
# Image encoding
# ------------------------------------------------------------------ #


def encode_image_from_numpy(
    topic: str,
    image: np.ndarray,
    frame_id: str = "camera",
    encoding: str = "rgb8",
    stamp_ns: Optional[int] = None,
    datatype: str = "sensor_msgs/Image",
    sequence: int = 0,
):
    """
    Encode a sensor_msgs/Image message built from a NumPy array.
    Supports grayscale or color images, e.g. HxW or HxWxC.
    Also supports depth images: use encoding='32FC1' for float32 depth or '16UC1' for uint16 depth.
    `encoding` must match the array layout (e.g. 'rgb8', 'bgr8', 'mono8', '32FC1', '16UC1').
    """
    if stamp_ns is None:
        stamp_ns = time.time_ns()

    h, w = image.shape[:2]
    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]

    # Auto-detect or correct encoding for depth images (2D, single channel) based on dtype
    if channels == 1 and image.ndim == 2:
        if image.dtype == np.uint16:
            # uint16 depth should use 16UC1 - override encoding if it doesn't match
            if encoding.lower() not in ("16uc1", "16sc1", "mono16"):
                encoding = "16UC1"
        elif image.dtype == np.float32:
            # float32 depth should use 32FC1 - override encoding if it doesn't match
            if encoding.lower() != "32fc1":
                encoding = "32FC1"
        elif image.dtype == np.uint8:
            # uint8 grayscale - only auto-detect if encoding is not specified
            if encoding.lower() not in ("mono8", "8uc1", "16uc1", "16sc1", "32fc1"):
                encoding = "mono8"

    # Handle different data types based on encoding
    # Check encoding first to determine the correct dtype handling
    if encoding.lower() == "32fc1":
        # Force float32 for 32FC1 encoding
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        data = np.ascontiguousarray(image).tobytes()
        bytes_per_pixel = 4 * channels
    elif encoding.lower() in ("16uc1", "16sc1", "mono16"):
        # Force uint16/int16 for 16-bit encodings
        if image.dtype not in (np.uint16, np.int16):
            image = image.astype(np.uint16)
        data = np.ascontiguousarray(image).tobytes()
        bytes_per_pixel = 2 * channels
    elif image.dtype == np.float32:
        # Float32 image (shouldn't happen for standard encodings, but handle it)
        data = np.ascontiguousarray(image).tobytes()
        bytes_per_pixel = 4 * channels
    elif image.dtype == np.uint16:
        # Uint16 image
        data = np.ascontiguousarray(image).tobytes()
        bytes_per_pixel = 2 * channels
    else:
        # Default to uint8 for regular images
        img_uint8 = np.ascontiguousarray(image.astype(np.uint8))
        data = img_uint8.tobytes()
        bytes_per_pixel = 1 * channels

    step = w * bytes_per_pixel
    sec, nanosec = _split_stamp_ns(stamp_ns)

    msg = {
        "header": {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": frame_id,
        },
        "height": h,
        "width": w,
        "encoding": encoding,
        "is_bigendian": 0,
        "step": step,
        "data": data,
    }
    return msg


# ------------------------------------------------------------------ #
# PointCloud2 encoding
# ------------------------------------------------------------------ #


def encode_pointcloud(
    topic: str,
    points_xyz: np.ndarray,
    colors_rgb: Optional[np.ndarray] = None,
    frame_id: str = "lidar",
    stamp_ns: Optional[int] = None,
    datatype: str = "sensor_msgs/PointCloud2",
    sequence: int = 0,
):
    """
    Encode a sensor_msgs/PointCloud2 with only x,y,z fields from an (N,3) float32 array.
    """
    if stamp_ns is None:
        stamp_ns = time.time_ns()

    pts = np.asarray(points_xyz, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 3, "points_xyz must be (N,3)"

    sec, nanosec = _split_stamp_ns(stamp_ns)
    N = pts.shape[0]

    # ROS2 PointField constants: FLOAT32 = 7, UINT8 = 2
    FLOAT32 = ROS2_POINT_FIELD_DTYPES["FLOAT32"]
    UINT8 = ROS2_POINT_FIELD_DTYPES["UINT8"]
    fields = [
        {"name": "x", "offset": 0, "datatype": FLOAT32, "count": 1},
        {"name": "y", "offset": 4, "datatype": FLOAT32, "count": 1},
        {"name": "z", "offset": 8, "datatype": FLOAT32, "count": 1},
    ]
    if colors_rgb is not None:
        assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3, "colors_rgb must be (N,3)"
        assert colors_rgb.shape[0] == N, "colors_rgb must have same number of points as points_xyz"
        # Convert colors to uint8 if needed
        colors_uint8 = np.ascontiguousarray(colors_rgb.astype(np.uint8))
        fields.append({"name": "r", "offset": 12, "datatype": UINT8, "count": 1})
        fields.append({"name": "g", "offset": 13, "datatype": UINT8, "count": 1})
        fields.append({"name": "b", "offset": 14, "datatype": UINT8, "count": 1})
        data = np.concatenate([pts, colors_uint8], axis=1).tobytes()
    else:
        data = pts.tobytes()

    # We'll store as unorganized cloud: height=1, width=N
    height = 1
    width = N
    if colors_rgb is not None:
        point_step = 12 + 3 * 1  # 3 * 4-byte float32 + 3 * 1-byte uint8
    else:
        point_step = 12  # 3 * 4-byte float32
    row_step = point_step * width

    msg = {
        "header": {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": frame_id,
        },
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": False,
        "point_step": point_step,
        "row_step": row_step,
        "data": data,
        "is_dense": True,
    }
    return msg


# ------------------------------------------------------------------ #
# TF encoding
# ------------------------------------------------------------------ #


def encode_tf(
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
    if stamp_ns is None:
        stamp_ns = time.time_ns()
    sec, nanosec = _split_stamp_ns(stamp_ns)

    tx, ty, tz = translation_xyz
    qx, qy, qz, qw = quaternion_xyzw

    msg = {
        "transforms": [
            {
                "header": {
                    "stamp": {"sec": sec, "nanosec": nanosec},
                    "frame_id": parent_frame,
                },
                "child_frame_id": child_frame,
                "transform": {
                    "translation": {"x": tx, "y": ty, "z": tz},
                    "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
                },
            }
        ]
    }
    return msg


class Ros2Encoder:
    def __init__(self):
        pass

    def encode_image_from_numpy(
        self,
        topic: str,
        image: np.ndarray,
        frame_id: str = "camera",
        encoding: str = "rgb8",
        stamp_ns: Optional[int] = None,
        datatype: str = "sensor_msgs/Image",
        sequence: int = 0,
    ):
        return encode_image_from_numpy(
            topic, image, frame_id, encoding, stamp_ns, datatype, sequence
        )

    def encode_pointcloud(
        self,
        topic: str,
        points_xyz: np.ndarray,
        colors_rgb: Optional[np.ndarray] = None,
        frame_id: str = "lidar",
        stamp_ns: Optional[int] = None,
        datatype: str = "sensor_msgs/PointCloud2",
        sequence: int = 0,
    ):
        return encode_pointcloud(
            topic, points_xyz, colors_rgb, frame_id, stamp_ns, datatype, sequence
        )

    def encode_tf(
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
        return encode_tf(
            topic,
            parent_frame,
            child_frame,
            translation_xyz,
            quaternion_xyzw,
            stamp_ns,
            datatype,
            sequence,
        )
