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

import numpy as np
import cv2

from dataclasses import dataclass

from pyslam.utilities.geometry import qvec2rotmat


@dataclass
class TransformData:
    parent: str
    child: str
    translation: np.ndarray[3]
    quaternion_xyzw: np.ndarray[4]
    matrix: np.ndarray[4, 4]
    stamp: float

    def __post_init__(self):
        self.matrix = qvec2rotmat(self.quaternion_xyzw)
        self.matrix[:3, 3] = self.translation


# -------------------------------------------------------------
#  TF
# -------------------------------------------------------------


def decode_tf_message(tf_msg):
    """
    tf2_msgs/TFMessage -> list of dicts:
    {
        'parent': str,
        'child': str,
        'translation': np.ndarray(3),
        'quaternion_xyzw': np.ndarray(4),
        'matrix': np.ndarray(4,4),
        'stamp': header.stamp (if present)
    }
    """
    transforms = []
    for t in tf_msg.transforms:
        tr = t.transform.translation
        rot = t.transform.rotation
        trans = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        quat = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64)
        R = qvec2rotmat(quat)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = trans
        # transforms.append(
        #     {
        #         "parent": t.header.frame_id,
        #         "child": t.child_frame_id,
        #         "translation": trans,
        #         "quaternion_xyzw": quat,
        #         "matrix": T,
        #         "stamp": getattr(t.header, "stamp", None),
        #     }
        # )
        transforms.append(
            TransformData(
                parent=t.header.frame_id,
                child=t.child_frame_id,
                translation=trans,
                quaternion_xyzw=quat,
                matrix=T,
                stamp=getattr(t.header, "stamp", None),
            )
        )
    return transforms


# -------------------------------------------------------------
#  PointCloud2
# -------------------------------------------------------------

# sensor_msgs/PointField datatype constants
_POINTFIELD_DTYPES = {
    1: ("int8", 1),
    2: ("uint8", 1),
    3: ("int16", 2),
    4: ("uint16", 2),
    5: ("int32", 4),
    6: ("uint32", 4),
    7: ("float32", 4),
    8: ("float64", 8),
}


def pointcloud2_to_array(msg):
    """
    sensor_msgs/PointCloud2 -> structured NumPy array (flattened).
    usage:
        arr = pointcloud2_to_array(msg)
        # If you have separate r, g, b fields:
        colors = np.vstack([arr["r"], arr["g"], arr["b"]]).T

        # Or if you have rgba:
        colors = np.vstack([arr["r"], arr["g"], arr["b"], arr["a"]]).T

        # You can also access any other fields:
        intensities = arr["intensity"]  # if present
        normals = np.vstack([arr["normal_x"], arr["normal_y"], arr["normal_z"]]).T  # if present

        # It does not unpack packed RGB fields. If RGB is stored as a single uint32 field named "rgb", it will return a single uint32 per point,
        # not separate r, g, b components. You would need to unpack it manually:
        if "rgb" in arr.dtype.names:
            rgb_packed = arr["rgb"].astype(np.uint32)
            r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
            g = ((rgb_packed >> 8) & 0xFF).astype(np.uint8)
            b = (rgb_packed & 0xFF).astype(np.uint8)
    """
    if msg.is_bigendian:
        raise NotImplementedError("Big-endian PointCloud2 not supported in this helper.")

    dtype_list = []
    offset = 0
    for f in msg.fields:
        while offset < f.offset:
            dtype_list.append(("_", np.uint8))
            offset += 1

        dtype_str, size = _POINTFIELD_DTYPES[f.datatype]
        dtype_list.append((f.name, np.dtype(dtype_str)))
        offset += size * f.count

    # padding after the last field
    while offset < msg.point_step:
        dtype_list.append(("_", np.uint8))
        offset += 1

    dtype = np.dtype(dtype_list)
    data = np.frombuffer(msg.data, dtype=dtype)
    # Flatten height/width into N points
    return data.reshape(-1)


def pointcloud2_to_xyz(msg, remove_nans=True):
    """
    sensor_msgs/PointCloud2 -> (N, 3) XYZ float32 array.
    """
    arr = pointcloud2_to_array(msg)
    if not {"x", "y", "z"}.issubset(arr.dtype.names):
        raise ValueError("PointCloud2 has no x/y/z fields")

    xyz = np.vstack(
        (arr["x"].astype(np.float32), arr["y"].astype(np.float32), arr["z"].astype(np.float32))
    ).T

    if remove_nans:
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]

    return xyz


def pointcloud2_to_xyz_rgb(msg, remove_nans=True):
    """
    sensor_msgs/PointCloud2 -> (N, 3) XYZ float32 array and (N, 3) RGB uint8 array.
    """
    arr = pointcloud2_to_array(msg)
    if not {"x", "y", "z", "r", "g", "b"}.issubset(arr.dtype.names):
        raise ValueError("PointCloud2 has no x/y/z/r/g/b fields")
    xyz = np.vstack(
        (arr["x"].astype(np.float32), arr["y"].astype(np.float32), arr["z"].astype(np.float32))
    ).T
    rgb = np.vstack(
        (arr["r"].astype(np.uint8), arr["g"].astype(np.uint8), arr["b"].astype(np.uint8))
    ).T

    if remove_nans:
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]
        rgb = rgb[mask]

    return xyz, rgb


# -------------------------------------------------------------
#  Image
# -------------------------------------------------------------


def _encoding_to_dtype_nchannels(encoding: str):
    enc = encoding.lower()
    if enc in ("mono8", "8uc1"):
        return np.uint8, 1
    if enc in ("mono16", "16uc1", "16sc1"):
        return np.uint16, 1
    if enc in ("bgr8", "rgb8"):
        return np.uint8, 3
    if enc in ("bgra8", "rgba8"):
        return np.uint8, 4
    if enc in ("32fc1",):
        return np.float32, 1
    raise ValueError(f"Unsupported image encoding: {encoding}")


def ros2_image_to_numpy(msg):
    """
    sensor_msgs/Image -> NumPy array shaped (H,W) or (H,W,C).
    Supports regular images (rgb8, mono8, etc.) and depth images (32FC1, 16UC1).
    """
    dtype, channels = _encoding_to_dtype_nchannels(msg.encoding)
    itemsize = np.dtype(dtype).itemsize

    buf = np.frombuffer(msg.data, dtype=dtype)
    total_pixels = len(buf)
    expected_pixels = msg.height * msg.width * channels

    # Check if data size matches expected size
    if total_pixels != expected_pixels:
        # If mismatch, try to infer dimensions from data size
        # This handles cases where step/width might be incorrect
        if total_pixels % (msg.height * channels) == 0:
            # Can infer width from data
            inferred_width = total_pixels // (msg.height * channels)
            img = buf.reshape(msg.height, inferred_width * channels)
            if channels == 1:
                return img.reshape(msg.height, inferred_width)
            else:
                return img.reshape(msg.height, inferred_width, channels)
        else:
            # Fallback: use step-based approach but validate
            row_stride = msg.step // itemsize
            if total_pixels < msg.height * row_stride:
                # Data is smaller than expected, adjust row_stride
                row_stride = total_pixels // msg.height
            img = buf.reshape(msg.height, row_stride)
            actual_width = min(row_stride, total_pixels // msg.height)
            img = img[:, : actual_width * channels]
            if channels == 1:
                return img.reshape(msg.height, actual_width)
            else:
                return img.reshape(msg.height, actual_width, channels)

    # Normal case: data size matches expected size
    row_stride = msg.step // itemsize
    img = buf.reshape(msg.height, row_stride)
    img = img[:, : msg.width * channels]

    if channels == 1:
        img = img.reshape(msg.height, msg.width)
    else:
        img = img.reshape(msg.height, msg.width, channels)
    return img


def ros2_compressed_image_to_numpy(msg):
    """
    sensor_msgs/CompressedImage -> NumPy image via OpenCV (if available).
    """
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    return img


# -------------------------------------------------------------
#  LaserScan
# -------------------------------------------------------------


def decode_laserscan(scan_msg):
    """
    sensor_msgs/LaserScan -> dict with numpy arrays.
    """
    return {
        "ranges": np.asarray(scan_msg.ranges, dtype=np.float32),
        "intensities": np.asarray(scan_msg.intensities, dtype=np.float32),
        "angle_min": scan_msg.angle_min,
        "angle_max": scan_msg.angle_max,
        "angle_increment": scan_msg.angle_increment,
        "time_increment": scan_msg.time_increment,
        "scan_time": scan_msg.scan_time,
        "range_min": scan_msg.range_min,
        "range_max": scan_msg.range_max,
    }


# -------------------------------------------------------------
#  Ros2Decoder
# -------------------------------------------------------------


class Ros2Decoder:
    def __init__(self):
        self.cache = {}

    def decode_tf_message(self, tf_msg):
        return decode_tf_message(tf_msg)

    def decode_pointcloud2(self, msg):
        return pointcloud2_to_xyz(msg)

    def decode_image(self, msg):
        return ros2_image_to_numpy(msg)

    def decode_compressed_image(self, msg):
        return ros2_compressed_image_to_numpy(msg)

    def decode_laserscan(self, msg):
        return decode_laserscan(msg)
