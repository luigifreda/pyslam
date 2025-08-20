"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
* Copyright (C) 2024 Anathonic <anathonic@protonmail.com>
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

from datetime import datetime

# import quaternion
from scipy.spatial.transform import Rotation as R_scipy
import numpy as np
import os


class UnsupportedFormatException(Exception):
    def __init__(self, format_type, message="Unsupported trajectory format"):
        self.format_type = format_type
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.format_type}"


class TrajectoryWriter:
    KITTI_FORMAT = "kitti"
    TUM_FORMAT = "tum"
    EUROC_FORMAT = "euroc"

    FORMAT_FUNCTIONS = {
        KITTI_FORMAT: "_write_kitti_trajectory",
        TUM_FORMAT: "_write_tum_trajectory",
        EUROC_FORMAT: "_write_euroc_trajectory",
    }

    def __init__(self, format_type: str, filename: str = None) -> None:
        self.filename = filename if filename is not None else self.generate_filename()
        self.file = None
        self.open_file()
        self.set_format_type(format_type)

    def __del__(self):
        self.close_file()

    def open_file(self):
        folder_path = os.path.dirname(self.filename)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not self.file:
            self.file = open(self.filename, "w")
            print(f"Trajectory file {self.filename} opened")
        else:
            print(f"ERROR: trajectory file {self.filename} already open")

    def close_file(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"Trajectory saved to {self.filename}")

    def set_format_type(self, format_type):
        self.format_type = format_type.lower()
        write_func_name = self.FORMAT_FUNCTIONS.get(self.format_type)
        if write_func_name:
            self.write_function = getattr(self, write_func_name)
        else:
            raise UnsupportedFormatException(self.format_type)

    def write_full_trajectory(self, poses, timestamps):
        for i, pose in enumerate(poses):
            R = pose[:3, :3]
            t = pose[:3, 3]
            timestamp = timestamps[i]
            self.write_trajectory(R, t, timestamp)

    def write_trajectory(self, R, t, timestamp):
        self.write_function(R, t, timestamp)

    def _get_format_func(self, array_or_scalar):
        """Return appropriate format function based on dtype or type."""
        if isinstance(array_or_scalar, np.ndarray):
            dtype = array_or_scalar.dtype
        elif isinstance(array_or_scalar, (np.float32, np.float64)):
            dtype = array_or_scalar.dtype
        elif isinstance(array_or_scalar, float):
            dtype = np.dtype("float64")  # Python float is 64-bit
        else:
            dtype = np.dtype(type(array_or_scalar))

        if dtype == np.float32:
            return lambda x: f"{x:.7g}"
        else:
            return lambda x: f"{x:.17g}"

    def _write_kitti_trajectory(self, R, t, timestamp) -> None:
        fmt = self._get_format_func(R)
        elements = [
            R[0, 0],
            R[0, 1],
            R[0, 2],
            t[0],
            R[1, 0],
            R[1, 1],
            R[1, 2],
            t[1],
            R[2, 0],
            R[2, 1],
            R[2, 2],
            t[2],
        ]
        self.file.write(" ".join(fmt(x) for x in elements) + "\n")

    def _write_tum_trajectory(self, R, t, timestamp) -> None:
        fmt = self._get_format_func(t)
        timestamp_fmt = self._get_format_func(timestamp)
        # q = quaternion.from_rotation_matrix(R)
        q = R_scipy.from_matrix(R).as_quat()  # [x, y, z, w]
        # elements = [timestamp, t[0], t[1], t[2], q.x, q.y, q.z, q.w]
        elements = [timestamp, t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
        self.file.write(
            " ".join(fmt(x) if i else timestamp_fmt(x) for i, x in enumerate(elements)) + "\n"
        )

    def _write_euroc_trajectory(self, R, t, timestamp) -> None:
        fmt = self._get_format_func(t)
        timestamp_fmt = self._get_format_func(timestamp)
        # q = quaternion.from_rotation_matrix(R)
        q = R_scipy.from_matrix(R).as_quat()  # [x, y, z, w]
        t_str = ", ".join(fmt(x) for x in t)
        # q_str = ', '.join(fmt(x) for x in [q.x, q.y, q.z, q.w])
        q_str = ", ".join(fmt(x) for x in [q[0], q[1], q[2], q[3]])
        self.file.write(f"{timestamp_fmt(timestamp)}, {t_str}, {q_str}\n")

    @staticmethod
    def generate_filename():
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"trajectory_estimates_{now}.txt"
