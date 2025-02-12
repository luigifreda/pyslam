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
import quaternion
import os


class UnsupportedFormatException(Exception):
    def __init__(self, format_type, message="Unsupported trajectory format"):
        self.format_type = format_type
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}: {self.format_type}'


class TrajectoryWriter:
    KITTI_FORMAT = 'kitti'
    TUM_FORMAT = 'tum'
    EUROC_FORMAT = 'euroc'

    FORMAT_FUNCTIONS = {
        KITTI_FORMAT: '_write_kitti_trajectory',
        TUM_FORMAT: '_write_tum_trajectory',
        EUROC_FORMAT: '_write_euroc_trajectory'
    }

    def __init__(self, format_type: str, filename: str = None) -> None:
        self.filename = filename or self.generate_filename()
        self.file = None
        self.open_file()        
        self.set_format_type(format_type)      
        
    def __del__(self):
        self.close_file()

    def open_file(self):
        folder_path = os.path.dirname(self.filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not self.file:
            self.file = open(self.filename, 'w')
            print(f'Trajectory file {self.filename} opened')
        else:
            print(f'ERROR: trajectory file {self.filename} already open')

    def close_file(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f'Trajectory saved to {self.filename}')
            
    def set_format_type(self, format_type):
        self.format_type = format_type.lower()     
        write_func_name = self.FORMAT_FUNCTIONS.get(self.format_type)
        if write_func_name:
            self.write_function = getattr(self, write_func_name)
        else:
            raise UnsupportedFormatException(self.format_type)             

    def write_full_trajectory(self, poses, timestamps):
        for i in range(len(poses)):
            R = poses[i][:3, :3]
            t = poses[i][:3, 3]
            timestamp = timestamps[i]
            self.write_trajectory(R, t, timestamp)

    def write_trajectory(self, R, t, timestamp):
        self.write_function(R, t, timestamp)

    def _write_kitti_trajectory(self, R, t, timestamp) -> None:
        self.file.write(f"{R[0, 0]:.9f} {R[0, 1]:.9f} {R[0, 2]:.9f} {t[0]:.9f} " \
               f"{R[1, 0]:.9f} {R[1, 1]:.9f} {R[1, 2]:.9f} {t[1]:.9f} " \
               f"{R[2, 0]:.9f} {R[2, 1]:.9f} {R[2, 2]:.9f} {t[2]:.9f}\n")

    def _write_tum_trajectory(self, R, t, timestamp) -> None:
        q = quaternion.from_rotation_matrix(R)
        self.file.write(f"{timestamp:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n")

    def _write_euroc_trajectory(self, R, t, timestamp) -> None:
        t_str = ', '.join(map(str, t))
        q = quaternion.from_rotation_matrix(R)
        pose_line = f"{timestamp:.6f}, {t_str}, {q.x:.9f}, {q.y:.9f}, {q.z:.9f}, {q.w:.9f}"
        self.file.write(pose_line + '\n')

    @staticmethod
    def generate_filename():
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"trajectory_estimates_{now}.txt"
