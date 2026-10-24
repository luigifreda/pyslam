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
*
* Sparse TUM-VI mocap poses, converted from body to the tracking camera.
"""

import csv
from pathlib import Path

import numpy as np

from .ground_truth import GroundTruth, GroundTruthType
from .tum_vi_dataset import image_records
from pyslam.utilities.geometry import xyzq2Tmat, rotmat2qvec


class TumViGroundTruth(GroundTruth):
    def __init__(self, settings):
        path = str(Path(settings["base_path"]).expanduser())
        name = settings["name"]
        super().__init__(path, name, start_frame_id=int(settings.get("start_frame_id", 0)),
                         type=GroundTruthType.TUM_VI)
        sequence = Path(path) / name
        filename = settings.get("groundtruth_file", "auto")
        filename = sequence / "mav0/mocap0/data.csv" if filename == "auto" else sequence / filename
        Tbc = np.asarray(settings.get("T_body_camera", np.eye(4)), dtype=np.float64)
        records = []
        with filename.open() as handle:
            for row in csv.reader(handle):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                timestamp = int(row[0]) * 1e-9
                x, y, z, qw, qx, qy, qz = map(float, row[1:8])
                pose = xyzq2Tmat(x, y, z, qx, qy, qz, qw) @ Tbc
                if not np.isfinite(pose).all():
                    continue
                records.append((timestamp, pose))
        records.sort(key=lambda sample: sample[0])
        if not records:
            raise ValueError(f"No valid TUM-VI mocap poses in {filename}")
        self.timestamps = np.array([record[0] for record in records])
        self.poses = np.ascontiguousarray([record[1] for record in records])
        self.trajectory = self.poses[:, :3, 3].copy()
        # GroundTruth.save() serializes data with NumpyB64Json. Keep numeric
        # camera-frame samples, rather than heterogeneous (timestamp, pose) tuples.
        self.data = np.column_stack((self.timestamps, self.trajectory,
                                     [rotmat2qvec(pose[:3, :3]) for pose in self.poses]))
        self.filename = str(filename)
        self.found = True
        _, _, timestamps = image_records(sequence, settings.get("timestamps"),
                                        stereo=settings.get("sensor_type", "stereo") == "stereo")
        self.image_timestamps = np.asarray(timestamps, dtype=np.float64) * 1e-9
        self.max_gap = float(settings.get("groundtruth_max_gap", 0.02))

    def getFull3dTrajectory(self):
        return self.trajectory, self.timestamps

    def getFull6dTrajectory(self):
        return self.trajectory, self.poses, self.timestamps

    def convertToSimpleDataset(self, filename="groundtruth.txt"):
        # Export mocap samples directly: camera-index iteration would omit late
        # mocap segments whenever sparse mocap has fewer samples than images.
        scales = np.r_[1., np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1)]
        np.savetxt(filename, np.column_stack((self.data, scales)), fmt="%.15f")

    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        timestamp = self.image_timestamps[frame_id]
        index = self._pose_index(timestamp)
        pose = self.poses[index]
        qx, qy, qz, qw = rotmat2qvec(pose[:3, :3])
        scale = 1.0
        if frame_id > 0:
            try:
                previous = self._pose_index(self.image_timestamps[frame_id - 1])
                scale = float(np.linalg.norm(self.trajectory[index] - self.trajectory[previous]))
            except ValueError:
                pass  # The preceding camera exposure may fall in a mocap gap.
        return self.timestamps[index], *pose[:3, 3], qx, qy, qz, qw, scale

    def _pose_index(self, timestamp):
        index = int(np.searchsorted(self.timestamps, timestamp))
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.timestamps)]
        index = min(candidates, key=lambda i: abs(self.timestamps[i] - timestamp))
        if abs(self.timestamps[index] - timestamp) > self.max_gap:
            raise ValueError("No TUM-VI mocap pose near this camera frame (mocap is sparse)")
        return index

    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        sample = self.getTimestampPoseAndAbsoluteScale(frame_id)
        return *sample[:4], sample[-1]
