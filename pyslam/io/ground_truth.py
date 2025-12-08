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
import csv
import json
import numpy as np
import traceback

from enum import Enum

from pyslam.utilities.system import Printer
from pyslam.utilities.geometry import rotmat2qvec, xyzq2Tmat
from pyslam.utilities.serialization import (
    SerializableEnum,
    register_class,
    NumpyJson,
    NumpyB64Json,
    SerializationJSON,
)


kScaleSimple = 1.0
kScaleKitti = 1.0
kScaleTum = 1.0
kScaleEuroc = 1.0
kScaleReplica = 1.0
kScaleScannet = 1.0


@register_class
class GroundTruthType(SerializableEnum):
    NONE = 1
    KITTI = 2
    TUM = 3
    EUROC = 4
    REPLICA = 5
    TARTANAIR = 6
    SIMPLE = 7
    SCANNET = 8


def groundtruth_factory(settings):

    type = GroundTruthType.NONE
    associations = None

    type = settings["type"]
    path = settings["base_path"]
    name = settings["name"]

    start_frame_id = 0
    if "start_frame_id" in settings:
        Printer.orange(f'groundtruth_factory - start_frame_id: {settings["start_frame_id"]}')
        start_frame_id = int(settings["start_frame_id"])

    print("using groundtruth: ", type)
    if type == "kitti":
        return KittiGroundTruth(
            path, name, associations, start_frame_id, type=GroundTruthType.KITTI
        )
    if type == "tum":
        if "associations" in settings:
            associations = settings["associations"]
        return TumGroundTruth(path, name, associations, start_frame_id, type=GroundTruthType.TUM)
    if type == "euroc":
        return EurocGroundTruth(
            path, name, associations, start_frame_id, type=GroundTruthType.EUROC
        )
    if type == "replica":
        return ReplicaGroundTruth(
            path, name, associations, start_frame_id, type=GroundTruthType.REPLICA
        )
    if type == "tartanair":
        return TartanairGroundTruth(
            path, name, associations, start_frame_id, type=GroundTruthType.TARTANAIR
        )
    if type == "scannet":
        return ScannetGroundTruth(
            path, name, associations, start_frame_id, type=GroundTruthType.SCANNET
        )
    if type == "video" or type == "folder":
        if "groundtruth_file" in settings:
            name = settings["groundtruth_file"]
            return SimpleGroundTruth(
                path, name, associations, start_frame_id, type=GroundTruthType.SIMPLE
            )
    if type == "ros1bag":
        if "groundtruth_file" in settings:
            name = settings["groundtruth_file"]
            return SimpleGroundTruth(
                None, name, associations, start_frame_id, type=GroundTruthType.SIMPLE
            )
    print("[groundtruth_factory] Not using groundtruth")
    print(
        "[groundtruth_factory] If you are using main_vo.py, your estimated trajectory will not make sense!"
    )
    return GroundTruth(path, name, associations, start_frame_id, type=GroundTruthType.NONE)


# base class
class GroundTruth(object):
    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.NONE):
        self.path = path
        self.name = name
        self.type = type
        self.associations = associations  # name of the file containing the associations
        self.filename = None
        self.associations_path = None
        self.data = None
        self.scale = 1
        self.start_frame_id = start_frame_id

        self.trajectory = None  # 3d position trajectory
        self.timestamps = None  # array of timestamps corresponding to the trajectory
        self.poses = None  # 6d pose trajectory in the form on an array of homogeneous transformation matrices

    def getDataLine(self, frame_id):
        frame_id += self.start_frame_id
        return self.data[frame_id].strip().split()

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        return 1, 0, 0, 0, 1

    # return timestamp,x,y,z,qx,qy,qz,qw,scale
    # NOTE: Keep in mind that datasets may not have orientation information!
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        return 1, 0, 0, 0, 0, 0, 0, 1, 1

    # convert the dataset into 'Simple' format [timestamp, x,y,z, qx,qy,qz,qw, scale]
    def convertToSimpleDataset(self, filename="groundtruth.txt"):
        if self.data is None:
            Printer.yellow("[GroundTruth] convertToSimpleDataset: data is None")
            return
        out_file = open(filename, "w")
        num_lines = len(self.data)
        for ii in range(num_lines):
            try:
                timestamp, x, y, z, qx, qy, qz, qw, scale = self.getTimestampPoseAndAbsoluteScale(
                    ii
                )
                if ii == 0:
                    scale = 1  # first sample: we do not have a relative
                print(
                    f"writing timestamp: {timestamp:.15f}, x: {x:.15f}, y: {y:.15f}, z: {z:.15f}, qx: {qx:.15f}, qy: {qy:.15f}, qz: {qz:.15f}, qw: {qw:.15f}, scale: {scale:.15f}"
                )
                out_file.write(
                    f"{timestamp:.15f} {x:.15f} {y:.15f} {z:.15f} {qx:.15f} {qy:.15f} {qz:.15f} {qw:.15f} {scale:.15f}\n"
                )  # (timestamp,x,y,z,scale) )
            except:
                pass
                # print(f'error in line {ii}')
                # print(traceback.format_exc())
        out_file.close()

    def getNumSamples(self):
        if self.data is None:
            Printer.yellow("[GroundTruth] getNumSamples: data is None")
            return 0
        num_lines = len(self.data)
        return num_lines

    def getClosestTimestamp(self, timestamp):
        if self.timestamps is None:
            Printer.red("ERROR: GroundTruth: getClosestTimestamp() called setting the timestamps!")
            return None
        return self.timestamps[np.argmin(np.abs(self.timestamps - timestamp))]

    def getClosestPosition(self, timestamp):
        if self.trajectory is None:
            Printer.red("ERROR: GroundTruth: getClosestPose() called setting the trajectory!")
            return None
        return self.trajectory[np.argmin(np.abs(self.timestamps - timestamp))]

    def getClosestPose(self, timestamp):
        if self.poses is None:
            Printer.red("ERROR: GroundTruth: getClosestPose() called setting the poses!")
            return None
        idx = np.argmin(np.abs(self.timestamps - timestamp))
        # print(f'getClosestPose(): idx: {idx}, timestamp: {self.timestamps[idx]}')
        return self.poses[idx]

    def getFull3dTrajectory(self):
        if self.trajectory is not None and self.timestamps is not None:
            return self.trajectory, self.timestamps
        if self.data is None:
            Printer.yellow("[GroundTruth] getNumSamples: data is None")
            return self.trajectory, self.timestamps
        num_lines = len(self.data)
        self.trajectory = []
        self.timestamps = []
        for ii in range(1, num_lines - 1):
            try:
                timestamp, x, y, z, scale = self.getTimestampPositionAndAbsoluteScale(ii)
                # print(f'timestamp: {timestamp}, x: {x}, y: {y}, z: {z}, scale: {scale}')
                self.timestamps.append(timestamp)
                self.trajectory.append([x, y, z])
            except:
                pass
        self.timestamps = np.ascontiguousarray(self.timestamps, dtype=np.float64)
        self.trajectory = np.ascontiguousarray(self.trajectory, dtype=np.float64)
        return self.trajectory, self.timestamps

    def getFull6dTrajectory(self):
        if self.trajectory is not None and self.poses is not None and self.timestamps is not None:
            return self.trajectory, self.poses, self.timestamps
        if self.data is None:
            Printer.yellow("[GroundTruth] getNumSamples: data is None")
            return self.trajectory, self.poses, self.timestamps
        num_lines = len(self.data)
        self.trajectory = []
        self.poses = []
        self.timestamps = []
        for ii in range(1, num_lines - 1):
            try:
                timestamp, x, y, z, qx, qy, qz, qw, scale = self.getTimestampPoseAndAbsoluteScale(
                    ii
                )
                # print(f'timestamp: {timestamp}, x: {x}, y: {y}, z: {z}, scale: {scale}')
                self.timestamps.append(timestamp)
                self.trajectory.append([x, y, z])
                self.poses.append(xyzq2Tmat(x, y, z, qx, qy, qz, qw))
            except:
                pass
        self.timestamps = np.ascontiguousarray(self.timestamps, dtype=float)
        self.trajectory = np.ascontiguousarray(self.trajectory, dtype=float)
        self.poses = np.ascontiguousarray(self.poses, dtype=float)
        return self.trajectory, self.poses, self.timestamps

    def to_json(self):
        self.getFull6dTrajectory()  # to make sure we have everything
        ret = {
            "path": self.path,
            "name": self.name,
            "type": SerializationJSON.serialize(self.type),
            "start_frame_id": self.start_frame_id,
            "filename": self.filename,
            "associations": self.associations,
            "data": json.dumps(NumpyB64Json.numpy_to_json(self.data)),
            "timestamps": (
                json.dumps(NumpyB64Json.numpy_to_json(self.timestamps))
                if self.timestamps is not None
                else None
            ),
            "trajectory": (
                json.dumps(NumpyB64Json.numpy_to_json(self.trajectory))
                if self.trajectory is not None
                else None
            ),
            "poses": (
                json.dumps(NumpyB64Json.numpy_to_json(self.poses))
                if self.poses is not None
                else None
            ),
        }
        return ret

    @staticmethod
    def from_json(json_str):
        path = json_str["path"]
        name = json_str["name"]
        type = SerializationJSON.deserialize(json_str["type"])
        start_frame_id = json_str["start_frame_id"]
        filename = json_str["filename"]
        associations = json_str["associations"]
        data = NumpyB64Json.json_to_numpy(json.loads(json_str["data"]))
        timestamps = (
            NumpyB64Json.json_to_numpy(json.loads(json_str["timestamps"]))
            if json_str["timestamps"] is not None
            else None
        )
        trajectory = (
            NumpyB64Json.json_to_numpy(json.loads(json_str["trajectory"]))
            if json_str["trajectory"] is not None
            else None
        )
        poses = (
            NumpyB64Json.json_to_numpy(json.loads(json_str["poses"]))
            if json_str["poses"] is not None
            else None
        )

        gt = GroundTruth(
            path=path,
            name=name,
            type=type,
            start_frame_id=start_frame_id,
            associations=associations,
        )
        gt.filename = filename
        gt.data = data
        gt.timestamps = timestamps
        gt.trajectory = trajectory
        gt.poses = poses
        return gt

    def save(self, path):
        filepath = path + "/gt.json"
        with open(filepath, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(path):
        filepath = os.path.join(path, "gt.json")
        if not os.path.exists(filepath):
            Printer.yellow(f"WARNING: GroundTruth.load(): file {filepath} does not exist!")
            return None
        with open(filepath, "r") as f:
            json_str = json.load(f)
        return GroundTruth.from_json(json_str)

    @staticmethod
    def associate(first_list, second_list, offset=0, max_difference=0.08 * 1e9):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first list of (stamp,data) tuples
        second_list -- second list of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- map: index_stamp_first -> (index_stamp_second, diff_stamps, first_timestamp, second_timestamp)
        """
        matches = {}
        first_flag = [False] * len(first_list)
        second_flag = [False] * len(second_list)
        # extract timestamps
        t1 = np.ascontiguousarray([float(data1[0]) for data1 in first_list])
        t2 = np.ascontiguousarray([(float(data2[0]) + offset) for data2 in second_list])
        for i, t in enumerate(t1):
            j = np.argmin(np.abs(t2 - t))
            if abs(t2[j] - t) < max_difference:
                first_flag[i] = True
                second_flag[j] = True
                matches[int(i)] = (int(j), abs(t2[j] - t), t, t2[j])
        missing_associations = [(i, a) for i, a in enumerate(first_list) if first_flag[i] is False]
        num_missing_associations = len(missing_associations)
        if num_missing_associations > 0:
            Printer.red(f"ERROR: {num_missing_associations} missing associations!")
        print(f"[associate] Number of matches: {len(matches)}")
        return matches


# Read the ground truth from a simple file containining [timestamp, x,y,z, qx, qy, qz, qw, scale] lines
# Use the file io/convert_kitti_groundtruth_to_simple.py to convert the ground truth of a dataset into this format.
class SimpleGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.KITTI):
        super().__init__(path, name, associations, start_frame_id, type)
        self.scale = kScaleSimple
        if path is not None:
            self.filename = path + "/" + name
        else:
            self.filename = name

        if not os.path.isfile(self.filename):
            error_message = (
                f"ERROR: [SimpleGroundTruth] Groundtruth file not found: {self.filename}!"
            )
            Printer.red(error_message)
            sys.exit(error_message)

        with open(self.filename) as f:
            self.data = f.readlines()
            self.data = np.ascontiguousarray(self.data)
            self.found = True
        if self.data is None:
            sys.exit(
                "ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!"
            )

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[1])
            y_prev = self.scale * float(ss[2])
            z_prev = self.scale * float(ss[3])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = float(ss[0])
        x = self.scale * float(ss[1])
        y = self.scale * float(ss[2])
        z = self.scale * float(ss[3])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        # print(f'reading frame {frame_id}, timestamp: {timestamp:.15f}, x: {x:.15f}, y: {y:.15f}, z: {z:.15f}, scale: {abs_scale:.15f}')
        return timestamp, x, y, z, abs_scale

    # return timestamp, x,y,z, qx,qy,qz,qw, scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[1])
            y_prev = self.scale * float(ss[2])
            z_prev = self.scale * float(ss[3])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = float(ss[0])
        x = self.scale * float(ss[1])
        y = self.scale * float(ss[2])
        z = self.scale * float(ss[3])
        qx = float(ss[4])
        qy = float(ss[5])
        qz = float(ss[6])
        qw = float(ss[7])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, qx, qy, qz, qw, abs_scale


class KittiGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.KITTI):
        super().__init__(path, name, associations, start_frame_id, type)
        self.scale = kScaleKitti
        self.filename = (
            path + "/poses/" + name + ".txt"
        )  # N.B.: this may depend on how you deployed the groundtruth files
        self.filename_timestamps = path + "/sequences/" + name + "/times.txt"

        if not os.path.isfile(self.filename):
            error_message = (
                f"ERROR: [KittiGroundTruth] Groundtruth file not found: {self.filename}!"
            )
            Printer.red(error_message)
            sys.exit(error_message)

        with open(self.filename) as f:
            self.data = f.readlines()
            self.data = np.ascontiguousarray(self.data)
            self.found = True
        if self.data is None:
            sys.exit(
                "ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!"
            )
        self.data_timestamps = None
        with open(self.filename_timestamps) as f:
            self.data_timestamps = f.readlines()
            self.found = True
        if self.data_timestamps is None:
            sys.exit(
                "ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!"
            )

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[3])
            y_prev = self.scale * float(ss[7])
            z_prev = self.scale * float(ss[11])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        x = self.scale * float(ss[3])
        y = self.scale * float(ss[7])
        z = self.scale * float(ss[11])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        timestamp = float(self.data_timestamps[frame_id].strip())
        # print(f'reading frame {frame_id}, timestamp: {timestamp:.15f}, x: {x:.15f}, y: {y:.15f}, z: {z:.15f}, scale: {abs_scale:.15f}')
        return timestamp, x, y, z, abs_scale

    # return timestamp,x,y,z,qx,qy,qz,qw,scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[3])
            y_prev = self.scale * float(ss[7])
            z_prev = self.scale * float(ss[11])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        x = self.scale * float(ss[3])
        y = self.scale * float(ss[7])
        z = self.scale * float(ss[11])
        r11 = float(ss[0])
        r12 = float(ss[1])
        r13 = float(ss[2])
        r21 = float(ss[4])
        r22 = float(ss[5])
        r23 = float(ss[6])
        r31 = float(ss[8])
        r32 = float(ss[9])
        r33 = float(ss[10])
        R = np.ascontiguousarray([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        q = rotmat2qvec(R)  # [qx, qy, qz, qw]
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        timestamp = float(self.data_timestamps[frame_id].strip())
        # print(f'reading frame {frame_id}, timestamp: {timestamp:.15f}, x: {x:.15f}, y: {y:.15f}, z: {z:.15f}, scale: {abs_scale:.15f}')
        return timestamp, x, y, z, q[0], q[1], q[2], q[3], abs_scale


class TumGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.TUM):
        super().__init__(path, name, associations, start_frame_id, type)
        self.scale = kScaleTum
        self.filename = (
            path + "/" + name + "/" + "groundtruth.txt"
        )  # N.B.: this may depend on how you deployed the groundtruth files
        if not os.path.isfile(self.filename):
            self.filename = path + "/" + name + "/" + "gt.freiburg"  # For ICL-NUIM support
        self.associations_path = (
            path + "/" + name + "/" + associations
        )  # N.B.: this may depend on how you name the associations file

        if not os.path.isfile(self.filename):
            error_message = f"ERROR: [TumGroundTruth] Groundtruth file not found: {self.filename}!"
            Printer.red(error_message)
            sys.exit(error_message)

        base_path = os.path.dirname(self.filename)
        print("[TumGroundTruth] base_path: ", base_path)

        with open(self.filename) as f:
            self.data = f.readlines()[3:]  # skip the first three rows, which are only comments
            self.data = [line.strip().split() for line in self.data]
            self.data = np.ascontiguousarray(self.data)
        if self.data is None:
            sys.exit("ERROR [TumGroundTruth] while reading groundtruth file!")
        if self.associations_path is not None:
            with open(self.associations_path) as f:
                self.associations_data = f.readlines()
                self.associations_data = [line.strip().split() for line in self.associations_data]
            if self.associations_data is None:
                sys.exit("ERROR [TumGroundTruth] while reading associations file!")

        associations_file = base_path + "/gt_associations.json"
        if True:  # not os.path.exists(associations_file):
            # Printer.orange('Computing groundtruth associations (one-time operation, results will be saved)...')
            if len(self.associations_data) == 0 or len(self.data) == 0:
                Printer.orange(
                    f"WARNING: you have #associations = {len(self.associations_data)} and #groundtruth samples = {len(self.data)}"
                )
            self.association_matches = self.associate(self.associations_data, self.data)
            # save associations
            with open(associations_file, "w") as f:
                json.dump(self.association_matches, f)
        else:
            with open(associations_file, "r") as f:
                data = json.load(f)
                self.association_matches = {int(k): v for k, v in data.items()}

    def getDataLine(self, frame_id):
        return self.data[self.association_matches[frame_id][0]]

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[1])
            y_prev = self.scale * float(ss[2])
            z_prev = self.scale * float(ss[3])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = float(ss[0])
        x = self.scale * float(ss[1])
        y = self.scale * float(ss[2])
        z = self.scale * float(ss[3])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, abs_scale

    # return timestamp, x,y,z, qx,qy,qz,qw, scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[1])
            y_prev = self.scale * float(ss[2])
            z_prev = self.scale * float(ss[3])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = float(ss[0])
        x = self.scale * float(ss[1])
        y = self.scale * float(ss[2])
        z = self.scale * float(ss[3])
        qx = float(ss[4])
        qy = float(ss[5])
        qz = float(ss[6])
        qw = float(ss[7])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, qx, qy, qz, qw, abs_scale


class EurocGroundTruth(GroundTruth):
    kReadTumConversion = False

    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.EUROC):
        super().__init__(path, name, associations, start_frame_id, type)
        self.scale = kScaleEuroc
        if EurocGroundTruth.kReadTumConversion:
            # NOTE: Use the script io/generate_euroc_groundtruths_as_tum.sh to generate these groundtruth files
            self.filename = path + "/" + name + "/mav0/state_groundtruth_estimate0/data.tum"
        else:
            # Use the original Euroc groundtruth file
            self.filename = path + "/" + name + "/mav0/state_groundtruth_estimate0/data.csv"

        base_path = os.path.dirname(self.filename)
        print("base_path: ", base_path)

        if not os.path.isfile(self.filename):
            if EurocGroundTruth.kReadTumConversion:
                error_message = f"ERROR: Groundtruth file not found: {self.filename}. Use the script io/generate_euroc_groundtruths_as_tum.sh to generate these groundtruth files!"
            else:
                error_message = f"ERROR: Groundtruth file not found: {self.filename}. Please, check how you deployed the files and if the code is consistent with this!"
            Printer.red(error_message)
            sys.exit(error_message)

        if EurocGroundTruth.kReadTumConversion:
            with open(self.filename) as f:
                self.data = f.readlines()
                self.data = [line.strip().split() for line in self.data]
                self.data = np.ascontiguousarray(self.data)
        else:
            self.data = self.read_gt_data_state(self.filename)
            self.data = np.ascontiguousarray(self.data)

        if len(self.data) > 0:
            self.found = True
            print("Processing Euroc groundtruth of lenght: ", len(self.data))

        if len(self.data) == 0:
            sys.exit(
                f"ERROR while reading groundtruth file {self.filename}: please, check how you deployed the files and if the code is consistent with this!"
            )

        self.image_left_csv_path = path + "/" + name + "/mav0/cam0/data.csv"
        self.image_data = self.read_image_data(self.image_left_csv_path)

        associations_file = base_path + "/associations.json"
        if True:  # not os.path.exists(associations_file):
            # Printer.orange('Computing groundtruth associations (one-time operation)...')
            self.association_matches = self.associate(self.image_data, self.data)
            # save associations
            with open(associations_file, "w") as f:
                json.dump(self.association_matches, f)
        else:
            with open(associations_file, "r") as f:
                data = json.load(f)
                self.association_matches = {int(k): v for k, v in data.items()}

    def read_image_data(self, csv_file):
        timestamps_and_filenames = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            for row in reader:
                timestamp_ns = int(row[0])
                filename = row[1]
                timestamp_s = float(timestamp_ns) * 1e-9
                timestamps_and_filenames.append((timestamp_s, filename))
        return timestamps_and_filenames

    def read_gt_data_state(self, csv_file):
        data = []
        with open(csv_file, "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                parts = line.strip().split(",")
                timestamp_ns = int(parts[0])
                position = np.ascontiguousarray([float(parts[1]), float(parts[2]), float(parts[3])])
                quaternion = np.ascontiguousarray(
                    [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                )  # qw, qx, qy, qz
                # velocity = np.ascontiguousarray([float(parts[8]), float(parts[9]), float(parts[10])])
                # accel_bias = np.ascontiguousarray([float(parts[11]), float(parts[12]), float(parts[13])])
                # gyro_bias = np.ascontiguousarray([float(parts[14]), float(parts[15]), float(parts[16])])
                # we expect the quaternion in the form [qx, qy, qz, qw] as in the TUM format
                data.append(
                    (
                        float(timestamp_ns) * 1e-9,
                        position[0],
                        position[1],
                        position[2],
                        quaternion[1],
                        quaternion[2],
                        quaternion[3],
                        quaternion[0],
                    )
                )
        return data

    def read_gt_data_pose(self, csv_file):
        data = []
        with open(csv_file, "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                parts = line.strip().split(",")
                timestamp_ns = int(parts[0])
                position = np.ascontiguousarray([float(parts[1]), float(parts[2]), float(parts[3])])
                quaternion = np.ascontiguousarray(
                    [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                )
                # we expect the quaternion in the form [qx, qy, qz, qw] as in the TUM format
                data.append(
                    (
                        float(timestamp_ns) * 1e-9,
                        position[0],
                        position[1],
                        position[2],
                        quaternion[1],
                        quaternion[2],
                        quaternion[3],
                        quaternion[0],
                    )
                )
        return data

    def getDataLine(self, frame_id):
        return self.data[self.association_matches[frame_id][0]]

    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[1])
            y_prev = self.scale * float(ss[2])
            z_prev = self.scale * float(ss[3])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        # print(f'ss[{frame_id}]: {ss}')
        timestamp = float(ss[0])
        x = self.scale * float(ss[1])
        y = self.scale * float(ss[2])
        z = self.scale * float(ss[3])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        # print(f'abs_scale: {abs_scale}')
        # from https://www.researchgate.net/profile/Michael-Burri/publication/291954561_The_EuRoC_micro_aerial_vehicle_datasets/links/56af0c6008ae19a38516937c/The-EuRoC-micro-aerial-vehicle-datasets.pdf
        return timestamp, x, y, z, abs_scale

    # return timestamp, x,y,z, qx,qy,qz,qw, scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[1])
            y_prev = self.scale * float(ss[2])
            z_prev = self.scale * float(ss[3])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = float(ss[0])
        x = self.scale * float(ss[1])
        y = self.scale * float(ss[2])
        z = self.scale * float(ss[3])
        qx = float(ss[4])
        qy = float(ss[5])
        qz = float(ss[6])
        qw = float(ss[7])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, qx, qy, qz, qw, abs_scale


class ReplicaGroundTruth(GroundTruth):
    kScale = 1.0  # 6553.5

    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.TUM):
        super().__init__(path, name, associations, start_frame_id, type)
        from pyslam.io.dataset import ReplicaDataset

        self.Ts = ReplicaDataset.Ts
        self.scale = kScaleTum
        self.filename = (
            path + "/" + name + "/" + "traj.txt"
        )  # N.B.: this may depend on how you deployed the groundtruth files

        if not os.path.exists(self.filename):
            error_message = (
                f"ERROR: [ReplicaGroundTruth] Groundtruth file not found: {self.filename}!"
            )
            Printer.red(error_message)
            sys.exit(error_message)

        base_path = os.path.dirname(self.filename)
        print("base_path: ", base_path)

        self.poses_ = []
        self.timestamps_ = []
        with open(self.filename) as f:
            self.data = f.readlines()
            for i, line in enumerate(self.data):
                pose = np.ascontiguousarray(list(map(float, line.split()))).reshape(4, 4)
                # pose = np.linalg.inv(pose)
                self.poses_.append(pose)
                timestamp = i * self.Ts
                self.timestamps_.append(timestamp)
            self.data = np.ascontiguousarray(self.data)
        print(f"Number of poses: {len(self.poses_)}")
        if self.data is None:
            sys.exit("ERROR while reading groundtruth file!")

    def getDataLine(self, frame_id):
        return self.data[frame_id]

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            pos_prev = self.poses_[frame_id - 1][0:3, 3] * ReplicaGroundTruth.kScale
            x_prev, y_prev, z_prev = pos_prev[0], pos_prev[1], pos_prev[2]
        except:
            x_prev, y_prev, z_prev = None, None, None
        timestamp = self.timestamps_[frame_id]
        pos = self.poses_[frame_id][0:3, 3] * ReplicaGroundTruth.kScale
        x, y, z = pos[0], pos[1], pos[2]
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, abs_scale

    # return timestamp, x,y,z, qx,qy,qz,qw, scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            pos_prev = self.poses_[frame_id - 1][0:3, 3] * ReplicaGroundTruth.kScale
            x_prev, y_prev, z_prev = pos_prev[0], pos_prev[1], pos_prev[2]
        except:
            x_prev, y_prev, z_prev = None, None, None
        timestamp = self.timestamps_[frame_id]
        pos = self.poses_[frame_id][0:3, 3] * ReplicaGroundTruth.kScale
        x, y, z = pos[0], pos[1], pos[2]
        R = self.poses_[frame_id][0:3, 0:3]
        q = rotmat2qvec(R)  # [qx, qy, qz, qw]
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, q[0], q[1], q[2], q[3], abs_scale


# Read the ground truth from files containining [x,y,z, qx, qy, qz, qw, scale] lines
class TartanairGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.KITTI):
        super().__init__(path, name, associations, start_frame_id, type)
        from pyslam.io.dataset import TartanairDataset

        self.Ts = TartanairDataset.Ts
        self.scale = kScaleSimple
        if path is not None:
            self.filename = path + "/" + name + "/" + "pose_left.txt"
        else:
            self.filename = name

        if not os.path.isfile(self.filename):
            error_message = (
                f"ERROR: [SimpleGroundTruth] Groundtruth file not found: {self.filename}!"
            )
            Printer.red(error_message)
            sys.exit(error_message)

        with open(self.filename) as f:
            self.data = f.readlines()
            self.data = np.ascontiguousarray(self.data)
            self.found = True
        if self.data is None:
            sys.exit(
                "ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!"
            )

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[0])
            y_prev = self.scale * float(ss[1])
            z_prev = self.scale * float(ss[2])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = frame_id * self.Ts
        x = self.scale * float(ss[0])
        y = self.scale * float(ss[1])
        z = self.scale * float(ss[2])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        # print(f'reading frame {frame_id}, timestamp: {timestamp:.15f}, x: {x:.15f}, y: {y:.15f}, z: {z:.15f}, scale: {abs_scale:.15f}')
        return timestamp, x, y, z, abs_scale

    # return timestamp, x,y,z, qx,qy,qz,qw, scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            ss = self.getDataLine(frame_id - 1)
            x_prev = self.scale * float(ss[0])
            y_prev = self.scale * float(ss[1])
            z_prev = self.scale * float(ss[2])
        except:
            x_prev, y_prev, z_prev = None, None, None
        ss = self.getDataLine(frame_id)
        timestamp = frame_id * self.Ts
        x = self.scale * float(ss[0])
        y = self.scale * float(ss[1])
        z = self.scale * float(ss[2])
        qx = float(ss[3])
        qy = float(ss[4])
        qz = float(ss[5])
        qw = float(ss[6])
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, qx, qy, qz, qw, abs_scale


class ScannetGroundTruth(GroundTruth):
    def __init__(
        self, path, name, associations=None, start_frame_id=0, type=GroundTruthType.SCANNET
    ):
        super().__init__(path, name, associations, start_frame_id, type)
        from pyslam.io.dataset import ScannetDataset

        self.Ts = ScannetDataset.Ts
        self.scale = kScaleScannet
        self.poses_path = path + "/scans/" + name + "/pose/"

        if not os.path.exists(self.poses_path) or os.listdir(self.poses_path) == []:
            error_message = (
                f"ERROR: [ScannetGroundTruth] Groundtruth directory not found: {self.poses_path}!"
            )
            Printer.red(error_message)
            sys.exit(error_message)

        base_path = os.path.dirname(self.poses_path)
        print("base_path: ", base_path)
        self.poses_ = []
        self.timestamps_ = []
        self.data = []
        # List poses files in directory.
        # To get the correct order we need to sort according to the integer name before .txt
        file_names = sorted(os.listdir(self.poses_path), key=lambda x: int(x.split(".")[0]))
        for file_name in file_names:
            pose = np.loadtxt(self.poses_path + file_name)
            self.poses_.append(pose)
            self.timestamps_.append(int(file_name.split(".")[0]) * self.Ts)
            self.data.append(pose)
        self.data = np.ascontiguousarray(self.data)
        print(f"Number of poses: {len(self.poses_)}")
        if self.data is None:
            sys.exit("ERROR while reading groundtruth file!")

    def getDataLine(self, frame_id):
        return super().getDataLine(frame_id)

    # return timestamp,x,y,z,scale
    def getTimestampPositionAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            pos_prev = self.poses_[frame_id - 1][0:3, 3] * self.scale
            x_prev, y_prev, z_prev = pos_prev[0], pos_prev[1], pos_prev[2]
        except:
            x_prev, y_prev, z_prev = None, None, None
        timestamp = self.timestamps_[frame_id]
        pos = self.poses_[frame_id][0:3, 3] * self.scale
        x, y, z = pos[0], pos[1], pos[2]
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, abs_scale

    # return timestamp, x,y,z, qx,qy,qz,qw, scale
    def getTimestampPoseAndAbsoluteScale(self, frame_id):
        frame_id += self.start_frame_id
        try:
            pos_prev = self.poses_[frame_id - 1]
            x_prev, y_prev, z_prev = pos_prev[0:3, 3] * self.scale
        except:
            x_prev, y_prev, z_prev = None, None, None
        timestamp = self.timestamps_[frame_id]
        pos = self.poses_[frame_id]
        x, y, z = pos[0:3, 3] * self.scale
        qx, qy, qz, qw = pos[0:4, 0]
        if x_prev is None:
            abs_scale = 1
        else:
            abs_scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
        return timestamp, x, y, z, qx, qy, qz, qw, abs_scale
