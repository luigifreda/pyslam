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

import sys
import numpy as np
from enum import Enum
import cv2
import os
import glob
import time
import csv
import re

from multiprocessing import Process, Queue, Value
from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from
from pyslam.utilities.serialization import SerializableEnum, register_class, SerializationJSON
from pyslam.utilities.strings import levenshtein_distance

import ujson as json

from .dataset_types import (
    DatasetType,
    SensorType,
    DatasetEnvironmentType,
    MinimalDatasetConfig,
    get_sensor_type,
)
from .dataset import (
    Dataset,
    FolderDataset,
    FolderDatasetParallel,
    KittiDataset,
    ScannetDataset,
    TumDataset,
    IclNuimDataset,
    EurocDataset,
    ReplicaDataset,
    TartanairDataset,
    VideoDataset,
    LiveDataset,
    SevenScenesDataset,
    NeuralRGBDDataset,
    RoverDataset,
)
from .mcap_dataset import McapDataset


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.config import Config  # Only imported when type checking, not at runtime


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kSettingsFolder = kRootFolder + "/settings"


def dataset_factory(config: "Config") -> Dataset:
    dataset_settings: dict = config.dataset_settings if config.dataset_settings is not None else {}
    type = DatasetType.NONE
    associations = None  # name of the file with the associations
    timestamps = None
    path = None
    is_color = None  # used for kitti datasets

    type = dataset_settings["type"].lower()
    name = dataset_settings["name"]

    sensor_type = SensorType.MONOCULAR
    if "sensor_type" in dataset_settings:
        sensor_type = get_sensor_type(dataset_settings["sensor_type"])
    Printer.green(f"dataset_factory - sensor_type: {sensor_type.name}")

    environment_type = DatasetEnvironmentType.OUTDOOR
    if "environment_type" in dataset_settings:
        environment_type = dataset_settings["environment_type"].lower()
        if environment_type == "outdoor":
            environment_type = DatasetEnvironmentType.OUTDOOR
        elif environment_type == "indoor":
            environment_type = DatasetEnvironmentType.INDOOR
        else:
            environment_type = DatasetEnvironmentType.OUTDOOR  # default value

    path = dataset_settings["base_path"]
    path = os.path.expanduser(path)

    start_frame_id = 0
    if "start_frame_id" in dataset_settings:
        Printer.green(f'dataset_factory - start_frame_id: {dataset_settings["start_frame_id"]}')
        start_frame_id = int(dataset_settings["start_frame_id"])

    if "associations" in dataset_settings:
        associations = dataset_settings["associations"]
    if "timestamps" in dataset_settings:
        timestamps = dataset_settings["timestamps"]
    if "is_color" in dataset_settings:
        is_color = dataset_settings["is_color"]

    dataset = None

    if type == "kitti":
        dataset = KittiDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.KITTI
        )
        dataset.set_is_color(is_color)
    if type == "tum":
        dataset = TumDataset(path, name, sensor_type, associations, start_frame_id, DatasetType.TUM)
    if type == "icl_nuim":
        dataset = IclNuimDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.ICL_NUIM
        )
    if type == "euroc":
        dataset = EurocDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.EUROC, config
        )
    if type == "replica":
        dataset = ReplicaDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.REPLICA
        )
    if type == "tartanair":
        dataset = TartanairDataset(
            path,
            name,
            sensor_type,
            associations,
            start_frame_id,
            DatasetType.TARTANAIR,
            environment_type,
        )
    if type == "video":
        dataset = VideoDataset(
            path, name, sensor_type, associations, timestamps, start_frame_id, DatasetType.VIDEO
        )
    if type == "folder":
        fps = 10  # a default value
        if "fps" in dataset_settings:
            fps = int(dataset_settings["fps"])
        dataset = FolderDataset(
            path,
            name,
            sensor_type,
            fps,
            associations,
            timestamps,
            start_frame_id,
            DatasetType.FOLDER,
        )
    if type == "live":
        dataset = LiveDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.LIVE
        )
    if type == "ros1bag":
        from pyslam.io.ros1bag_dataset import Ros1bagDataset

        fps = 10  # a default value
        if "fps" in dataset_settings:
            fps = int(dataset_settings["fps"])
        dataset = Ros1bagDataset(
            path,
            name,
            sensor_type,
            associations,
            start_frame_id,
            DatasetType.ROS1BAG,
            environment_type,
            fps,
            config,
        )
    if type == "ros2bag":
        from pyslam.io.ros2bag_dataset import Ros2bagDataset

        fps = 10  # fps default value
        rate = 1.0  # rate multiplier default value
        if "fps" in dataset_settings:
            fps = int(dataset_settings["fps"])
        if "rate" in dataset_settings:
            rate = float(dataset_settings["rate"])
        # dataset = Ros2bagDataset(path, name, sensor_type, associations, start_frame_id, DatasetType.ROS1BAG, environment_type, fps, rate, config)
        dataset = Ros2bagDataset(
            path,
            name,
            sensor_type,
            associations,
            start_frame_id,
            DatasetType.ROS1BAG,
            environment_type,
            fps,
            config,
        )
    if type == "mcap":
        if "fps" in dataset_settings:
            fps = int(dataset_settings["fps"])
        dataset = McapDataset(
            path,
            name,
            sensor_type,
            associations,
            start_frame_id,
            DatasetType.MCAP,
            environment_type,
            fps=fps,
            config=config,
        )
    if type == "scannet":
        dataset = ScannetDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.SCANNET, config
        )
    if type == "seven_scenes" or type == "7scenes":
        dataset = SevenScenesDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.SEVEN_SCENES, config
        )
    if type == "neural_rgbd" or type == "neuralrgbd":
        dataset = NeuralRGBDDataset(
            path, name, sensor_type, associations, start_frame_id, DatasetType.NEURAL_RGBD, config
        )
    if type == "rover":
        camera_name = dataset_settings["camera_name"]
        dataset = RoverDataset(
            path,
            name,
            camera_name,
            sensor_type,
            associations,
            start_frame_id,
            type=DatasetType.ROVER,
            environment_type=environment_type,
        )

    dataset.minimal_config = MinimalDatasetConfig(config=config)

    return dataset
