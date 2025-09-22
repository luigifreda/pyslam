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

from enum import Enum
from pyslam.utilities.utils_serialization import SerializableEnum, register_class, SerializationJSON

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.config import Config  # Only imported when type checking, not at runtime


@register_class
class DatasetType(SerializableEnum):
    NONE = 1
    KITTI = 2
    TUM = 3
    EUROC = 4
    REPLICA = 5
    TARTANAIR = 6
    VIDEO = 7
    FOLDER = 8  # generic folder of pics
    ROS1BAG = 9
    ROS2BAG = 10
    LIVE = 11
    SCANNET = 12


@register_class
class DatasetEnvironmentType(SerializableEnum):
    INDOOR = 1
    OUTDOOR = 2


@register_class
class SensorType(SerializableEnum):
    MONOCULAR = 0
    STEREO = 1
    RGBD = 2


# A minimal dataset config for serialization
class MinimalDatasetConfig:
    def __init__(
        self,
        config: "Config" = None,
        dataset_settings=None,
        cam_settings=None,
        cam_stereo_settings=None,
    ):
        self.dataset_settings = config.dataset_settings if config is not None else dataset_settings
        self.cam_settings = config.cam_settings if config is not None else cam_settings
        self.cam_stereo_settings = (
            config.cam_stereo_settings if config is not None else cam_stereo_settings
        )

    def to_json(self):
        return {
            "dataset_settings": SerializationJSON.serialize(self.dataset_settings),
            "cam_settings": SerializationJSON.serialize(self.cam_settings),
            "cam_stereo_settings": SerializationJSON.serialize(self.cam_stereo_settings),
        }

    @staticmethod
    def from_json(json_str):
        return MinimalDatasetConfig(
            dataset_settings=SerializationJSON.deserialize(json_str["dataset_settings"]),
            cam_settings=SerializationJSON.deserialize(json_str["cam_settings"]),
            cam_stereo_settings=SerializationJSON.deserialize(json_str["cam_stereo_settings"]),
        )
