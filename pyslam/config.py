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

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

import os
import yaml
import numpy as np
import math
import ujson as json

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import locally_configure_qt_environment
from pyslam.io.dataset_types import get_sensor_type, SensorType
from pyslam.config_parameters import Parameters


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.join(kScriptFolder, "..")  # root folder of the repository
kDefaultConfigPath = os.path.join(kRootFolder, "config.yaml")
kDefaultConfigLibsPath = os.path.join(kRootFolder, "config_libs.yaml")


# Class for reading libs, dataset, system, and camera settings (from config.yaml) from yaml files.
# Input:
#   config_path: path to config yaml file where dataset, system, and camera settings are stored
#   config_libs_path: path to config libs yaml file where lib paths are stored
class Config:
    def __init__(
        self,
        config_path=kDefaultConfigPath,
        config_libs_path=kDefaultConfigLibsPath,
        root_folder=kRootFolder,
    ):
        self.root_folder = root_folder
        self.config_path = config_path  # path to config.yaml: dataset, system, and camera settings
        self.config_libs_path = config_libs_path  # path to config_libs.yaml: lib paths
        with open(self.config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(self.config_libs_path, "r") as f:
            self.config_libs = yaml.load(f, Loader=yaml.FullLoader)
        self.cam_settings: dict | None = None
        self.system_settings: dict | None = None
        self.dataset_settings: dict | None = None
        self.dataset_type: str | None = None
        self.sensor_type = None
        self.system_state_settings: dict | None = None
        self.system_state_folder_path: str | None = None
        self.system_state_load = False
        self.ros_settings: dict = {}
        self.feature_tracker_config: dict | None = None
        self.loop_detection_config: dict | None = None
        self.semantic_mapping_config: dict | None = None
        self.global_parameters: dict | None = None

        self.trajectory_saving_settings: dict = {}

        self.start_frame_id = 0

        # locally_configure_qt_environment()

        self.set_core_lib_paths()
        self.read_lib_paths()

        self.get_dataset_settings()
        self.get_general_system_settings()
        self.get_system_state_settings()
        self.get_trajectory_saving_settings()
        self.get_and_set_global_parameters()

    # read core lib paths from config.yaml and set sys paths
    def set_core_lib_paths(self):
        self.core_lib_paths = self.config_libs["CORE_LIB_PATHS"]
        for path in self.core_lib_paths:
            ext_path = self.root_folder + "/" + self.core_lib_paths[path]
            # print( "importing path: ", ext_path )
            sys.path.append(ext_path)

    # read lib paths from config.yaml
    def read_lib_paths(self):
        self.lib_paths = self.config_libs["LIB_PATHS"]

    # set sys path of lib
    def set_lib(self, lib_name, prepend=False, verbose=False):
        ext_path = None
        if lib_name in self.lib_paths:
            if verbose:
                print(f"[Config] setting lib {lib_name} with paths: {self.lib_paths[lib_name]}")
            lib_paths = [e.strip() for e in self.lib_paths[lib_name].split(",")]
            # print('setting lib paths:',lib_paths)
            for lib_path in lib_paths:
                ext_path = self.root_folder + "/" + lib_path
                # print( "importing path: ", ext_path )
                if not prepend:
                    sys.path.append(ext_path)
                else:
                    sys.path.insert(0, ext_path)
                if verbose:
                    print("[Config] adding path: ", ext_path)
        else:
            print("[Config] cannot set lib: ", lib_name)
            if verbose:
                print(f"[Config] available lib paths: {self.lib_paths.keys()}")
        return ext_path

    def remove_lib(self, lib_name, verbose=False):
        if lib_name in self.lib_paths:
            lib_paths = [e.strip() for e in self.lib_paths[lib_name].split(",")]
            for lib_path in lib_paths:
                ext_path = self.root_folder + "/" + lib_path
                if verbose:
                    print("[Config] removing path: ", ext_path)
                sys.path.remove(ext_path)

    # get dataset settings
    def get_dataset_settings(self):
        self.dataset_type = self.config["DATASET"]["type"]
        self.dataset_settings = self.config[self.dataset_type]
        sensor_type_str = self.dataset_settings["sensor_type"].lower()
        self.sensor_type = get_sensor_type(sensor_type_str)
        self.dataset_path = self.dataset_settings["base_path"]
        self.dataset_settings["base_path"] = os.path.join(self.root_folder, self.dataset_path)
        # print('dataset_settings: ', self.dataset_settings)
        str_dataset_settings_type = self.dataset_settings["type"].lower()
        if (
            str_dataset_settings_type == "ros1bag"
            or str_dataset_settings_type == "ros2bag"
            or str_dataset_settings_type == "mcap"
        ):
            self.get_ros_bag_settings()

    # get general system settings
    def get_general_system_settings(self):
        self.system_settings = None
        input_settings_path = self.config[self.dataset_type]["settings"]
        if not os.path.isabs(input_settings_path):
            input_settings_path = os.path.join(self.root_folder, input_settings_path)
        self.general_settings_filepath = input_settings_path
        # If sensor is stereo and stereo settings exist, use stereo settings
        if (
            self.sensor_type == SensorType.STEREO
            and "settings_stereo" in self.config[self.dataset_type]
        ):
            input_stereo_settings_path = self.config[self.dataset_type]["settings_stereo"]
            if not os.path.isabs(input_stereo_settings_path):
                input_stereo_settings_path = os.path.join(
                    self.root_folder, input_stereo_settings_path
                )
            self.general_settings_filepath = input_stereo_settings_path
            Printer.orange("[Config] Using stereo settings file: " + self.general_settings_filepath)
            print("------------------------------------")
        if self.general_settings_filepath is not None:
            with open(self.general_settings_filepath, "r") as stream:
                try:
                    self.system_settings = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)
        self.cam_settings = self.system_settings

    def get_system_state_settings(self):
        self.system_state_settings = self.config["SYSTEM_STATE"]
        self.system_state_load = self.system_state_settings["load_state"]
        self.system_state_folder_path = (
            self.root_folder + "/" + self.system_state_settings["folder_path"]
        )
        folder_path_exists = os.path.exists(self.system_state_folder_path)
        folder_path_is_not_empty = (
            os.path.getsize(self.system_state_folder_path) > 0 if folder_path_exists else False
        )
        if self.system_state_load and not (folder_path_exists and folder_path_is_not_empty):
            Printer.red(
                "[Config] System state folder does not exist or is empty: "
                + self.system_state_folder_path
            )
            self.system_state_load = False

    # get trajectory save settings
    def get_trajectory_saving_settings(self):
        self.trajectory_saving_settings = self.config["SAVE_TRAJECTORY"]

    def get_trajectory_saving_paths(self, datatime_string=None):
        dt_string = "" if datatime_string is None else "_" + datatime_string
        trajectory_saving_base_path = self.trajectory_saving_settings["output_folder"] + dt_string
        trajectory_online_file_path = (
            trajectory_saving_base_path
            + "/"
            + self.trajectory_saving_settings["basename"]
            + "_online.txt"
        )  # online estimates (depend only on the past estimates)
        trajectory_final_file_path = (
            trajectory_saving_base_path
            + "/"
            + self.trajectory_saving_settings["basename"]
            + "_final.txt"
        )  # final estimates (depend on the past and future estimates)
        return trajectory_online_file_path, trajectory_final_file_path, trajectory_saving_base_path

    def get_and_set_global_parameters(self):
        # for changing the global parameters default values from the config file
        self.global_parameters = self.config["GLOBAL_PARAMETERS"]
        if self.global_parameters is not None:
            Printer.orange("[Config] Setting global parameters: ", self.global_parameters)
            from pyslam.config_parameters import Parameters, set_from_dict

            set_from_dict(Parameters, self.global_parameters)

    def get_ros_bag_settings(self):
        self.ros_settings = self.config[self.dataset_type]["ros_settings"]
        self.ros_settings["bag_path"] = os.path.join(
            self.dataset_settings["base_path"], self.dataset_settings["name"]
        )
        # print(f'ROS settings: {self.ros_settings}')

    # calibration matrix
    @property
    def K(self):
        if not hasattr(self, "_K"):
            fx = self.cam_settings["Camera.fx"]
            cx = self.cam_settings["Camera.cx"]
            fy = self.cam_settings["Camera.fy"]
            cy = self.cam_settings["Camera.cy"]
            self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return self._K

    # inverse of calibration matrix
    @property
    def Kinv(self):
        if not hasattr(self, "_Kinv"):
            fx = self.cam_settings["Camera.fx"]
            cx = self.cam_settings["Camera.cx"]
            fy = self.cam_settings["Camera.fy"]
            cy = self.cam_settings["Camera.cy"]
            self._Kinv = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]])
        return self._Kinv

    # distortion coefficients
    @property
    def DistCoef(self):
        if not hasattr(self, "_DistCoef"):
            k1 = self.cam_settings.get("Camera.k1", 0)
            k2 = self.cam_settings.get("Camera.k2", 0)
            p1 = self.cam_settings.get("Camera.p1", 0)
            p2 = self.cam_settings.get("Camera.p2", 0)
            k3 = 0
            if "Camera.k3" in self.cam_settings:
                k3 = self.cam_settings.get("Camera.k3", 0)
            self._DistCoef = np.array([k1, k2, p1, p2, k3])
            if self.sensor_type == SensorType.STEREO:
                self._DistCoef = np.array([0, 0, 0, 0, 0])
                Printer.orange(
                    "[Config] WARNING: Using stereo camera, images are automatically rectified, and DistCoef is set to [0,0,0,0,0]"
                )
        return self._DistCoef

    # baseline times fx
    @property
    def bf(self):
        if not hasattr(self, "_bf"):
            self._bf = self.cam_settings["Camera.bf"]
        return self._bf

    # camera width
    @property
    def width(self):
        if not hasattr(self, "_width"):
            self._width = self.cam_settings["Camera.width"]
        return self._width

    # camera height
    @property
    def height(self):
        if not hasattr(self, "_height"):
            self._height = self.cam_settings["Camera.height"]
        return self._height

    # camera fps
    @property
    def fps(self):
        if not hasattr(self, "_fps"):
            self._fps = self.cam_settings["Camera.fps"]
        return self._fps

    # depth factor
    @property
    def depth_factor(self):
        if not hasattr(self, "_depth_factor"):
            if "DepthMapFactor" in self.cam_settings:
                self._depth_factor = self.cam_settings["DepthMapFactor"]
            else:
                self._depth_factor = 1.0
        return self._depth_factor

    @property
    def depth_threshold(self):
        if not hasattr(self, "_depth_threshold"):
            if "ThDepth" in self.cam_settings:
                self._depth_threshold = self.cam_settings["ThDepth"]
            else:
                self._depth_threshold = float("inf")
        return self._depth_threshold

    @property
    def num_features_to_extract(self):
        if not hasattr(self, "_num_features_to_extract"):
            if "FeatureTrackerConfig.nFeatures" in self.system_settings:
                self._num_features_to_extract = self.system_settings[
                    "FeatureTrackerConfig.nFeatures"
                ]
            else:
                self._num_features_to_extract = 0
        return self._num_features_to_extract

    @property
    def feature_tracker_config_name(self):
        if not hasattr(self, "_feature_tracker_config_name"):
            if "FeatureTrackerConfig.name" in self.system_settings:
                self._feature_tracker_config_name = self.system_settings[
                    "FeatureTrackerConfig.name"
                ]
            else:
                self._feature_tracker_config_name = None
        return self._feature_tracker_config_name

    @property
    def loop_detection_config_name(self):
        if not hasattr(self, "_loop_detection_config_name"):
            if "LoopDetectionConfig.name" in self.system_settings:
                self._loop_detection_config_name = self.system_settings["LoopDetectionConfig.name"]
            else:
                self._loop_detection_config_name = None
        return self._loop_detection_config_name

    @property
    def semantic_mapping_config_name(self):
        if not hasattr(self, "_semantic_mapping_config_name"):
            if "SemanticMappingConfig.name" in self.system_settings:
                self._semantic_mapping_config_name = self.system_settings[
                    "SemanticMappingConfig.name"
                ]
            else:
                self._semantic_mapping_config_name = None
        return self._semantic_mapping_config_name

    @property
    def far_points_threshold(self):
        if not hasattr(self, "_far_points_threshold"):
            if "Matching.farPointsThreshold" in self.system_settings:
                self._far_points_threshold = self.system_settings["Matching.farPointsThreshold"]
            else:
                self._far_points_threshold = None
        return self._far_points_threshold

    @property
    def use_fov_centers_based_kf_generation(self):
        if not hasattr(self, "_use_fov_centers_based_kf_generation"):
            self._use_fov_centers_based_kf_generation = False
            if "KeyFrame.useFovCentersBasedGeneration" in self.system_settings:
                self._use_fov_centers_based_kf_generation = bool(
                    self.system_settings["KeyFrame.useFovCentersBasedGeneration"]
                )
            else:
                self._use_fov_centers_based_kf_generation = (
                    Parameters.kUseFovCentersBasedKfGeneration
                )
        return self._use_fov_centers_based_kf_generation

    @property
    def max_fov_centers_distance(self):
        if not hasattr(self, "_max_fov_centers_distance"):
            self._max_fov_centers_distance = -1
            if "KeyFrame.maxFovCentersDistance" in self.system_settings:
                self._max_fov_centers_distance = self.system_settings[
                    "KeyFrame.maxFovCentersDistance"
                ]
            else:
                self._max_fov_centers_distance = Parameters.kMaxFovCentersDistanceForKfGeneration
        return self._max_fov_centers_distance

    # stereo settings
    @property
    def cam_stereo_settings(self):
        if not hasattr(self, "_cam_stereo_settings"):
            self._cam_stereo_settings = None
            left, right = {}, {}
            if "LEFT.D" in self.cam_settings:
                left_D = self.cam_settings["LEFT.D"]
                left_D = np.array(left_D["data"], dtype=float).reshape(
                    left_D["rows"], left_D["cols"]
                )
                left["D"] = left_D
            if "LEFT.K" in self.cam_settings:
                left_K = self.cam_settings["LEFT.K"]
                left_K = np.array(left_K["data"], dtype=float).reshape(
                    left_K["rows"], left_K["cols"]
                )
                left["K"] = left_K
            if "LEFT.R" in self.cam_settings:
                left_R = self.cam_settings["LEFT.R"]
                left_R = np.array(left_R["data"], dtype=float).reshape(
                    left_R["rows"], left_R["cols"]
                )
                left["R"] = left_R
            if "LEFT.P" in self.cam_settings:
                left_P = self.cam_settings["LEFT.P"]
                left_P = np.array(left_P["data"], dtype=float).reshape(
                    left_P["rows"], left_P["cols"]
                )
                left["P"] = left_P

            if "RIGHT.D" in self.cam_settings:
                right_D = self.cam_settings["RIGHT.D"]
                right_D = np.array(right_D["data"], dtype=float).reshape(
                    right_D["rows"], right_D["cols"]
                )
                right["D"] = right_D
            if "RIGHT.K" in self.cam_settings:
                right_K = self.cam_settings["RIGHT.K"]
                right_K = np.array(right_K["data"], dtype=float).reshape(
                    right_K["rows"], right_K["cols"]
                )
                right["K"] = right_K
            if "RIGHT.R" in self.cam_settings:
                right_R = self.cam_settings["RIGHT.R"]
                right_R = np.array(right_R["data"], dtype=float).reshape(
                    right_R["rows"], right_R["cols"]
                )
                right["R"] = right_R
            if "RIGHT.P" in self.cam_settings:
                right_P = self.cam_settings["RIGHT.P"]
                right_P = np.array(right_P["data"], dtype=float).reshape(
                    right_P["rows"], right_P["cols"]
                )
                right["P"] = right_P

            if len(left) > 0 and len(right) > 0:
                self._cam_stereo_settings = {"left": left, "right": right}
        # print(f'[config] stereo settings: {self._cam_stereo_settings}')
        return self._cam_stereo_settings

    def from_json(self, json_str):
        """
        Update the config from a JSON string or dict.
        The payload may contain any of the following optional fields; missing fields
        leave current values unchanged:
        - cam_settings: dict
        - cam_stereo_settings: dict
        - dataset_settings: dict
        - system_settings: dict
        - ros_settings: dict
        - feature_tracker_config: dict
        - loop_detection_config: dict
        - semantic_mapping_config: dict
        """
        # Accept a JSON string or already-parsed dict; normalize to dict
        data = json_str
        if isinstance(json_str, str):
            data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError(f"from_json expects a JSON string or dict, got {type(data)}")

        # Update top-level settings if present; keep existing if missing
        cam_settings = data.get("cam_settings")
        if cam_settings is not None:
            self.cam_settings = cam_settings

        # Stereo settings are optional; if absent, let property recompute from cam_settings
        cam_stereo_settings = data.get("cam_stereo_settings")
        if cam_stereo_settings is not None:
            self._cam_stereo_settings = cam_stereo_settings
        else:
            if hasattr(self, "_cam_stereo_settings"):
                delattr(self, "_cam_stereo_settings")

        dataset_settings = data.get("dataset_settings")
        if dataset_settings is not None:
            self.dataset_settings = dataset_settings

        # sensor_type may be provided via dataset_settings; guard for absence
        sensor_type = None
        if isinstance(self.dataset_settings, dict):
            sensor_type = self.dataset_settings.get("sensor_type")
        if sensor_type is not None:
            self.sensor_type = str(sensor_type).lower()

        system_settings = data.get("system_settings")
        if system_settings is not None:
            self.system_settings = system_settings

        self.ros_settings = data.get("ros_settings", self.ros_settings or {})
        self.feature_tracker_config = data.get(
            "feature_tracker_config", self.feature_tracker_config
        )
        self.loop_detection_config = data.get("loop_detection_config", self.loop_detection_config)
        self.semantic_mapping_config = data.get(
            "semantic_mapping_config", self.semantic_mapping_config
        )

        # Invalidate cached derived properties to ensure consistency with new settings
        for attr in (
            "_K",
            "_Kinv",
            "_DistCoef",
            "_bf",
            "_width",
            "_height",
            "_fps",
            "_depth_factor",
            "_depth_threshold",
            "_num_features_to_extract",
            "_feature_tracker_config_name",
            "_loop_detection_config_name",
            "_semantic_mapping_config_name",
            "_far_points_threshold",
            "_use_fov_centers_based_kf_generation",
            "_max_fov_centers_distance",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def to_json(self):
        return {
            "cam_settings": self.cam_settings,
            "cam_stereo_settings": self.cam_stereo_settings,
            "dataset_settings": self.dataset_settings,
            "system_settings": self.system_settings,
            "ros_settings": self.ros_settings,
            "feature_tracker_config": self.feature_tracker_config,
            "loop_detection_config": self.loop_detection_config,
            "semantic_mapping_config": self.semantic_mapping_config,
        }


if __name__ != "__main__":
    # We automatically read and set lib paths when this file is called via 'import'
    cfg = Config()
