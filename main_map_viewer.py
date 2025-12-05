#!/usr/bin/env -S python3 -O
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

import argparse
import numpy as np
import cv2
import math
import time
from datetime import datetime

from pyslam.config import Config

from pyslam.slam.slam import Slam, SlamState, SlamMode
from pyslam.slam import PinholeCamera
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType
from pyslam.io.ground_truth import GroundTruth

from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.system import getchar, Printer

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.config_parameters import Parameters


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.map_point import MapPoint
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.slam import Slam


datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":

    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=config.system_state_folder_path,
        help="path where we have saved the system state",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        type=str,
        default=config.system_state_folder_path,
        help="Path to save the system state if needed",
    )
    args = parser.parse_args()

    camera = PinholeCamera()
    feature_tracker_config = FeatureTrackerConfigs.TEST

    # create SLAM object
    slam = Slam(camera, feature_tracker_config, slam_mode=SlamMode.MAP_BROWSER)
    # load the system state
    slam.load_system_state(args.path)
    camera = slam.camera  # update the camera after having reloaded the state
    groundtruth = GroundTruth.load(args.path)  # load ground truth from saved state
    viewer_scale = (
        slam.viewer_scale() if slam.viewer_scale() > 0 else 0.1
    )  # 0.1 is the default viewer scale
    print(f"viewer_scale: {viewer_scale}")

    viewer3D = Viewer3D(viewer_scale)
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
        viewer3D.set_gt_trajectory(
            gt_traj3d, gt_timestamps, align_with_scale=slam.sensor_type == SensorType.MONOCULAR
        )

    is_map_save = False  # save map on GUI
    is_bundle_adjust = False  # bundle adjust on GUI

    while not viewer3D.is_closed():
        time.sleep(0.1)

        # 3D display (map display)
        viewer3D.draw_slam_map(slam)

        is_map_save = viewer3D.is_map_save() and is_map_save == False
        is_bundle_adjust = viewer3D.is_bundle_adjust() and is_bundle_adjust == False

        if is_map_save:
            output_path = args.output_path + "_" + datetime_string
            slam.save_system_state(output_path)

        if is_bundle_adjust:
            slam.bundle_adjust()

    viewer3D.quit()  # NOTE: first quit the viewer3D then the slam
    slam.quit()
