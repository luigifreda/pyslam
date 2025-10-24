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

import os
import sys
import cv2
import numpy as np

from pyslam.config import Config

from pyslam.utilities.system import Printer
from pyslam.utilities.depth import (
    depth2pointcloud,
    img_from_depth,
    filter_shadow_points,
    PointCloud,
)

from pyslam.slam import PinholeCamera
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared

from pyslam.depth_estimation.depth_estimator_factory import (
    depth_estimator_factory,
    DepthEstimatorType,
)

from pyslam.io.dataset_types import DatasetType, SensorType, DatasetEnvironmentType
from pyslam.io.dataset_factory import dataset_factory

from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.config_parameters import Parameters

import torch
import time

from pyslam.viz.viewer3D import Viewer3D


# online loop closure detection by using DBoW3
if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config)

    cam = PinholeCamera(config)

    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = 2000
    print("tracker_config: ", tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)
    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared.
    FeatureTrackerShared.set_feature_tracker(feature_tracker)

    # Select your depth estimator (see the file depth_estimator_configs.py).
    depth_estimator_type = DepthEstimatorType.DEPTH_PRO
    min_depth = 0
    max_depth = 50 if dataset.environmentType() == DatasetEnvironmentType.OUTDOOR else 10
    precision = torch.float16
    depth_estimator = depth_estimator_factory(
        depth_estimator_type=depth_estimator_type,
        min_depth=min_depth,
        max_depth=max_depth,
        dataset_env_type=dataset.environmentType(),
        precision=precision,
        camera=cam,
    )

    Printer.green(f"Depth estimator: {depth_estimator_type.name}")

    viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)

    show_directly_point_cloud_if_available = True

    key_cv = None
    is_paused = False  # pause/resume on GUI

    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    while not viewer3D.is_closed():

        timestamp, img, img_right = None, None, None

        if not is_paused:

            if dataset.is_ok:
                timestamp = dataset.getTimestamp()  # get current timestamp
                img = dataset.getImageColor(img_id)
                img_right = (
                    dataset.getImageColorRight(img_id)
                    if dataset.sensor_type == SensorType.STEREO
                    else None
                )

            if img is not None:
                print("----------------------------------------")
                print(f"processing img {img_id}")

                start_time = time.time()

                depth_prediction, pts3d_prediction = depth_estimator.infer(img, img_right)

                print(f"inference time: {time.time() - start_time}")

                cv2.imshow("color image", img)
                if img_right is not None:
                    cv2.imshow("color image right", img_right)

                if pts3d_prediction is not None and show_directly_point_cloud_if_available:

                    # We have predicted a 3D point cloud.
                    print(f"got directly point cloud: {pts3d_prediction.points.shape}")

                    depth_img = img_from_depth(depth_prediction, img_min=0, img_max=max_depth)
                    cv2.imshow("depth prediction", depth_img)

                    # Draw directly the predicted points cloud
                    if viewer3D is not None:
                        viewer3D.draw_dense_geometry(point_cloud=pts3d_prediction)

                else:

                    # We use the depth to build a 3D point cloud and visualize it.

                    # Filter depth
                    filter_depth = True  # do you want to filter the depth?
                    if filter_depth:
                        depth_filtered = filter_shadow_points(depth_prediction, delta_depth=None)
                    else:
                        depth_filtered = depth_prediction

                    # Visualize depth map
                    depth_img = img_from_depth(depth_prediction, img_min=0, img_max=max_depth)
                    depth_filtered_img = img_from_depth(
                        depth_filtered, img_min=0, img_max=max_depth
                    )

                    # Visualize 3D point cloud
                    if viewer3D is not None:
                        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        point_cloud = depth2pointcloud(
                            depth_filtered, image_rgb, cam.fx, cam.fy, cam.cx, cam.cy, max_depth
                        )
                        viewer3D.draw_dense_geometry(point_cloud=point_cloud)

                    cv2.imshow("depth prediction", depth_img)
                    cv2.imshow("depth filtered", depth_filtered_img)

            else:
                time.sleep(0.1)

            img_id += 1

        else:
            time.sleep(0.1)

        # get keys
        key_cv = cv2.waitKey(1) & 0xFF

        if viewer3D is not None:
            is_paused = viewer3D.is_paused()

        if key_cv == ord("q"):
            break

    if viewer3D is not None:
        viewer3D.quit()
