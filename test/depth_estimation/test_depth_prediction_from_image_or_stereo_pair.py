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
import argparse
import cv2
import numpy as np

from pyslam.config import Config

from pyslam.utilities.logging import Printer
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run depth prediction from dataset stream (default) or from a single image/stereo pair."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single left image. If set, the script runs on image input instead of dataset stream.",
    )
    parser.add_argument(
        "--image-right",
        "--image_right",
        dest="image_right",
        type=str,
        default=None,
        help="Optional right image path for stereo depth prediction (used with --image).",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=("indoor", "outdoor"),
        default="indoor",
        help="Environment type used in image mode to set depth estimator defaults.",
    )
    parser.add_argument(
        "--viewer-scale",
        type=float,
        default=None,
        help="Optional 3D viewer scale override.",
    )
    parser.add_argument(
        "--disable-viewer",
        action="store_true",
        help="Disable 3D viewer and only show 2D image windows.",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Flip input image(s) vertically before depth inference (image mode only).",
    )
    parser.add_argument(
        "--balance-stereo-light",
        action="store_true",
        help="Match right-image global light to left image before stereo inference.",
    )
    return parser.parse_args()


def read_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image at path: {image_path}")
    return img


def match_global_light(src_bgr, ref_bgr, eps=1e-6):
    src_y = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    ref_y = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    gain = (ref_y.mean() + eps) / (src_y.mean() + eps)
    out = src_bgr.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def match_global_mean_std(src_bgr, ref_bgr, eps=1e-6):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    src_y = src[:, :, 0]
    ref_y = ref[:, :, 0]

    src_mean, src_std = src_y.mean(), src_y.std()
    ref_mean, ref_std = ref_y.mean(), ref_y.std()

    out_y = (src_y - src_mean) * (ref_std / (src_std + eps)) + ref_mean
    src[:, :, 0] = np.clip(out_y, 0, 255)

    out = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out


def apply_clahe_bgr(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb[:, :, 0] = clahe.apply(y)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


if __name__ == "__main__":
    args = parse_args()

    config = Config()
    use_image_mode = args.image is not None
    dataset = dataset_factory(config) if not use_image_mode else None

    cam = PinholeCamera(config)

    # tracker_config = FeatureTrackerConfigs.ORB2
    # tracker_config["num_features"] = 2000
    # print("tracker_config: ", tracker_config)
    # feature_tracker = feature_tracker_factory(**tracker_config)
    # # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared.
    # FeatureTrackerShared.set_feature_tracker(feature_tracker)

    # Select your depth estimator (see the file depth_estimator_configs.py).
    depth_estimator_type = DepthEstimatorType.DEPTH_MAST3R
    min_depth = 0
    if use_image_mode:
        dataset_env_type = (
            DatasetEnvironmentType.OUTDOOR
            if args.environment == "outdoor"
            else DatasetEnvironmentType.INDOOR
        )
    else:
        dataset_env_type = dataset.environmentType()

    max_depth = 50 if dataset_env_type == DatasetEnvironmentType.OUTDOOR else 10
    precision = torch.float16
    depth_estimator = depth_estimator_factory(
        depth_estimator_type=depth_estimator_type,
        min_depth=min_depth,
        max_depth=max_depth,
        dataset_env_type=dataset_env_type,
        precision=precision,
        camera=cam,
    )

    Printer.green(f"Depth estimator: {depth_estimator_type.name}")

    viewer_scale = (
        args.viewer_scale
        if args.viewer_scale is not None
        else (dataset.scale_viewer_3d if dataset is not None else 1.0)
    )
    viewer3D = None if args.disable_viewer else Viewer3D(scale=viewer_scale)

    show_directly_point_cloud_if_available = True

    key_cv = None
    is_paused = False  # pause/resume on GUI

    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    while True:
        if viewer3D is not None and viewer3D.is_closed():
            break

        timestamp, img, img_right = None, None, None

        if not is_paused:

            if use_image_mode:
                img = read_image(args.image)
                img_right = read_image(args.image_right) if args.image_right else None
                if args.flip_vertical:
                    img = cv2.flip(img, 0)
                    if img_right is not None:
                        img_right = cv2.flip(img_right, 0)
            else:
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
                if use_image_mode:
                    print(f"processing input image: {args.image}")
                else:
                    print(f"processing img {img_id}")

                if args.balance_stereo_light and img_right is not None:
                    print("balancing stereo light")
                    # img_right = match_global_light(img_right, img)
                    # img_right = match_global_mean_std(img_right, img)
                    img_right = apply_clahe_bgr(img_right)
                    img = apply_clahe_bgr(img)

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

            if not use_image_mode:
                img_id += 1

        else:
            time.sleep(0.1)

        # get keys
        key_cv = cv2.waitKey(1) & 0xFF

        if viewer3D is not None:
            is_paused = viewer3D.is_paused()

        if key_cv == ord("q"):
            break

        # In image mode we process once and then keep windows/viewer open until 'q'.
        if use_image_mode:
            is_paused = True

    if viewer3D is not None:
        viewer3D.quit()
