import sys

import pyslam.config as config

config.cfg.set_lib("gaussian_splatting")

import argparse
import numpy as np
import cv2
import math
import time

import platform

from pyslam.config import Config

from pyslam.slam.slam import Slam, SlamState
from pyslam.slam import PinholeCamera
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.io.ground_truth import groundtruth_factory

from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.system import getchar
from pyslam.utilities.logging import Printer
from pyslam.utilities.geometry import inv_T

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.depth_estimation.depth_estimator_factory import (
    depth_estimator_factory,
    DepthEstimatorType,
)

from pyslam.config_parameters import Parameters

import torch
import torch.multiprocessing as mp

from monogs.gaussian_splatting_manager import GaussianSplattingManager
from monogs.utils.config_utils import load_config

import signal
import os


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kResultsFolder = kRootFolder + "/results/gaussian_splatting"


gsm = None


# intercept the SIGINT signal
def signal_handler(signal, frame):
    print("You pressed Ctrl+C!")
    if gsm is not None:
        gsm.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Load a SLAM state and build a 3D Gaussian splatting model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="../../results/slam_state",
        help="path where we have saved the system state",
    )
    parser.add_argument(
        "--config", type=str, default="../../thirdparty/monogs/configs/rgbd/tum/fr3_office.yaml"
    )
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = Config()

    groundtruth = groundtruth_factory(config.dataset_settings)
    gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()

    cam = PinholeCamera(config)
    feature_tracker_config = FeatureTrackerConfigs.TEST

    # create SLAM object
    slam = Slam(cam, feature_tracker_config)
    time.sleep(1)  # to show initial messages

    slam.load_system_state(args.path)
    viewer_scale = (
        slam.viewer_scale() if slam.viewer_scale() > 0 else 0.1
    )  # 0.1 is the default viewer scale
    print(f"viewer_scale: {viewer_scale}")

    print(f"sensor_type: {slam.sensor_type}")

    viewer3D = Viewer3D(viewer_scale)
    if viewer3D is not None:
        viewer3D.draw_slam_map(slam)

    gs_config = load_config(args.config)
    gsm = GaussianSplattingManager(
        gs_config,
        monocular=(slam.sensor_type == SensorType.MONOCULAR),
        live_mode=False,
        use_gui=True,
        eval_rendering=False,
        use_dataset=False,
        save_results=True,
        save_dir=kResultsFolder,
    )
    gsm.start()

    map = slam.map
    num_map_keyframes = map.num_keyframes()
    keyframes = map.get_keyframes()

    depth_estimator = None
    # DEPTH_ANYTHING_V2, DEPTH_PRO, DEPTH_RAFT_STEREO, DEPTH_SGBM, etc.
    depth_estimator_type = DepthEstimatorType.DEPTH_RAFT_STEREO
    max_depth = 20
    if slam.sensor_type == SensorType.STEREO:
        depth_estimator = depth_estimator_factory(
            depth_estimator_type=depth_estimator_type,
            max_depth=max_depth,
            dataset_env_type=slam.environment_type,
            camera=slam.camera,
        )

    print(f"inserting #keyframes: {num_map_keyframes}")
    if num_map_keyframes > 0:

        pose0_gt = gt_poses[keyframes[0].img_id]
        inv_pose0_gt = inv_T(pose0_gt)
        pose0 = keyframes[0].Twc()
        inv_pose0 = inv_T(pose0)

        for kf in keyframes:

            # print('-----------------------------------')
            # print(f'inserting keyframe: {kf.id}, img_id: {kf.img_id}. img shape: {kf.img.shape}, depth shape: {kf.depth_img.shape} type: {kf.depth_img.dtype}')

            # if the depth image is not available, we need to compute it
            if kf.depth_img is None:
                if depth_estimator is not None:
                    print
                    kf.depth_img, pts3d_prediction = depth_estimator.infer(kf.img, kf.img_right)
                else:
                    print("depth_img is None")
                    continue

            img = cv2.cvtColor(kf.img, cv2.COLOR_BGR2RGB)

            gt_pose = inv_pose0_gt @ gt_poses[kf.img_id]  # Twc
            # inv_gt_pose = inv_T(gt_pose) # Tcw
            inv_gt_pose = None

            # pose = inv_pose0 @ kf.Twc()
            pose = kf.Twc()
            inv_pose = inv_T(pose)  # Tcw

            gsm.add_keyframe(kf.id, kf.camera, img, kf.depth_img, inv_pose, inv_gt_pose)
            time.sleep(1)

    while gsm.frontend.is_running:
        time.sleep(1)  # wait for the frontend to finish
    gsm.stop()

    slam.quit()
