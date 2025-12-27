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
import os

import numpy as np
import unittest
from unittest import TestCase

import pyslam.config as config
from pyslam.config_parameters import Parameters

USE_CPP = True
Parameters.USE_CPP_CORE = USE_CPP

from pyslam.slam.cpp import cpp_module, python_module, CPP_AVAILABLE

if not CPP_AVAILABLE:
    print("❌ cpp_module imported successfully but C++ core is not available")
    sys.exit(1)
else:
    print("✅ cpp_module imported successfully")

if USE_CPP:
    Frame = cpp_module.Frame
    KeyFrame = cpp_module.KeyFrame
    Map = cpp_module.Map
    MapPoint = cpp_module.MapPoint
    Sim3Pose = cpp_module.Sim3Pose
    Camera = cpp_module.Camera
    PinholeCamera = cpp_module.PinholeCamera
    optimizer_g2o = cpp_module.optimizer_g2o
    optimizer_gtsam = cpp_module.optimizer_gtsam
    print("Using C++ module")
else:
    Frame = python_module.Frame
    KeyFrame = python_module.KeyFrame
    Map = python_module.Map
    MapPoint = python_module.MapPoint
    Sim3Pose = python_module.Sim3Pose
    Camera = python_module.Camera
    PinholeCamera = python_module.PinholeCamera
    optimizer_g2o = python_module.optimizer_g2o
    optimizer_gtsam = python_module.optimizer_gtsam
    print("Using Python module")

from pyslam.utilities.geometry import (
    rotation_matrix_from_yaw_pitch_roll,
    poseRt,
    inv_T,
)
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.local_features.feature_tracker import feature_tracker_factory


class TestPoseOptimizerConvergence(TestCase):
    fx = 517.306408
    fy = 516.469215
    cx = 318.643040
    cy = 255.313989
    bf = 40
    b = bf / fx
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Kinv = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]])
    width = 640
    height = 480
    min_depth, max_depth = 0.5, 10.0
    Trl = np.array([[1, 0, 0, -b], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # left w.r.t. right

    def generate_random_frame(self):

        pixel_noise_std_dev = 0.5

        camera = PinholeCamera(config=None)
        camera.fx = self.fx
        camera.fy = self.fy
        camera.cx = self.cx
        camera.cy = self.cy
        camera.width = self.width
        camera.height = self.height
        camera.bf = self.bf
        camera.b = self.b
        camera.fps = 30
        camera.set_intrinsic_matrices()
        camera.K = self.K
        camera.Kinv = self.Kinv

        print(f"camera: {camera.to_json()}")

        # Set a seed for reproducibility
        np.random.seed(0)  # You can change the seed value to any integer

        yaw_deg, pitch_deg, roll_deg = np.random.uniform(-60, 60, size=3)
        tx, ty, tz = np.random.uniform(-1, 1, size=3)
        self.gt_Twc = poseRt(
            rotation_matrix_from_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg),
            np.array([tx, ty, tz]),
        )
        self.gt_Tcw = inv_T(self.gt_Twc)

        frame = Frame(camera=camera, img=None)

        # Number of keypoints
        num_keypoints = 100
        num_stereo_keypoints = 40

        assert num_keypoints > num_stereo_keypoints

        # Random 2D keypoints (u, v)
        frame.kpsu = np.random.rand(num_keypoints, 2) * np.array([camera.width, camera.height])
        frame.octaves = np.full(num_keypoints, 0).astype(np.int32)  # all at level 0

        tracker_config = FeatureTrackerConfigs.ORB2
        tracker_config["num_features"] = 1000
        feature_tracker = feature_tracker_factory(**tracker_config)
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

        # Random depths in left camera (between min_depth and max_depth)
        depths = np.random.uniform(self.min_depth, self.max_depth, size=num_keypoints)

        noisy_kpsu = frame.kpsu + np.random.normal(0, pixel_noise_std_dev, size=frame.kpsu.shape)
        # points_3d_l = camera.unproject_points_3d(frame.kpsu, depths)
        points_3d_l = camera.unproject_points_3d(noisy_kpsu, depths)

        # Here we assume camera pose is gt_Twc
        points_3d_w = (self.gt_Twc[:3, :3] @ points_3d_l.T + self.gt_Twc[:3, 3].reshape(3, 1)).T
        print(f"points_3d_w: {points_3d_w.shape}")
        frame.points = [
            MapPoint(points_3d_w[i], np.array([0, 0, 0], dtype=np.uint8))
            for i in range(num_keypoints)
        ]

        if False:
            uv_l, depths_l = camera.project(
                (self.gt_Tcw[:3, :3] @ points_3d_w.T + self.gt_Tcw[:3, 3].reshape(3, 1)).T
            )
            check_uv_l = np.linalg.norm(frame.kpsu - uv_l, axis=1)
            check_depths_l = depths - depths_l
            assert np.all(check_uv_l < 1e-5)
            assert np.all(check_depths_l < 1e-5)

        points_3d_r = (self.Trl[:3, :3] @ points_3d_l.T + self.Trl[:3, 3].reshape(3, 1)).T
        uv_r, depths_r = camera.project(points_3d_r)

        if False:
            check_vr = (
                frame.kpsu[:, 1].ravel() - uv_r[:, 1].ravel()
            )  # vr must be the same in a rectified pair
            check_depths_r = depths - depths_r
            assert np.all(check_vr < 1e-5)
            assert np.all(check_depths_r < 1e-5)

        kps_ur = uv_r[:, 0].ravel()
        kps_ur[num_stereo_keypoints:] = -1.0
        frame.kps_ur = kps_ur
        # print(f"frame.kps_ur: {frame.kps_ur}")

        # Set gt pose and initial frame pose (ground truth pose is np.eye(4), we will use it directly)
        # print(f'self.gt_Rcw: {self.gt_Tcw[:3, :3].flatten()}, self.gt_tcw: {self.gt_Tcw[:3, 3].flatten()}')
        initial_Tcw = self.gt_Tcw.copy()
        initial_Tcw[:3, :3] = initial_Tcw[:3, :3] @ rotation_matrix_from_yaw_pitch_roll(
            -10, 10, -10
        )  # degs, introduce a shift that shoud be removed by the optimizer
        initial_Tcw[:3, 3] = initial_Tcw[:3, 3] + np.array(
            [0.3, 0.3, 0.3]
        )  # introduce a shift that shoud be removed by the optimizer
        frame.update_pose(initial_Tcw)

        frame.outliers = [False for i in range(num_keypoints)]  # Initially no outliers

        return frame

    def compute_reprojection_error(self, frame):
        for idx, p in enumerate(frame.points):
            if p is None:
                print(f"frame.points[{idx}] is None")
            else:
                _pt = p.pt()
        points_3d_w = np.array([p.pt() for p in frame.points if p is not None]).reshape(-1, 3)
        # print(f"points_3d_w: {points_3d_w}")
        print(f"points_3d_w shape: {points_3d_w.shape}")
        frame_Tcw = frame.Tcw()
        points_3d_l = (frame_Tcw[:3, :3] @ points_3d_w.T + frame_Tcw[:3, 3].reshape(3, 1)).T
        # print(f"points_3d_l: {points_3d_l}")
        print(f"points_3d_l shape: {points_3d_l.shape}")
        camera = frame.camera
        print(f"camera: {camera}")
        print(f"frame.camera: {camera.to_json()}")
        uv_l, depths_l = frame.camera.project(points_3d_l)
        print(f"uv_l: {uv_l.shape}")
        print(f"depths_l: {depths_l.shape}")
        error = np.linalg.norm(frame.kpsu - uv_l, axis=1)
        RMSE = np.sqrt(np.mean(error**2))
        return RMSE

    def test_pose_optimization_convergence(self):

        # Generate the random frame
        frame = self.generate_random_frame()

        # Compute reprojection error
        RMSE = self.compute_reprojection_error(frame)
        print(f"RMSE before: {RMSE}")

        # frame.ensure_contiguous_arrays()
        # return

        use_optimizer_gtsam = False
        print(f"use_optimizer_gtsam: {use_optimizer_gtsam}")
        if use_optimizer_gtsam:
            pose_optimization = optimizer_gtsam.pose_optimization
        else:
            pose_optimization = optimizer_g2o.pose_optimization

        if True:
            # mean_squared_error, success, num_valid_points = optimizer_gtsam.pose_optimization(frame, verbose=True, rounds=10)
            mean_squared_error, success, num_valid_points = pose_optimization(
                frame, verbose=True, rounds=10
            )

        # Get the estimated pose_cw
        pose_estimated = frame.pose()
        print(f"pose_estimated: {pose_estimated}")

        Rcw = pose_estimated[:3, :3]
        tcw = pose_estimated[:3, 3]
        gt_Rcw = self.gt_Tcw[:3, :3]
        gt_tcw = self.gt_Tcw[:3, 3]
        R_diff = np.linalg.norm(Rcw @ gt_Rcw.T - np.eye(3))
        t_diff = np.linalg.norm(tcw - gt_tcw)
        print(f"R diff: {R_diff}, t diff: {t_diff}")

        # Compute reprojection error
        RMSE = self.compute_reprojection_error(frame)
        print(f"RMSE: {RMSE}")

        # Compare the estimated pose to the identity pose (np.eye(4))
        self.assertTrue(R_diff < 1e-3)
        self.assertTrue(t_diff < 1e-3)
        self.assertTrue(success)
        self.assertGreater(mean_squared_error, 0)
        # self.assertEqual(num_valid_points, len(frame.points))


if __name__ == "__main__":
    unittest.main()
