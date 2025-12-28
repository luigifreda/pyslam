import sys

import pyslam.config as config


import numpy as np

# import gtsam

import unittest
from unittest import TestCase

from pyslam.config_parameters import Parameters
from pyslam.utilities.system import Printer

Parameters.USE_CPP_CORE = False
Printer.info("Using Python core - see the tests in pyslam/slam/cpp/tests_py for C++ core tests")

from pyslam.slam import Frame, PinholeCamera, MapPoint, optimizer_gtsam, optimizer_g2o
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.utilities.geometry import rotation_matrix_from_yaw_pitch_roll, poseRt, inv_T


class FakeFeatureManager:
    def __init__(self, num_keypoints):
        self.inv_level_sigmas2 = [1.0] * num_keypoints
        self.level_sigmas = [1.0] * num_keypoints


class FakeMapPoint:
    def __init__(self, pt):
        self._pt = pt.ravel()

    def pt(self):
        return self._pt


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
        camera.K = self.K
        camera.Kinv = self.Kinv

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
        FeatureTrackerShared.feature_manager = FakeFeatureManager(num_keypoints)

        # Random depths in left camera (between min_depth and max_depth)
        depths = np.random.uniform(self.min_depth, self.max_depth, size=num_keypoints)

        noisy_kpsu = frame.kpsu + np.random.normal(0, pixel_noise_std_dev, size=frame.kpsu.shape)
        # points_3d_l = camera.unproject_points_3d(frame.kpsu, depths)
        points_3d_l = camera.unproject_points_3d(noisy_kpsu, depths)

        # Here we assume camera pose is gt_Twc
        points_3d_w = (self.gt_Twc[:3, :3] @ points_3d_l.T + self.gt_Twc[:3, 3].reshape(3, 1)).T
        print(f"points_3d_w: {points_3d_w.shape}")
        frame.points = [FakeMapPoint(points_3d_w[i]) for i in range(num_keypoints)]

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

        frame.kps_ur = uv_r[:, 0].ravel()
        frame.kps_ur[num_stereo_keypoints + 1 : -1] = -1

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

        frame.outliers = {i: False for i in range(num_keypoints)}  # Initially no outliers

        return frame

    def compute_reprojection_error(self, frame):
        frame_Tcw = frame.Tcw()
        Rcw = frame_Tcw[:3, :3]
        tcw = frame_Tcw[:3, 3]
        points_3d_w = np.array([p.pt() for p in frame.points]).reshape(-1, 3)
        points_3d_l = (Rcw @ points_3d_w.T + tcw.reshape(3, 1)).T
        uv_l, depths_l = frame.camera.project(points_3d_l)
        error = np.linalg.norm(frame.kpsu - uv_l, axis=1)
        RMSE = np.sqrt(np.mean(error**2))
        return RMSE

    def test_pose_optimization_convergence(self):

        # Generate the random frame
        frame = self.generate_random_frame()

        # Compute reprojection error
        RMSE = self.compute_reprojection_error(frame)
        print(f"RMSE before: {RMSE}")

        if True:
            # mean_squared_error, success, num_valid_points = optimizer_gtsam.pose_optimization(frame, verbose=True, rounds=10)
            mean_squared_error, success, num_valid_points = optimizer_g2o.pose_optimization(
                frame, verbose=True, rounds=10
            )

        # Get the estimated pose_cw
        pose_estimated = frame.pose()

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
