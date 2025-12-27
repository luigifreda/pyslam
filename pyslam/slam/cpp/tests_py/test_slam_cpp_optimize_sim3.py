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

from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.utilities.geometry import (
    rotation_matrix_from_yaw_pitch_roll,
    poseRt,
    inv_T,
)
from pyslam.utilities.synthetic_data import (
    generate_random_points_2d,
    backproject_points,
    project_points,
)


def check_solution(
    uv1, points_3d_w1, uv2, points_3d_w2, K1, Rc1w, tc1w, K2, Rc2w, tc2w, scalec1c2, Rc1c2, tc1c2
):
    eps = 1e-9

    # transform points from world coordinates to camera coordinates
    points_3d_c1 = (Rc1w @ np.array(points_3d_w1).T + tc1w.reshape(3, 1)).T
    points_3d_c2 = (Rc2w @ np.array(points_3d_w2).T + tc2w.reshape(3, 1)).T

    aligned_points_c1 = (scalec1c2 * Rc1c2 @ np.array(points_3d_c2).T + tc1c2.reshape(3, 1)).T
    average_alignment_error = np.mean(np.linalg.norm(aligned_points_c1 - points_3d_c1, axis=1))
    print(f"[check_solution] Average 3d alignment error: {average_alignment_error}")

    # project points 2 on 1
    project1 = (K1 @ (scalec1c2 * Rc1c2 @ np.array(points_3d_c2).T + tc1c2.reshape(3, 1))).T
    project1_z = np.where(np.abs(project1[:, 2]) < eps, eps, project1[:, 2])
    proj_err1 = np.mean(np.linalg.norm(project1[:, :2] / project1_z[:, np.newaxis] - uv1, axis=1))
    print(f"[check_solution] Average projection error 1: {proj_err1}")
    # project points 1 on 2
    Rc2c1 = Rc1c2.T
    tc2c1 = -Rc1c2.T @ tc1c2 / scalec1c2
    scalec2c1 = 1.0 / scalec1c2
    project2 = (K2 @ (scalec2c1 * Rc2c1 @ np.array(points_3d_c1).T + tc2c1.reshape(3, 1))).T
    project2_z = np.where(np.abs(project2[:, 2]) < eps, eps, project2[:, 2])
    proj_err2 = np.mean(np.linalg.norm(project2[:, :2] / project2_z[:, np.newaxis] - uv2, axis=1))
    print(f"[check_solution] Average projection error 2: {proj_err2}")
    return average_alignment_error, proj_err1, proj_err2


class TestOptimizeSim3(TestCase):
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
    th2 = 10.0  # =Parameters.kLoopClosingTh2

    def init_camera(self):
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

        self.camera = camera
        self.K1 = camera.K
        self.K2 = camera.K

        # Set a seed for reproducibility
        np.random.seed(0)  # You can change the seed value to any integer

    def create_perfect_world(self):

        num_points = 100
        self.num_points = num_points
        print(f"num_points: {num_points}")

        # camera data
        camera = self.camera
        K1 = camera.K
        K2 = camera.K

        # keyframe1 data
        yaw1_deg, pitch1_deg, roll1_deg = np.random.uniform(-180, 180, size=3)
        t1x, t1y, t1z = np.random.uniform(-1, 1, size=3)

        Rwc1 = rotation_matrix_from_yaw_pitch_roll(
            yaw1_deg, pitch1_deg, roll1_deg
        )  # Rotation matrix for KF1
        twc1 = np.array([t1x, t1y, t1z]).reshape(3, 1)  # Translation vector for KF1
        Rc1w = Rwc1.T
        tc1w = -Rwc1.T @ twc1
        self.gt_Twc1 = poseRt(Rwc1, twc1.ravel())
        self.gt_Tc1w = poseRt(Rc1w, tc1w.ravel())

        # keyframe2 data
        dyaw2_deg, dpitch2_deg, droll2_deg = np.random.uniform(-5, 5, size=3)
        dt2x, dt2y, dt2z = np.random.uniform(-0.1, 0.1, size=3)

        Rwc2 = Rwc1 @ rotation_matrix_from_yaw_pitch_roll(
            dyaw2_deg, dpitch2_deg, droll2_deg
        )  # Rotation matrix for KF2
        twc2 = twc1 + np.array([dt2x, dt2y, dt2z]).reshape(3, 1)  # Translation vector for KF2
        Rc2w = Rwc2.T
        tc2w = -Rwc2.T @ twc2
        self.gt_Twc2 = poseRt(Rwc2, twc2.ravel())
        self.gt_Tc2w = poseRt(Rc2w, tc2w.ravel())

        Rc2c1 = Rc2w @ Rc1w.T
        tc2c1 = tc2w - (Rc2c1 @ tc1w)

        Rc1c2 = Rc2c1.T
        tc1c2 = -Rc1c2 @ tc2c1

        # print(f'Rc1c2: {Rc1c2}')
        # print(f'tc1c2: {tc1c2}')

        # generate random points in camera1 image and back project them with random depths
        points_2d_c1 = generate_random_points_2d(camera.width, camera.height, num_points)
        points_3d_c1, depths_c1 = backproject_points(K1, points_2d_c1, 1.0, 10.0)

        # check which points are visible in camera 2
        points_3d_c2 = (Rc2c1 @ points_3d_c1.T + tc2c1.reshape((3, 1))).T
        points_2d_c2, _, mask = project_points(K2, None, points_3d_c2, camera.width, camera.height)

        # remove points that are not visible in camera 2
        mask = mask.ravel()
        points_2d_c1 = points_2d_c1[mask, :]
        points_3d_c1 = points_3d_c1[mask, :]
        points_2d_c2 = points_2d_c2[mask, :]
        points_3d_c2 = points_3d_c2[mask, :]

        points_3d_w1 = (Rwc1 @ points_3d_c1.T + twc1.reshape(3, 1)).T
        points_3d_w2 = points_3d_w1.copy()
        print(f"visible 3D points shape: {points_3d_w1.shape}")
        # print(f'points 3D: {points_3d_w1}')

        # save initial values as our first guess (without actually knowing the motion)
        self.Rc1c2_initial = Rc1c2.copy().reshape(3, 3)
        self.tc1c2_initial = tc1c2.copy().reshape(3, 1)

        self.points_2d_c1 = points_2d_c1
        self.points_2d_c2 = points_2d_c2
        self.points_3d_c1 = points_3d_c1
        self.points_3d_c2 = points_3d_c2
        self.points_3d_w1 = points_3d_w1
        self.points_3d_w2 = points_3d_w2

        self.Rc1w = Rc1w
        self.tc1w = tc1w
        self.Rwc1 = Rwc1
        self.twc1 = twc1

        self.Rc2w = Rc2w
        self.tc2w = tc2w

        self.Rc1c2 = Rc1c2
        self.tc1c2 = tc1c2

        print(f"Checking solution before motion of camera 2...")
        check_solution(
            points_2d_c1,
            points_3d_w1,
            points_2d_c2,
            points_3d_w2,
            K1,
            Rc1w,
            tc1w,
            K2,
            Rc2w,
            tc2w,
            1.0,
            Rc1c2,
            tc1c2,
        )

        points_3d_c1_check = (self.Rc1c2 @ self.points_3d_c2.T + self.tc1c2.reshape(3, 1)).T
        assert self.points_3d_c1.shape == points_3d_c1_check.shape
        err_3d_c1 = np.linalg.norm(self.points_3d_c1 - points_3d_c1_check)
        # err_3d_c1 = np.mean(np.linalg.norm(self.points_3d_c1 - points_3d_c1_check, axis=1))
        print(f"3D c2 to 3D c1 error: {err_3d_c1}")

    def apply_motion(self):
        # RELATIVE MOTION:
        # Now we have perfectly matched points.
        # Let's simulate a sim3 motion on camera 2 relative to camera 1:
        # Tc1c2_d = Tc1c2 * gt_delta_T

        delta_angle_degs = 1  # degs, between c1 and c2
        delta_t_meters = 0.01  # meters, between c1 and c2
        delta_s = 0.1  # scale, between c1 and c2

        # grount-truth DELTA sim3
        gt_delta_R = rotation_matrix_from_yaw_pitch_roll(
            np.random.uniform(-delta_angle_degs, delta_angle_degs),
            np.random.uniform(-delta_angle_degs, delta_angle_degs),
            np.random.uniform(-delta_angle_degs, delta_angle_degs),
        )
        gt_delta_t = np.random.uniform(-delta_t_meters, delta_t_meters, size=3).reshape(3, 1)
        gt_delta_s = np.random.uniform(1.0 - delta_s, 1.0 + delta_s)

        # Tc1c2_d = Tc1c2 * gt_delta_T
        Rc1c2_d = self.Rc1c2 @ gt_delta_R
        tc1c2_d = self.Rc1c2 @ gt_delta_t + self.tc1c2

        # At this point we have identified a GT sim3:
        # gt_Sc1c2 = (Rc1c2_d, tc1c2_d, gt_delta_s)

        self.Rc1c2 = Rc1c2_d.copy()
        self.tc1c2 = tc1c2_d.copy()

        self.gt_Rc1c2 = Rc1c2_d
        self.gt_tc1c2 = tc1c2_d
        self.gt_sc1c2 = gt_delta_s

        self.gt_Sc1c2 = Sim3Pose(Rc1c2_d, tc1c2_d, gt_delta_s)
        self.gt_Sc2c1 = self.gt_Sc1c2.inverse()

        self.gt_Rc2c1 = self.gt_Sc2c1.R
        self.gt_tc2c1 = self.gt_Sc2c1.t
        self.gt_sc2c1 = self.gt_Sc2c1.s

        # print(f'gt Sc1c2: {self.gt_Sc1c2}')

        # update camera 2 pose after relative motion (starting from camera 1 pose)
        Rwc2 = self.Rwc1 @ self.Rc1c2
        twc2 = self.Rwc1 @ self.tc1c2 + self.twc1
        Rc2w = Rwc2.T
        tc2w = -Rc2w @ twc2
        Tc2w = poseRt(Rc2w, tc2w.ravel())
        self.gt_Tc2w = Tc2w

        self.Rc2w = Rc2w
        self.tc2w = tc2w
        self.Rwc2 = Rwc2
        self.twc2 = twc2
        self.Tc2w = Tc2w

        # update points in camera frame 2
        points_3d_c2_d = self.gt_Sc2c1.map_points(self.points_3d_c1)
        self.points_3d_c2 = points_3d_c2_d
        self.points_2d_c2, _, mask2 = project_points(
            self.K2, None, self.points_3d_c2, self.camera.width, self.camera.height
        )
        print(f"points_3d_c2_d shape: {points_3d_c2_d.shape}")
        print(f"Rwc2 shape: {Rwc2.shape}")
        print(f"twc2 shape: {twc2.shape}")
        self.points_3d_w2 = (Rwc2 @ points_3d_c2_d.T + twc2.reshape(3, 1)).T

        self.points_2d_c1 = self.points_2d_c1[mask2, :]
        self.points_2d_c2 = self.points_2d_c2[mask2, :]
        self.points_3d_c1 = self.points_3d_c1[mask2, :]
        self.points_3d_c2 = self.points_3d_c2[mask2, :]
        self.points_3d_w1 = self.points_3d_w1[mask2, :]
        self.points_3d_w2 = self.points_3d_w2[mask2, :]

        self.fix_scale = abs(gt_delta_s - 1.0) < 1e-6
        print(f"fix_scale: {self.fix_scale}")

        if False:
            Rc1c2_check = self.Rc1w @ self.Rc2w.T
            err_Rc1c2 = np.linalg.norm(self.Rc1c2 - Rc1c2_check)
            print(f"Rc1c2 error: {err_Rc1c2}")

            Rc2c1_check = self.Rc2w @ self.Rc1w.T
            err_Rc2c1 = np.linalg.norm(self.Rc1c2.T - Rc2c1_check)
            print(f"Rc2c1 error: {err_Rc2c1}")

            points_3d_c2_check = (self.Rc2w @ self.points_3d_w2.T + self.tc2w.reshape(3, 1)).T
            assert self.points_3d_c2.shape == points_3d_c2_check.shape
            err_3d_c2 = np.linalg.norm(self.points_3d_c2 - points_3d_c2_check)
            print(f"3D w2 to 3D c2 error: {err_3d_c2}")

            points_3d_c1_check = (self.Rc1w @ self.points_3d_w1.T + self.tc1w.reshape(3, 1)).T
            assert self.points_3d_c1.shape == points_3d_c1_check.shape
            err_3d_c1 = np.linalg.norm(self.points_3d_c1 - points_3d_c1_check)
            print(f"3D w1 to 3D c1 error: {err_3d_c1}")

            points_3d_w2_check = (self.Rwc2 @ self.points_3d_c2.T + self.twc2.reshape(3, 1)).T
            assert self.points_3d_w2.shape == points_3d_w2_check.shape
            err_3d_w2 = np.linalg.norm(self.points_3d_w2 - points_3d_w2_check)
            print(f"3D c2 to 3D w2 error: {err_3d_w2}")

            points_3d_w1_check = (self.Rwc1 @ self.points_3d_c1.T + self.twc1.reshape(3, 1)).T
            assert self.points_3d_w1.shape == points_3d_w1_check.shape
            err_3d_w1 = np.linalg.norm(self.points_3d_w1 - points_3d_w1_check)
            print(f"3D c1 to 3D w1 error: {err_3d_w1}")

            points_2d_c2_check, _, _ = project_points(
                K2, None, self.points_3d_c2, camera.width, camera.height
            )
            assert self.points_2d_c2.shape == points_2d_c2_check.shape
            err_2d_c2 = np.linalg.norm(self.points_2d_c2 - points_2d_c2_check)
            print(f"3D c2 to 2D uv error: {err_2d_c2}")

            points_2d_c1_check, _, _ = project_points(
                K1, None, self.points_3d_c1, camera.width, camera.height
            )
            assert self.points_2d_c1.shape == points_2d_c1_check.shape
            err_2d_c1 = np.linalg.norm(self.points_2d_c1 - points_2d_c1_check)
            print(f"3D c1 to 2D uv error: {err_2d_c1}")

            points_3d_c1_check = (
                gt_delta_s * self.Rc1c2 @ self.points_3d_c2.T + self.tc1c2.reshape(3, 1)
            ).T
            assert self.points_3d_c1.shape == points_3d_c1_check.shape
            err_3d_c1 = np.linalg.norm(self.points_3d_c1 - points_3d_c1_check)
            # err_3d_c1 = np.mean(np.linalg.norm(self.points_3d_c1 - points_3d_c1_check, axis=1))
            print(f"3D c2 to 3D c1 error: {err_3d_c1}")

        print(f"Checking solution after motion with gt corrections ...")
        check_solution(
            self.points_2d_c1,
            self.points_3d_w1,
            self.points_2d_c2,
            self.points_3d_w2,
            self.K1,
            self.Rc1w,
            self.tc1w,
            self.K2,
            self.Rc2w,
            self.tc2w,
            self.gt_sc1c2,
            self.gt_Rc1c2,
            self.gt_tc1c2,
        )

        if True:
            print(f"Checking solution after motion without gt corrections ...")
            check_solution(
                self.points_2d_c1,
                self.points_3d_w1,
                self.points_2d_c2,
                self.points_3d_w2,
                self.K1,
                self.Rc1w,
                self.tc1w,
                self.K2,
                self.Rc2w,
                self.tc2w,
                1.0,
                self.Rc1c2_initial,
                self.tc1c2_initial,
            )

        default_color = np.array([255, 0, 0], dtype=np.uint8)

        # prepare frame 1
        num_keypoints = self.points_2d_c1.shape[0]
        frame1 = Frame(camera=self.camera, img=None)
        frame1.update_pose(self.gt_Tc1w.copy())
        frame1.kpsu = self.points_2d_c1
        frame1.octaves = np.full(num_keypoints, 0).astype(np.int32)  # All at level 0
        frame1.outliers = np.full(num_keypoints, False, dtype=bool)  # Initially no outliers
        frame1.points = np.array(
            [MapPoint(self.points_3d_w1[i], default_color) for i in range(num_keypoints)]
        )
        # FeatureTrackerShared.feature_manager = FakeFeatureManager(num_keypoints)
        tracker_config = FeatureTrackerConfigs.ORB2
        tracker_config["num_features"] = 1000
        feature_tracker = feature_tracker_factory(**tracker_config)
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

        # prepare frame 2
        num_keypoints2 = self.points_2d_c2.shape[0]
        assert num_keypoints2 == num_keypoints
        frame2 = Frame(camera=self.camera, img=None)
        frame2.update_pose(self.gt_Tc2w.copy())
        frame2.kpsu = self.points_2d_c2
        frame2.octaves = np.full(num_keypoints2, 0).astype(np.int32)  # all at level 0
        frame2.outliers = np.full(num_keypoints2, False, dtype=bool)  # Initially no outliers
        frame2.points = np.array(
            [MapPoint(self.points_3d_w2[i], default_color) for i in range(num_keypoints2)]
        )

        # prepare kf1
        self.kf1 = KeyFrame(frame=frame1)
        for i, mp in enumerate(self.kf1.points):
            res = mp.add_observation(self.kf1, i)
            # print(f'kf1: added point {i} with res: {res}')

        # prepare kf2
        self.kf2 = KeyFrame(frame=frame2)
        for i, mp in enumerate(self.kf2.points):
            res = mp.add_observation(self.kf2, i)
            # print(f'kf2: added point {i} with res: {res}')

        self.map_points1 = self.kf1.get_points()
        self.map_points2 = self.kf2.get_points()
        self.map_point_matches12 = (
            self.map_points2
        )  # map_point_matches12[i] = map point of kf2 matched with i-th map point of kf1

    def test_optimize_sim3(self):

        self.init_camera()

        self.create_perfect_world()

        print("----------------------------------------------------------------------")

        self.apply_motion()

        print("----------------------------------------------------------------------")

        print(f"starting optimize_sim3... ")

        use_optimizer_gtsam = False
        print(f"use_optimizer_gtsam: {use_optimizer_gtsam}")
        if use_optimizer_gtsam:
            optimize_sim3 = optimizer_gtsam.optimize_sim3
        else:
            optimize_sim3 = optimizer_g2o.optimize_sim3

        print(f"Rc1c2_initial: {self.Rc1c2_initial.ravel()}")
        print(f"tc1c2_initial: {self.tc1c2_initial.ravel()}")
        print(f"sc1c2_initial: {1.0}")

        # Call the optimize_sim3 function
        num_correspondences, R12_opt, t12_opt, s12_opt, delta_err = optimize_sim3(
            self.kf1,
            self.kf2,
            self.map_points1,
            self.map_point_matches12,
            self.Rc1c2_initial,
            self.tc1c2_initial,
            1.0,
            self.th2,
            self.fix_scale,
            verbose=True,
        )

        self.assertGreater(num_correspondences, 0)

        print(f"num_correspondences: {num_correspondences}")
        print(f"delta_err: {delta_err}")
        print(f"R12_opt: {R12_opt.ravel()}")
        print(f"t12_opt: {t12_opt.ravel()}")
        print(f"s12_opt: {s12_opt}")

        R12_diff = np.linalg.norm(R12_opt @ self.gt_Rc1c2.T - np.eye(3))
        t12_diff = np.linalg.norm(t12_opt.ravel() - self.gt_tc1c2.ravel())
        s12_diff = np.linalg.norm(s12_opt - self.gt_sc1c2)
        print(f"R diff: {R12_diff}, t diff: {t12_diff}, s diff: {s12_diff}")

        print(f"Checking solution with computed corrections")
        check_solution(
            self.points_2d_c1,
            self.points_3d_w1,
            self.points_2d_c2,
            self.points_3d_w2,
            self.K1,
            self.Rc1w,
            self.tc1w,
            self.K2,
            self.Rc2w,
            self.tc2w,
            s12_opt,
            R12_opt,
            t12_opt,
        )

        # Check if the result is valid
        self.assertGreater(
            num_correspondences, 0, "Number of correspondences should be greater than 0"
        )
        self.assertTrue(R12_diff < 1e-3)
        self.assertTrue(t12_diff < 1e-3)

        # self.assertEqual(R12_opt.shape, (3, 3), "R12 should be a 3x3 matrix")
        # self.assertEqual(t12_opt.shape, (3,), "t12 should be a 3D vector")
        # self.assertGreater(s12_opt, 0, "Scale should be positive")

        print("done")


if __name__ == "__main__":
    unittest.main()
