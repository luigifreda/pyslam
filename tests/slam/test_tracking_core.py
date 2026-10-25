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
*
* TrackingCore helpers on synthetic frames, for the core PYSLAM_USE_CPP selects.
"""

import unittest

import numpy as np
from slam_test_support import USE_CPP, SensorType, _make_camera, init_feature_tracker
from pyslam.config_parameters import Parameters
from pyslam.slam import Frame, MapPoint, TrackingCore
from pyslam.utilities.geometry import poseRt, rotation_matrix_from_yaw_pitch_roll

NUM_POINTS = 200
NUM_WRONG_MATCHES = 20


def make_frame(camera, Tcw, points_w):
    """A frame whose keypoints are the exact projections of points_w."""
    n = len(points_w)
    frame = Frame(camera=camera, img=None)
    frame.update_pose(Tcw)
    points_c = (Tcw[:3, :3] @ points_w.T + Tcw[:3, 3:4]).T
    uv, _ = camera.project(points_c)
    frame.kps = np.ascontiguousarray(uv, dtype=np.float32)
    frame.kpsu = frame.kps.copy()
    frame.kpsn = np.ascontiguousarray(points_c[:, :2] / points_c[:, 2:3], dtype=np.float32)
    frame.octaves = np.zeros(n, dtype=np.int32)
    frame.des = np.zeros((n, 32), dtype=np.uint8)
    frame.depths = np.ascontiguousarray(points_c[:, 2], dtype=np.float32)
    frame.points = np.array([None] * n, dtype=object)
    frame.outliers = np.zeros(n, dtype=bool)
    return frame


def has_point(frame, idx):
    return frame.points[idx] is not None


class TrackingCoreTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_feature_tracker()
        cls.camera = _make_camera(USE_CPP)

    def two_views(self, planar=False):
        rng = np.random.default_rng(0)
        z = np.full(NUM_POINTS, 4.0) if planar else rng.uniform(3.0, 8.0, NUM_POINTS)
        points_w = np.column_stack((
            rng.uniform(-2.0, 2.0, NUM_POINTS), rng.uniform(-1.5, 1.5, NUM_POINTS), z))
        Rcw = rotation_matrix_from_yaw_pitch_roll(4.0, 2.0, 1.0)
        Tcw = poseRt(Rcw, np.array([0.3, 0.0, 0.05]))
        f_ref = make_frame(self.camera, np.eye(4), points_w)
        f_cur = make_frame(self.camera, Tcw, points_w)
        idxs_ref = np.arange(NUM_POINTS)
        idxs_cur = idxs_ref.copy()
        idxs_cur[:NUM_WRONG_MATCHES] = np.roll(idxs_cur[:NUM_WRONG_MATCHES], 7)
        return f_ref, f_cur, idxs_ref, idxs_cur, Rcw


class TestPoseFromEssentialMatrix(TrackingCoreTestCase):
    def test_rotation_is_recovered_and_wrong_matches_removed(self):
        f_ref, f_cur, idxs_ref, idxs_cur, Rcw = self.two_views()
        f_cur.update_pose(np.eye(4))
        idxs_ref_out, idxs_cur_out, num_inliers = TrackingCore.estimate_pose_by_fitting_ess_mat(
            f_ref, f_cur, idxs_ref, idxs_cur)
        self.assertEqual(num_inliers, NUM_POINTS - NUM_WRONG_MATCHES)
        self.assertEqual(len(idxs_ref_out), num_inliers)
        np.testing.assert_array_equal(np.asarray(idxs_ref_out), np.asarray(idxs_cur_out))
        pose = np.asarray(f_cur.pose())
        np.testing.assert_allclose(pose[:3, :3], Rcw, atol=1e-3)
        # The essential matrix has no scale: the translation stays the reference one.
        np.testing.assert_allclose(pose[:3, 3], np.zeros(3), atol=1e-9)

    def test_empty_or_mismatched_input_returns_no_inliers(self):
        f_ref, f_cur, idxs_ref, _, _ = self.two_views()
        _, _, num_inliers = TrackingCore.estimate_pose_by_fitting_ess_mat(
            f_ref, f_cur, idxs_ref, idxs_ref[:-1])
        self.assertEqual(num_inliers, 0)


class TestHomographyRansac(TrackingCoreTestCase):
    def test_planar_scene_keeps_true_matches(self):
        f_ref, f_cur, idxs_ref, idxs_cur, _ = self.two_views(planar=True)
        ok, idxs_cur_out, idxs_ref_out, num_matched, num_outliers = (
            TrackingCore.find_homography_with_ransac(f_cur, f_ref, idxs_cur, idxs_ref))
        self.assertTrue(ok)
        self.assertEqual(num_matched, NUM_POINTS - NUM_WRONG_MATCHES)
        self.assertEqual(num_outliers, NUM_WRONG_MATCHES)
        np.testing.assert_array_equal(np.asarray(idxs_cur_out), np.asarray(idxs_ref_out))

    def test_too_few_inliers_fails(self):
        f_ref, f_cur, idxs_ref, idxs_cur, _ = self.two_views(planar=True)
        ok, idxs_cur_out, _, num_matched, _ = TrackingCore.find_homography_with_ransac(
            f_cur, f_ref, idxs_cur, idxs_ref, 5.0, NUM_POINTS)
        self.assertFalse(ok)
        self.assertEqual(len(idxs_cur_out), 0)
        self.assertEqual(num_matched, 0)


class TestCloseAndVoPoints(TrackingCoreTestCase):
    def depth_frame(self):
        # depth_threshold of the test camera: ThDepth * b = 50 * 0.08 = 4 m
        frame = make_frame(self.camera, np.eye(4), np.column_stack((
            np.linspace(-0.5, 0.5, 6), np.linspace(-0.3, 0.3, 6), np.ones(6))))
        frame.depths = np.array([1.0, 2.0, 3.0, 10.0, -1.0, 0.5], dtype=np.float32)
        points = np.array([None] * 6, dtype=object)
        points[0] = MapPoint(np.array([0.0, 0.0, 1.0]), np.array([0, 0, 255]))
        points[1] = MapPoint(np.array([0.0, 0.0, 2.0]), np.array([0, 0, 255]))
        frame.points = points
        frame.outliers = np.array([False, True, False, False, False, False])
        return frame

    def test_count_close_points(self):
        frame = self.depth_frame()
        self.assertEqual(self.camera.depth_threshold, 4.0)
        num_tracked, num_non_tracked, tracked_mask = (
            TrackingCore.count_tracked_and_non_tracked_close_points(frame, SensorType.RGBD))
        # close (0 < d < 4): 0, 1, 2, 5; tracked: 0 (1 is an outlier)
        self.assertEqual((num_tracked, num_non_tracked), (1, 3))
        np.testing.assert_array_equal(np.asarray(tracked_mask), [True, False, False, False, False, False])

    def test_monocular_has_no_close_points(self):
        num_tracked, num_non_tracked, _ = TrackingCore.count_tracked_and_non_tracked_close_points(
            self.depth_frame(), SensorType.MONOCULAR)
        self.assertEqual((num_tracked, num_non_tracked), (0, 0))

    def test_vo_points_cover_close_depths(self):
        frame = self.depth_frame()
        created = TrackingCore.create_vo_points(frame, 1)
        for idx in (0, 1, 2, 5):
            self.assertTrue(has_point(frame, idx), idx)
        self.assertFalse(has_point(frame, 4))  # invalid depth
        # Python stops at the threshold; C++ (like ORB-SLAM2) also adds the first farther point.
        self.assertIn(len(created), (4, 5))
        depths = sorted(float(np.asarray(p.pt()).ravel()[2]) for p in created)
        np.testing.assert_allclose(depths[:4], [0.5, 1.0, 2.0, 3.0], atol=1e-5)

    def test_vo_points_take_the_closest_beyond_threshold(self):
        frame = self.depth_frame()
        frame.depths = np.array([5.0, 9.0, 6.0, 10.0, -1.0, 7.0], dtype=np.float32)
        created = TrackingCore.create_vo_points(frame, 2)
        self.assertEqual(len(created), 2)
        self.assertTrue(has_point(frame, 0) and has_point(frame, 2))
        self.assertFalse(has_point(frame, 3))

    def test_vo_points_need_depths(self):
        frame = make_frame(self.camera, np.eye(4), np.array([[0.0, 0.0, 1.0]]))
        frame.depths = np.array([-1.0], dtype=np.float32)
        self.assertEqual(len(TrackingCore.create_vo_points(frame, Parameters.kMaxNumVisualOdometryPoints)), 0)


if __name__ == "__main__":
    unittest.main()
