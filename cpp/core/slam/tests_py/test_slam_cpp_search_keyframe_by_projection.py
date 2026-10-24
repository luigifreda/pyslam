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

"""Regression tests for search_keyframe_by_projection skip indices.

already_matched_ref_idxs are positions in kf_ref.get_matched_points(), a compact
list that is shorter than the keypoint array. Relocalization used to pass keypoint
indices of the current frame; the C++ skip-mask is a vector<bool> sized to the compact
list, so those indices wrote past its allocation and aborted the process in free().
"""

import unittest
from unittest import TestCase

import numpy as np

import pyslam.config as config
from pyslam.config_parameters import Parameters

USE_CPP = True
Parameters.USE_CPP_CORE = USE_CPP

from pyslam.slam.cpp import CPP_AVAILABLE, cpp_module
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.utilities.geometry import poseRt


kMaxDescriptorDistance = 50.0
kMaxReprojDistance = 10.0
kRatioTest = 0.9


def _make_camera(module):
    camera = module.PinholeCamera(config=None)
    camera.fx = 517.306408
    camera.fy = 516.469215
    camera.cx = 318.643040
    camera.cy = 255.313989
    camera.width = 640
    camera.height = 480
    camera.bf = 40.0
    camera.b = camera.bf / camera.fx
    camera.fps = 30
    camera.set_intrinsic_matrices()
    camera.K = np.array(
        [[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float64
    )
    camera.Kinv = np.array(
        [
            [1 / camera.fx, 0, -camera.cx / camera.fx],
            [0, 1 / camera.fy, -camera.cy / camera.fy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    camera.u_min = 0.0
    camera.u_max = float(camera.width)
    camera.v_min = 0.0
    camera.v_max = float(camera.height)
    if hasattr(camera, "is_distorted"):
        camera.is_distorted = False
    return camera


def _project(K, pts_c):
    proj = (K @ pts_c.T).T
    return proj[:, :2] / proj[:, 2:3]


def _setup_feature_tracker():
    tracker_config = FeatureTrackerConfigs.ORB2.copy()
    tracker_config["num_features"] = 200
    feature_tracker = feature_tracker_factory(**tracker_config)
    FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)
    return feature_tracker


def _build_pair(module):
    """kf_ref has map points only on odd keypoints; f_cur sees the scene with no map points.

    With map points on half the keypoints, compact indices (positions in
    get_matched_points()) differ from keypoint indices, as in a real keyframe.
    """
    camera = _make_camera(module)
    K = camera.K

    xs, ys = np.meshgrid(np.linspace(-0.6, 0.6, 6), np.linspace(-0.4, 0.4, 4))
    pts_w = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 3.0)], axis=1)
    n = pts_w.shape[0]

    T_ref = np.eye(4)
    t_cur_w = np.array([0.12, 0.0, 0.0], dtype=np.float64)
    T_cur = poseRt(np.eye(3), -t_cur_w)

    kps_ref = _project(K, pts_w).astype(np.float32)
    kps_cur = _project(K, pts_w - t_cur_w.reshape(1, 3)).astype(np.float32)

    rng = np.random.RandomState(7)
    des = rng.randint(0, 256, size=(n, 32), dtype=np.uint8)
    color = np.array([0, 255, 0], dtype=np.uint8)
    octaves = np.zeros(n, dtype=np.int32)

    def _make_frame(Tcw, kps):
        frame = module.Frame(camera=camera, img=None)
        frame.update_pose(Tcw.copy())
        frame.kps = kps.copy()
        frame.kpsu = kps.copy()
        frame.octaves = octaves.copy()
        frame.des = des.copy()
        frame.angles = np.zeros(n, dtype=np.float32)
        frame.outliers = np.zeros(n, dtype=bool)
        return frame

    kp_idxs_with_points = [i for i in range(n) if i % 2 == 1]

    frame_ref = _make_frame(T_ref, kps_ref)
    points = [None] * n
    for i in kp_idxs_with_points:
        points[i] = module.MapPoint(pts_w[i], color)
    frame_ref.points = np.array(points, dtype=object)
    kf_ref = module.KeyFrame(frame=frame_ref)

    f_cur = _make_frame(T_cur, kps_cur)
    f_cur.points = np.array([None] * n, dtype=object)

    ow_cur = np.asarray(f_cur.Ow()).reshape(3)
    kf_points = kf_ref.get_points()
    for i in kp_idxs_with_points:
        mp = kf_points[i]
        mp.add_observation(kf_ref, i)
        mp.des = np.ascontiguousarray(des[i].copy())
        # Keep the predicted octave at 0 so it matches the synthetic keypoints in f_cur.
        mp._min_distance = 0.1
        mp._max_distance = float(np.linalg.norm(pts_w[i] - ow_cur)) + 1e-3
        mp.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    num_matched = len(kp_idxs_with_points)
    return kf_ref, f_cur, n, num_matched


def _search(kf_ref, f_cur, already_matched_ref_idxs):
    return cpp_module.ProjectionMatcher.search_keyframe_by_projection(
        kf_ref,
        f_cur,
        max_reproj_distance=kMaxReprojDistance,
        max_descriptor_distance=kMaxDescriptorDistance,
        ratio_test=kRatioTest,
        already_matched_ref_idxs=already_matched_ref_idxs,
    )


@unittest.skipUnless(CPP_AVAILABLE, "C++ core is not available")
class TestSearchKeyframeByProjectionCpp(TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_feature_tracker()

    def test_returns_compact_ref_indices(self):
        kf_ref, f_cur, n, num_matched = _build_pair(cpp_module)
        self.assertLess(num_matched, n, "test setup: compact list must be shorter than keypoints")

        idxs_ref, idxs_cur, num_found = _search(kf_ref, f_cur, [])

        self.assertEqual(int(num_found), num_matched)
        self.assertTrue(all(0 <= int(i) < num_matched for i in idxs_ref))

    def test_skips_valid_compact_indices(self):
        kf_ref, f_cur, n, num_matched = _build_pair(cpp_module)
        skip = [0, 1]

        idxs_ref, _, num_found = _search(kf_ref, f_cur, skip)

        self.assertEqual(int(num_found), num_matched - len(skip))
        self.assertFalse(set(int(i) for i in idxs_ref) & set(skip))

    def test_out_of_range_indices_are_ignored(self):
        """Keypoint-sized and negative indices must not write past the skip-mask."""
        kf_ref, f_cur, n, num_matched = _build_pair(cpp_module)
        # Indices at and beyond the compact length, up to the frame keypoint range and
        # much larger, as relocalization used to pass with 2000-4000 features.
        out_of_range = [num_matched, n - 1, 4000, 100000, -1, -50]

        idxs_ref, _, num_found = _search(kf_ref, f_cur, [0] + out_of_range)

        self.assertEqual(int(num_found), num_matched - 1)
        self.assertNotIn(0, [int(i) for i in idxs_ref])


if __name__ == "__main__":
    unittest.main()
