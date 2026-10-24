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

"""Regression tests for search_by_sim3 match expansion / reciprocity.

The 2->1 pass must record reciprocal candidates even when the 1->2 pass already
matched that kf1 point. Otherwise the final agreement check resets every new
pair and the function can only return its seed matches.
"""

import unittest
from unittest import TestCase

import numpy as np

import pyslam.config as config
from pyslam.config_parameters import Parameters

USE_CPP = True
Parameters.USE_CPP_CORE = USE_CPP

from pyslam.slam.cpp import CPP_AVAILABLE, cpp_module, python_module
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.slam.geometry_matchers import _search_by_sim3
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.utilities.geometry import poseRt


kMaxDescriptorDistance = 50.0
kNumSeedMatches = 3


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


def _build_loop_pair(module):
    """Two keyframes of the same 3D scene with unique matching descriptors.

    Only the first kNumSeedMatches correspondences are given as seeds. The
    remaining points are visible under the true Sim3 and share descriptors, so
    search_by_sim3 must expand them.
    """
    camera = _make_camera(module)
    K = camera.K

    xs, ys = np.meshgrid(np.linspace(-0.6, 0.6, 4), np.linspace(-0.4, 0.4, 3))
    pts_w = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 3.0)], axis=1)
    n = pts_w.shape[0]

    Tc1w = np.eye(4)
    t_c2_w = np.array([0.12, 0.0, 0.0], dtype=np.float64)
    Tc2w = poseRt(np.eye(3), -t_c2_w)

    pts_c1 = pts_w
    pts_c2 = pts_w - t_c2_w.reshape(1, 3)
    kps1 = _project(K, pts_c1).astype(np.float32)
    kps2 = _project(K, pts_c2).astype(np.float32)

    rng = np.random.RandomState(42)
    des = rng.randint(0, 256, size=(n, 32), dtype=np.uint8)
    color = np.array([255, 0, 0], dtype=np.uint8)
    octaves = np.zeros(n, dtype=np.int32)

    def _make_keyframe(Tcw, kps, pts3d):
        frame = module.Frame(camera=camera, img=None)
        frame.update_pose(Tcw.copy())
        frame.kps = kps.copy()
        frame.kpsu = kps.copy()
        frame.octaves = octaves.copy()
        frame.des = des.copy()
        frame.outliers = np.zeros(n, dtype=bool)
        mps = [module.MapPoint(pts3d[i], color) for i in range(n)]
        frame.points = np.array(mps, dtype=object)
        kf = module.KeyFrame(frame=frame)
        ow = np.asarray(kf.Ow()).reshape(3)
        for i, mp in enumerate(kf.get_points()):
            mp.add_observation(kf, i)
            mp.des = np.ascontiguousarray(des[i].copy())
            dist = float(np.linalg.norm(pts3d[i] - ow))
            # Keep predicted octave at 0 so it matches the synthetic keypoints.
            mp._min_distance = 0.1
            mp._max_distance = dist
            mp.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return kf

    kf1 = _make_keyframe(Tc1w, kps1, pts_w)
    kf2 = _make_keyframe(Tc2w, kps2, pts_w)

    # Sim3 from camera 2 to camera 1: p1 = s R p2 + t
    R12 = np.eye(3)
    t12 = t_c2_w.copy()
    s12 = 1.0
    idxs1 = list(range(kNumSeedMatches))
    idxs2 = list(range(kNumSeedMatches))
    return kf1, kf2, idxs1, idxs2, s12, R12, t12, n


def _assert_expansion(test_case, num_found, matches12, n, label):
    test_case.assertGreater(
        num_found,
        kNumSeedMatches,
        f"{label}: search_by_sim3 must expand beyond the {kNumSeedMatches} seed matches, "
        f"got {num_found}. If this is 0 extra matches, the 2->1 reciprocity guard is back.",
    )
    for i in range(kNumSeedMatches):
        test_case.assertEqual(
            int(matches12[i]),
            i,
            f"{label}: seed match {i} should be preserved, got {matches12[i]}",
        )
    extra = [i for i in range(n) if int(matches12[i]) == i and i >= kNumSeedMatches]
    test_case.assertGreaterEqual(
        len(extra),
        1,
        f"{label}: expected at least one expanded reciprocal pair, matches12={matches12}",
    )


class TestSearchBySim3Python(TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_feature_tracker()

    def test_python_expands_beyond_seed_matches(self):
        kf1, kf2, idxs1, idxs2, s12, R12, t12, n = _build_loop_pair(python_module)
        num_found, matches12, matches21 = _search_by_sim3(
            kf1,
            kf2,
            idxs1,
            idxs2,
            s12,
            R12,
            t12,
            max_descriptor_distance=kMaxDescriptorDistance,
        )
        _assert_expansion(self, num_found, matches12, n, "python")
        self.assertEqual(int(num_found), int(np.sum(np.asarray(matches12) != -1)))


@unittest.skipUnless(CPP_AVAILABLE, "C++ core is not available")
class TestSearchBySim3Cpp(TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_feature_tracker()

    def test_cpp_expands_beyond_seed_matches(self):
        kf1, kf2, idxs1, idxs2, s12, R12, t12, n = _build_loop_pair(cpp_module)
        num_found, matches12, matches21 = cpp_module.ProjectionMatcher.search_by_sim3(
            kf1,
            kf2,
            idxs1,
            idxs2,
            s12,
            R12,
            t12,
            max_descriptor_distance=kMaxDescriptorDistance,
        )
        _assert_expansion(self, num_found, matches12, n, "cpp")
        self.assertEqual(int(num_found), int(np.sum(np.asarray(matches12) != -1)))


if __name__ == "__main__":
    unittest.main()
