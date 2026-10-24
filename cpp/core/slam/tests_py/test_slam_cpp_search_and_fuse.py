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

"""Regression test for search_and_fuse on monocular keyframes.

Monocular keyframes have an empty kps_ur. search_and_fuse used to read
kps_ur[kd_idx] unconditionally, which segfaulted in local mapping (fuse_map_points).
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


def _build_mono_pair(module):
    """kf1 owns map points of a planar scene; kf2 sees the same scene but has no map points.

    Both keyframes are monocular (empty kps_ur). Fusing kf1's points into kf2 must add
    observations to kf2 instead of crashing.
    """
    camera = _make_camera(module)
    K = camera.K

    xs, ys = np.meshgrid(np.linspace(-0.6, 0.6, 4), np.linspace(-0.4, 0.4, 3))
    pts_w = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 3.0)], axis=1)
    n = pts_w.shape[0]

    Tc1w = np.eye(4)
    t_c2_w = np.array([0.12, 0.0, 0.0], dtype=np.float64)
    Tc2w = poseRt(np.eye(3), -t_c2_w)

    kps1 = _project(K, pts_w).astype(np.float32)
    kps2 = _project(K, pts_w - t_c2_w.reshape(1, 3)).astype(np.float32)

    rng = np.random.RandomState(42)
    des = rng.randint(0, 256, size=(n, 32), dtype=np.uint8)
    color = np.array([255, 0, 0], dtype=np.uint8)
    octaves = np.zeros(n, dtype=np.int32)

    def _make_frame(Tcw, kps):
        frame = module.Frame(camera=camera, img=None)
        frame.update_pose(Tcw.copy())
        frame.kps = kps.copy()
        frame.kpsu = kps.copy()
        frame.octaves = octaves.copy()
        frame.des = des.copy()
        frame.outliers = np.zeros(n, dtype=bool)
        return frame

    frame1 = _make_frame(Tc1w, kps1)
    frame1.points = np.array([module.MapPoint(pts_w[i], color) for i in range(n)], dtype=object)
    kf1 = module.KeyFrame(frame=frame1)

    frame2 = _make_frame(Tc2w, kps2)
    frame2.points = np.array([None] * n, dtype=object)
    kf2 = module.KeyFrame(frame=frame2)

    ow2 = np.asarray(kf2.Ow()).reshape(3)
    for i, mp in enumerate(kf1.get_points()):
        mp.add_observation(kf1, i)
        mp.des = np.ascontiguousarray(des[i].copy())
        # Keep the predicted octave low so it matches the synthetic keypoints in kf2.
        mp._min_distance = 0.1
        mp._max_distance = float(np.linalg.norm(pts_w[i] - ow2)) + 1e-3
        mp.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return kf1, kf2, n


@unittest.skipUnless(CPP_AVAILABLE, "C++ core is not available")
class TestSearchAndFuseMonoCpp(TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_feature_tracker()

    def test_cpp_fuses_into_mono_keyframe(self):
        kf1, kf2, n = _build_mono_pair(cpp_module)
        kps_ur = kf2.kps_ur  # the binding returns None when kps_ur is empty
        self.assertTrue(kps_ur is None or len(kps_ur) == 0, "test setup: kf2 must be monocular")

        num_fused = cpp_module.ProjectionMatcher.search_and_fuse(
            kf1.get_points(),
            kf2,
            max_descriptor_distance=kMaxDescriptorDistance,
        )

        self.assertEqual(int(num_fused), n)
        for i, mp in enumerate(kf1.get_points()):
            self.assertTrue(mp.is_in_keyframe(kf2), f"point {i} should be observed by kf2")


if __name__ == "__main__":
    unittest.main()
