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
* Adaptive descriptor-distance and reprojection thresholds, for the core PYSLAM_USE_CPP selects.
"""

import unittest

import numpy as np
from slam_test_support import USE_CPP, _make_camera, init_feature_tracker
from pyslam.config_parameters import Parameters
from pyslam.slam import Frame, SLAMDynamicConfig
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared

INIT_DISTANCE = 50.0
NUM_MATCHES = 100


def frame_with_descriptors(camera, des):
    frame = Frame(camera=camera, img=None)
    frame.kps = np.zeros((len(des), 2), dtype=np.float32)
    frame.des = np.ascontiguousarray(des)
    return frame


class TestSlamDynamicConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_feature_tracker()
        cls.camera = _make_camera(USE_CPP)

    def setUp(self):
        saved = Parameters.kMaxDescriptorDistance

        def restore():
            Parameters.kMaxDescriptorDistance = saved
            FeatureTrackerShared.update_cpp_module_dynamic_config_parameters()

        self.addCleanup(restore)

    def update_descriptors(self, config, des_ref, des_cur, times):
        f_ref = frame_with_descriptors(self.camera, des_ref)
        f_cur = frame_with_descriptors(self.camera, des_cur)
        idxs = list(range(len(des_ref)))
        for _ in range(times):
            sigma = config.update_descriptor_stats(f_ref, f_cur, idxs, idxs)
        return sigma

    def test_descriptor_sigma_shrinks_to_lower_bound(self):
        config = SLAMDynamicConfig(INIT_DISTANCE)
        des = np.random.default_rng(1).integers(0, 256, (NUM_MATCHES, 32), dtype=np.uint8)
        first = self.update_descriptors(config, des, des, 1)
        self.assertLess(first, INIT_DISTANCE)
        last = self.update_descriptors(config, des, des, 50)
        self.assertAlmostEqual(last, 0.7 * INIT_DISTANCE, places=3)
        self.assertAlmostEqual(Parameters.kMaxDescriptorDistance, last, places=3)

    def test_descriptor_sigma_grows_to_upper_bound(self):
        config = SLAMDynamicConfig(INIT_DISTANCE)
        rng = np.random.default_rng(2)
        des_ref = rng.integers(0, 256, (NUM_MATCHES, 32), dtype=np.uint8)
        des_cur = rng.integers(0, 256, (NUM_MATCHES, 32), dtype=np.uint8)
        last = self.update_descriptors(config, des_ref, des_cur, 50)
        self.assertAlmostEqual(last, 1.3 * INIT_DISTANCE, places=3)

    def test_reprojection_sigma_is_a_moving_average_with_floor(self):
        config = SLAMDynamicConfig(INIT_DISTANCE)
        start = Parameters.kMaxReprojectionDistanceMap
        value = config.update_reproj_err_map_stats(start + 10.0)
        self.assertAlmostEqual(value, start + 1.0, places=4)
        for _ in range(200):
            value = config.update_reproj_err_map_stats(0.0)
        self.assertEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
