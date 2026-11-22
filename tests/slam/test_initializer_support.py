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
* Bounded stereo startup fallback and reset; no dataset or model weights needed.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from slam_test_support import SensorType, make_slam, _frame, _texture
import pyslam.slam.initializer as initializer_module
from pyslam.config_parameters import Parameters
from pyslam.slam.initializer import Initializer
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared


class InitializerSupportTest(unittest.TestCase):
    def test_stereo_startup_boosts_both_extractors_and_restores_them_after_initialization(self):
        slam = make_slam(self, SensorType.STEREO)
        texture = _texture()
        for index in range(3):
            left, depth = _frame(texture, index)
            # Exact synthetic depth isolates budget/state transitions from stereo matching.
            slam.track(left, left, depth, index, float(index))
            expected = 1600 if index < 2 else 800
            self.assertEqual(FeatureTrackerShared.feature_tracker.num_features, expected)
            self.assertEqual(FeatureTrackerShared.feature_tracker_right.num_features, expected)
            self.assertEqual(slam.tracking.state.name, "NOT_INITIALIZED" if index == 0 else "OK")

    def test_ba_acceptance_compares_reprojection_before_and_after(self):
        # Solver chi2 includes a third stereo residual and is not comparable to
        # the left-image reprojection error. Still reject failed BA or worsening poses.
        for chi2, after, expected in ((10.0, 0.5, True), (0.1, 3.0, False),
                                     (-1.0, 0.5, False), (float("nan"), 0.5, False),
                                     (1.0, float("-inf"), False)):
            with self.subTest(chi2=chi2, after=after):
                initializer = Initializer(SensorType.STEREO)
                initializer.num_min_triangulated_points = 30
                initializer.num_min_features = 40
                initializer.mask_match = np.ones((40, 1), dtype=np.uint8)
                frame = MagicMock()
                frame.kps = frame.kpsn = np.zeros((40, 2))
                frame.camera.K = np.eye(3)
                frame.camera.D = np.zeros(5)
                frame.unproject_points_3d.return_value = (np.ones((40, 3)), np.ones(40, dtype=bool))
                local_map = MagicMock()
                local_map.points = list(range(40))
                local_map.compute_mean_reproj_error.side_effect = [2.0, after]
                point = MagicMock()
                point.pt.return_value = np.ones(3)
                local_map.add_points.return_value = (40, np.ones(40, dtype=bool), [point] * 40)
                local_map.optimize.return_value = (chi2, 0)
                matches = MagicMock(idxs1=np.arange(40), idxs2=np.arange(40))
                with patch.object(initializer_module, "Map", return_value=local_map), \
                     patch.object(initializer_module, "KeyFrame", side_effect=lambda f, img: f), \
                     patch.object(initializer_module, "match_frames", return_value=matches), \
                     patch.object(initializer_module.FeatureTrackerShared, "oriented_features", False), \
                     patch.object(initializer, "estimatePose", return_value=np.eye(4)), \
                     patch.object(initializer_module.cv2, "solvePnP", return_value=(True, np.zeros(3), np.zeros(3))):
                    _, accepted = initializer.process_frames(frame, None, frame)
                self.assertEqual(accepted, expected)
                self.assertEqual(local_map.compute_mean_reproj_error.call_count, 2)

    def test_monocular_ba_acceptance_keeps_original_solver_metric(self):
        # Monocular BA has only reprojection edges: the original chi2 comparison is kept
        # and no extra post-BA reprojection pass runs.
        for chi2, rejected_by_ba in ((3.0, True), (float("nan"), True), (0.5, False)):
            with self.subTest(chi2=chi2):
                initializer = Initializer(SensorType.MONOCULAR)
                initializer.check_min_frame_distance = False  # same mock frame as ref/cur
                initializer.num_min_triangulated_points = 30
                initializer.num_min_features = 40
                initializer.mask_match = np.ones((40, 1), dtype=np.uint8)
                frame = MagicMock()
                frame.kps = frame.kpsn = np.zeros((40, 2))
                frame.compute_points_median_depth.return_value = 1.0
                frame.Ow.return_value = np.zeros(3)  # zero baseline: rejected after BA
                local_map = MagicMock()
                local_map.points = list(range(40))
                local_map.compute_mean_reproj_error.return_value = 2.0
                point = MagicMock()
                point.pt.return_value = np.ones(3)
                local_map.add_points.return_value = (40, np.ones(40, dtype=bool), [point] * 40)
                local_map.optimize.return_value = (chi2, 0)
                matches = MagicMock(idxs1=np.arange(40), idxs2=np.arange(40))
                with patch.object(initializer_module, "Map", return_value=local_map), \
                     patch.object(initializer_module, "KeyFrame", side_effect=lambda f, img: f), \
                     patch.object(initializer_module, "match_frames", return_value=matches), \
                     patch.object(initializer_module, "triangulate_normalized_points",
                                  return_value=(np.ones((40, 3)), np.ones(40, dtype=bool))), \
                     patch.object(initializer_module.FeatureTrackerShared, "oriented_features", False), \
                     patch.object(initializer, "estimatePose", return_value=np.eye(4)):
                    _, accepted = initializer.process_frames(frame, None, frame)
                self.assertFalse(accepted)
                self.assertEqual(local_map.compute_mean_reproj_error.call_count, 1)
                # Only a BA-accepted pose reaches the monocular depth/baseline checks.
                self.assertEqual(frame.compute_points_median_depth.called, not rejected_by_ba)

    def test_stereo_fallback_eventually_handles_sparse_depth_without_dropping_below_pnp_support(self):
        initializer = Initializer(SensorType.STEREO)
        with patch.object(Parameters, "kInitializerNumMinTriangulatedPointsStereo", 100), \
             patch.object(Parameters, "kInitializerNumMinNumPointsForPnPWithDepth", 15):
            thresholds = []
            for failures in (0, 10, 11, 20, 30, 1000):
                initializer.num_failures = failures
                initializer.check_num_failures()
                thresholds.append(initializer.num_min_triangulated_points)
            self.assertEqual(thresholds[:2], [100, 100])
            self.assertGreater(thresholds[2], 60)
            self.assertGreater(thresholds[3], 40)
            self.assertEqual(thresholds[4:], [30, 30])
            self.assertTrue(all(a >= b for a, b in zip(thresholds, thresholds[1:])))

    def test_stereo_floor_does_not_raise_a_deliberately_lower_configured_target(self):
        with patch.object(Parameters, "kInitializerNumMinTriangulatedPointsStereo", 10):
            initializer = Initializer(SensorType.STEREO)
            initializer.num_failures = 100
            initializer.check_num_failures()
            self.assertEqual(initializer.num_min_triangulated_points, 10)

    def test_monocular_and_rgbd_keep_their_existing_one_step_fallback(self):
        for sensor in (SensorType.MONOCULAR, SensorType.RGBD):
            initializer = Initializer(sensor)
            base = initializer.num_min_triangulated_points
            for failures in (11, 1000):
                initializer.num_failures = failures
                initializer.check_num_failures()
                self.assertAlmostEqual(initializer.num_min_triangulated_points, base * 2 / 3)

    def test_reset_restores_startup_threshold_and_checks(self):
        initializer = Initializer(SensorType.STEREO)
        initializer.num_failures = 100
        initializer.check_num_failures()
        initializer.check_cell_coverage = False
        initializer.check_min_frame_distance = False
        initializer.frames.append(object())
        initializer.f_ref = object()
        initializer.reset()
        self.assertEqual(initializer.num_failures, 0)
        self.assertEqual(initializer.num_min_triangulated_points, Parameters.kInitializerNumMinTriangulatedPointsStereo)
        self.assertEqual(initializer.check_cell_coverage, Parameters.kInitializerUseCellCoverageCheck)
        self.assertEqual(initializer.check_min_frame_distance, Parameters.kInitializerUseMinFrameDistanceCheck)
        self.assertFalse(initializer.frames)
        self.assertIsNone(initializer.f_ref)
