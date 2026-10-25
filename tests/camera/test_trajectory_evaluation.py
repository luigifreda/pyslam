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

import unittest
import tempfile
from types import SimpleNamespace

import numpy as np

from trajectory_evaluation import translation_ate
from run_pinhole_regression import compare
from replay_sequence import groundtruth_samples


class TestTranslationATE(unittest.TestCase):
    def test_tum_uses_raw_mocap_instead_of_repeated_frame_associations(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as file:
            file.write("# timestamp tx ty tz qx qy qz qw\n1 2 3 4 0 0 0 1\n2 3 4 5 0 0 0 1\n")
            file.flush()
            gt = SimpleNamespace(type=SimpleNamespace(name="TUM"), filename=file.name, scale=2.)
            positions, timestamps = groundtruth_samples(gt)
        np.testing.assert_array_equal(timestamps, [1., 2.])
        np.testing.assert_array_equal(positions, [[4., 6., 8.], [6., 8., 10.]])

    def setUp(self):
        self.ts = np.arange(5) * 0.02
        self.xyz = np.array([[0., 0, 0], [1, 0, 0], [1, 2, 0], [0, 1, 1], [1, 0, 2]])

    def test_known_similarity_and_metric_scale(self):
        R = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        reference = 2.5 * self.xyz @ R.T + [4, -3, 1]
        result = translation_ate(self.ts, self.xyz, self.ts, reference, find_scale=True)
        self.assertLess(result['ate_rmse_m'], 1e-12)
        self.assertAlmostEqual(result['alignment_scale'], 2.5)
        self.assertGreater(translation_ate(self.ts, self.xyz, self.ts, reference)['ate_rmse_m'], 0.5)

    def test_sparse_gap_is_not_interpolated(self):
        ts = np.r_[self.ts, 2.0, 2.01, 5.0]
        xyz = np.vstack([self.xyz, [[100, 0, 0]] * 3])
        gt_ts = np.r_[self.ts, 4.0]
        gt_xyz = np.vstack([self.xyz, [[0, 0, 0]]])
        result = translation_ate(ts, xyz, gt_ts, gt_xyz)
        self.assertEqual(result['gt_associations'], 5)
        self.assertLess(result['ate_rmse_m'], 1e-12)

    def test_interpolated_motion(self):
        ts = (self.ts[1:] + self.ts[:-1]) / 2
        xyz = (self.xyz[1:] + self.xyz[:-1]) / 2
        result = translation_ate(ts, xyz, self.ts, self.xyz)
        self.assertEqual(result['gt_associations'], 4)
        self.assertLess(result['ate_rmse_m'], 1e-12)

    def test_invalid_and_degenerate_input(self):
        for ts, xyz in ((self.ts[:2], self.xyz[:2]),
                        (self.ts, np.zeros_like(self.xyz)),
                        (self.ts[::-1], self.xyz),
                        (self.ts, self.xyz * np.nan)):
            with self.subTest(ts=ts), self.assertRaises(ValueError):
                translation_ate(ts, xyz, self.ts, self.xyz)


class TestReplayRegressionGate(unittest.TestCase):
    def setUp(self):
        self.report = dict(dataset='TUM_DATASET', name='sequence', sensor='RGBD', start=0,
                           processed_frames=100, alignment='se3',
                           track_and_map_ms={'mean': 10.0}, states={'OK': 99},
                           gt_associations=99, trajectory_poses=99, ate_rmse_m=0.01)

    def test_identical_runs_pass(self):
        self.assertEqual(compare(self.report, self.report)['failures'], [])

    def test_timing_accuracy_and_coverage_fail_independently(self):
        for changes, failure in (({'track_and_map_ms': {'mean': 15}}, 'throughput'),
                                 ({'ate_rmse_m': 0.1}, 'ate'),
                                 ({'states': {'OK': 90}}, 'tracking_coverage'),
                                 ({'trajectory_poses': 90}, 'trajectory_coverage'),
                                 ({'gt_associations': 90}, 'mocap_coverage')):
            with self.subTest(failure=failure):
                result = compare(self.report, dict(self.report, **changes))
                self.assertEqual(result['failures'], [failure])

    def test_incompatible_or_unevaluable_runs_fail(self):
        for change in ({'sensor': 'MONOCULAR'}, {'local_mapping': 'threaded'},
                       {'loop_closure': True}, {'ate_rmse_m': -1}, {'gt_associations': 0},
                       {'ate_rmse_m': float('nan')}, {'track_and_map_ms': {'mean': 0}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare(self.report, dict(self.report, **change))


if __name__ == '__main__':
    unittest.main()
