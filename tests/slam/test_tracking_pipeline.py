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
* Tracking and local mapping on a synthetic sequence, for the core PYSLAM_USE_CPP selects.
*
* Covers initialization, pose tracking, keyframe insertion (need_new_keyframe), and
* local-mapping triangulation through Slam.track().
"""

import unittest

import numpy as np
from slam_test_support import SensorType, make_slam, track_sequence

NUM_FRAMES = 14


class TestRgbdTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slam = make_slam(cls)
        cls.records = track_sequence(cls.slam, range(NUM_FRAMES))

    def test_initializes_on_second_frame(self):
        self.assertEqual(self.records[0]["state"], "NOT_INITIALIZED")
        self.assertEqual(self.records[1]["state"], "OK")
        self.assertEqual(self.records[1]["num_keyframes"], 2)
        self.assertGreater(self.records[1]["num_points"], 300)

    def test_stays_initialized(self):
        self.assertTrue(all(r["state"] == "OK" for r in self.records[1:]))

    def test_metric_pose_follows_camera_motion(self):
        for i, record in enumerate(self.records[2:], start=2):
            with self.subTest(frame=i):
                expected = np.array([record["true_x"], 0.0, 0.0])
                self.assertLess(np.linalg.norm(record["position"] - expected), 0.01)

    def test_keyframes_are_inserted_while_moving(self):
        counts = [r["num_keyframes"] for r in self.records[1:]]
        self.assertTrue(all(b - a in (0, 1) for a, b in zip(counts, counts[1:])))
        self.assertGreaterEqual(counts[-1], 5)

    def test_local_mapping_creates_points(self):
        self.assertGreater(self.records[-1]["num_points"], self.records[1]["num_points"])
        self.assertGreater(self.slam.local_mapping.total_num_triangulated_points, 0)

    def test_final_trajectory_covers_tracked_frames(self):
        poses, timestamps, ids = self.slam.get_final_trajectory()
        # Frame ids come from a process-wide counter, so only their spacing is fixed.
        self.assertEqual(list(ids), list(range(ids[0], ids[0] + NUM_FRAMES - 1)))
        self.assertEqual(list(timestamps), [float(i) for i in range(1, NUM_FRAMES)])
        last = np.asarray(poses[-1], float)
        self.assertLess(abs(last[0, 3] - self.records[-1]["true_x"]), 0.01)


class TestRgbdStaticCamera(unittest.TestCase):
    def test_no_keyframes_without_motion(self):
        slam = make_slam(self)
        records = track_sequence(slam, [0] + [1] * NUM_FRAMES)
        self.assertTrue(all(r["state"] == "OK" for r in records[1:]))
        counts = [r["num_keyframes"] for r in records]
        self.assertEqual(counts[-1], counts[NUM_FRAMES // 2])
        positions = np.array([r["position"] for r in records[2:]])
        self.assertLess(np.ptp(positions, axis=0).max(), 0.005)


class TestMonocularTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slam = make_slam(cls, SensorType.MONOCULAR)
        cls.records = track_sequence(cls.slam, range(0, 3 * NUM_FRAMES, 3), SensorType.MONOCULAR)

    def test_initializes_from_two_views(self):
        initialized = [i for i, r in enumerate(self.records) if r["state"] == "OK"]
        self.assertTrue(initialized)
        first = initialized[0]
        self.assertLessEqual(first, 3)
        self.assertEqual(self.records[first]["num_keyframes"], 2)
        self.assertGreater(self.records[first]["num_points"], 300)
        self.assertTrue(all(r["state"] == "OK" for r in self.records[first:]))

    def test_trajectory_is_correct_up_to_scale(self):
        tracked = [r for r in self.records if r["position"] is not None]
        self.assertGreaterEqual(len(tracked), NUM_FRAMES - 4)
        positions = np.array([r["position"] for r in tracked])
        true_x = np.array([r["true_x"] for r in tracked])
        scale = positions[:, 0] / true_x
        self.assertLess(np.std(scale) / np.mean(scale), 0.05)
        self.assertLess(np.abs(positions[:, 1:]).max(), 0.05 * positions[:, 0].max())

    def test_local_mapping_triangulates(self):
        self.assertGreater(self.slam.local_mapping.total_num_triangulated_points, 100)


if __name__ == "__main__":
    unittest.main()
