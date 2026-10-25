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
* Map-point and keyframe culling rules of the Python LocalMappingCore (ORB-SLAM2 style).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyslam.config  # noqa: F401  (sets up the import paths)
from pyslam.config_parameters import Parameters
from pyslam.io.dataset_types import SensorType
from pyslam.slam.local_mapping_core import LocalMappingCore


class FakePoint:
    def __init__(self, first_kid=0, found_ratio=1.0, num_obs=10, bad=False):
        self.first_kid = first_kid
        self.found_ratio = found_ratio
        self.num_obs = num_obs
        self.bad = bad
        self.obs = []

    def is_bad(self):
        return self.bad

    def set_bad(self):
        self.bad = True

    def get_found_ratio(self):
        return self.found_ratio

    def num_observations(self):
        return len(self.obs) if self.obs else self.num_obs

    def observations(self):
        return list(self.obs)


class FakeMap:
    def __init__(self):
        self.removed = []

    def remove_point(self, p):
        self.removed.append(p)


class FakeCamera:
    depth_threshold = 4.0


class FakeKeyFrame:
    def __init__(self, kid, num_points, timestamp=None, depth=1.0, octave=0):
        self.kid = kid
        self.id = kid
        self.timestamp = float(kid) if timestamp is None else timestamp
        self.parent = None
        self.camera = FakeCamera()
        self.points = [None] * num_points
        self.depths = np.full(num_points, depth)
        self.octaves = np.full(num_points, octave)
        self.bad = False
        self.covisible = []
        self.fov_center_w = np.array([[float(kid)], [0.0], [0.0]])

    def get_points(self):
        return self.points

    def get_covisible_keyframes(self):
        return list(self.covisible)

    def is_bad(self):
        return self.bad

    def set_bad(self):
        self.bad = True


def make_core(sensor_type, kf_cur):
    core = LocalMappingCore(FakeMap(), sensor_type)
    core.kf_cur = kf_cur
    return core


class TestCullMapPoints(unittest.TestCase):
    def cull(self, points, sensor_type=SensorType.RGBD, current_kid=5):
        core = make_core(sensor_type, FakeKeyFrame(current_kid, 0))
        core.recently_added_points = set(points)
        num_culled = core.cull_map_points()
        return core, num_culled

    def test_low_found_ratio_is_culled(self):
        p = FakePoint(first_kid=5, found_ratio=0.2)
        core, _ = self.cull([p])
        self.assertTrue(p.is_bad())
        self.assertEqual(core.map.removed, [p])
        self.assertEqual(core.recently_added_points, set())

    def test_few_observations_after_two_keyframes_is_culled(self):
        rgbd_weak = FakePoint(first_kid=3, num_obs=3)
        rgbd_young = FakePoint(first_kid=4, num_obs=3)
        core, num_culled = self.cull([rgbd_weak, rgbd_young])
        self.assertTrue(rgbd_weak.is_bad())
        self.assertFalse(rgbd_young.is_bad())
        self.assertEqual(core.recently_added_points, {rgbd_young})
        self.assertEqual(num_culled, 1)

    def test_monocular_needs_fewer_observations(self):
        p = FakePoint(first_kid=3, num_obs=3)
        core, _ = self.cull([p], sensor_type=SensorType.MONOCULAR)
        self.assertFalse(p.is_bad())
        self.assertEqual(core.recently_added_points, {p})

    def test_old_good_points_leave_recent_set(self):
        p = FakePoint(first_kid=2, num_obs=10)
        core, _ = self.cull([p])
        self.assertFalse(p.is_bad())
        self.assertEqual(core.map.removed, [])
        self.assertEqual(core.recently_added_points, set())

    def test_already_bad_points_are_dropped_without_removal(self):
        p = FakePoint(first_kid=5, bad=True)
        core, _ = self.cull([p])
        self.assertEqual(core.map.removed, [])
        self.assertEqual(core.recently_added_points, set())


class TestCullKeyFrames(unittest.TestCase):
    NUM_POINTS = 20

    def setUp(self):
        self.saved_time_distance = Parameters.kKeyframeMaxTimeDistanceInSecForCulling
        Parameters.kKeyframeMaxTimeDistanceInSecForCulling = 0.5
        self.kf_cur = FakeKeyFrame(10, 0)
        self.root = FakeKeyFrame(0, 0)
        self.others = [FakeKeyFrame(20 + i, self.NUM_POINTS) for i in range(3)]

    def tearDown(self):
        Parameters.kKeyframeMaxTimeDistanceInSecForCulling = self.saved_time_distance

    def keyframe_seen_by_others(self, kid, num_shared, other_octave=0, depth=1.0):
        kf = FakeKeyFrame(kid, self.NUM_POINTS, depth=depth)
        kf.parent = self.root
        for i in range(self.NUM_POINTS):
            p = FakePoint()
            p.obs = [(kf, i)]
            if i < num_shared:
                for other in self.others:
                    other.octaves[i] = other_octave
                    p.obs.append((other, i))
            kf.points[i] = p
        self.kf_cur.covisible = [kf]
        return kf

    def cull(self, use_fov_centers=False, max_fov_distance=1.0):
        core = make_core(SensorType.RGBD, self.kf_cur)
        return core.cull_keyframes(use_fov_centers, max_fov_distance)

    def test_redundant_keyframe_is_culled(self):
        kf = self.keyframe_seen_by_others(5, num_shared=self.NUM_POINTS)
        self.assertEqual(self.cull(), 1)
        self.assertTrue(kf.is_bad())

    def test_keyframe_below_redundancy_ratio_is_kept(self):
        kf = self.keyframe_seen_by_others(5, num_shared=int(0.9 * self.NUM_POINTS))
        self.assertEqual(self.cull(), 0)
        self.assertFalse(kf.is_bad())

    def test_coarser_observations_do_not_count(self):
        kf = self.keyframe_seen_by_others(5, num_shared=self.NUM_POINTS, other_octave=2)
        self.assertEqual(self.cull(), 0)
        self.assertFalse(kf.is_bad())

    def test_first_keyframe_is_never_culled(self):
        kf = self.keyframe_seen_by_others(0, num_shared=self.NUM_POINTS)
        kf.parent = FakeKeyFrame(-10, 0)
        self.assertEqual(self.cull(), 0)

    def test_keyframe_close_in_time_to_parent_is_kept(self):
        kf = self.keyframe_seen_by_others(5, num_shared=self.NUM_POINTS)
        kf.parent = FakeKeyFrame(4, 0, timestamp=kf.timestamp - 0.1)
        self.assertEqual(self.cull(), 0)

    def test_far_points_are_not_counted(self):
        kf = self.keyframe_seen_by_others(5, num_shared=self.NUM_POINTS, depth=10.0)
        self.assertEqual(self.cull(), 0)
        self.assertFalse(kf.is_bad())

    def test_fov_center_gap_prevents_culling(self):
        kf = self.keyframe_seen_by_others(5, num_shared=self.NUM_POINTS)
        far = FakeKeyFrame(50, 0)
        self.kf_cur.covisible = [kf, far]
        self.assertEqual(self.cull(use_fov_centers=True, max_fov_distance=1.0), 0)
        self.assertEqual(self.cull(use_fov_centers=False), 1)


if __name__ == "__main__":
    unittest.main()
