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
* Bound the optimized poses while retaining fixed observers of local points.
"""

import unittest
import camera_test_support
from pyslam.config_parameters import Parameters
from pyslam.slam.map import LocalCovisibilityMap


class Keyframe:
    def __init__(self, bad=False):
        self.bad, self.neighbors, self.points = bad, [], []

    def is_bad(self):
        return self.bad

    def get_covisible_keyframes(self):
        return self.neighbors

    def get_matched_good_points(self):
        return self.points


class Point:
    def __init__(self, viewers):
        self.viewers = viewers

    def keyframes(self):
        return self.viewers


class TestLocalMapLimits(unittest.TestCase):
    def test_strong_neighbors_bounded_but_other_observers_remain_fixed(self):
        limit = Parameters.kMaxNumOfKeyframesInLocalMap
        current = Keyframe()
        current.neighbors = [Keyframe(bad=i == 0) for i in range(limit + 20)]
        weak = current.neighbors[-1]
        point = Point([current, weak])
        current.points = [point]
        local, points, fixed = LocalCovisibilityMap().update(current)
        self.assertEqual(len(local), limit)
        self.assertIn(current, local)
        self.assertIn(current.neighbors[1], local)
        self.assertNotIn(current.neighbors[0], local)
        self.assertNotIn(weak, local)
        self.assertEqual(points, {point})
        self.assertEqual(fixed, {weak})

    def test_small_neighborhood_preserves_all_good_keyframes(self):
        current = Keyframe()
        current.neighbors = [Keyframe(), Keyframe(bad=True)]
        local = LocalCovisibilityMap().update_keyframes(current)
        self.assertEqual(set(local), {current, current.neighbors[0]})


if __name__ == '__main__':
    unittest.main()
