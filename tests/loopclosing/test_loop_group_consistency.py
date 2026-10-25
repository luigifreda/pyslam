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
* A loop candidate is accepted only after its covisibility group is seen in consecutive checks.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyslam.config  # noqa: F401  (sets up the import paths)
from pyslam.loop_closing.loop_closing import LoopGroupConsistencyChecker


class FakeKeyFrame:
    def __init__(self, kid, bad=False):
        self.kid = kid
        self.bad = bad
        self.connected = []

    def is_bad(self):
        return self.bad

    def get_connected_keyframes(self):
        return list(self.connected)

    def __repr__(self):
        return f"KF({self.kid})"


def covisible_chain(first_id, n):
    kfs = [FakeKeyFrame(first_id + i) for i in range(n)]
    for i, kf in enumerate(kfs):
        kf.connected = [k for k in kfs[max(0, i - 1) : i + 2] if k is not kf]
    return kfs


class TestLoopGroupConsistencyChecker(unittest.TestCase):
    def setUp(self):
        self.checker = LoopGroupConsistencyChecker(consistency_threshold=3)
        self.place = covisible_chain(0, 6)
        self.other_place = covisible_chain(100, 6)
        self.current = FakeKeyFrame(1000)

    def check(self, candidates):
        return self.checker.check_candidates(self.current, candidates)

    def test_accepted_after_threshold_consecutive_detections(self):
        # Overlapping covisibility groups count as the same place.
        results = [self.check([self.place[i]]) for i in range(4)]
        self.assertEqual(results, [False, False, False, True])
        self.assertEqual(self.checker.enough_consistent_candidates, [self.place[3]])
        self.assertEqual(self.checker.consistent_groups[0].consistency, 3)

    def test_interruption_resets_consistency(self):
        self.check([self.place[0]])
        self.check([self.place[1]])
        self.assertFalse(self.check([]))
        self.assertEqual(self.checker.consistent_groups, [])
        self.assertFalse(self.check([self.place[2]]))
        self.assertFalse(self.check([self.place[3]]))
        self.assertFalse(self.check([self.place[4]]))
        self.assertTrue(self.check([self.place[5]]))

    def test_disjoint_places_are_not_consistent(self):
        for i in range(4):
            place = self.place if i % 2 == 0 else self.other_place
            self.assertFalse(self.check([place[i]]))

    def test_bad_candidates_are_ignored(self):
        self.check([self.place[0]])
        self.check([self.place[1]])
        self.check([self.place[2]])
        self.place[3].bad = True
        self.assertFalse(self.check([self.place[3]]))
        self.assertEqual(self.checker.consistent_groups, [])

    def test_every_consistent_candidate_is_reported(self):
        for i in range(3):
            self.check([self.place[i]])
        self.assertTrue(self.check([self.place[3], self.place[4], self.other_place[0]]))
        self.assertEqual(self.checker.enough_consistent_candidates, [self.place[3], self.place[4]])

    def test_clear_consistency_groups(self):
        for i in range(3):
            self.check([self.place[i]])
        self.checker.clear_consistency_groups()
        self.assertFalse(self.check([self.place[3]]))


if __name__ == "__main__":
    unittest.main()
