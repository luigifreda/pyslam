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
* FeatureMatcher.match(): shared validation in the base class, per-matcher _match_impl.
"""

import unittest

import cv2
import numpy as np

import pyslam.config  # noqa: F401  (native library paths)
from pyslam.local_features import feature_matcher as fm


class RecordingMatcher(fm.FeatureMatcher):
    def __init__(self, detector_free=False):
        super().__init__(matcher_type=fm.FeatureMatcherTypes.BF)
        self.detector_free = detector_free
        self.calls = []

    def _match_impl(self, result, img1, img2, des1, des2, kps1, kps2, ratio_test, row_matching,
                    max_disparity, data):
        self.calls.append(ratio_test)
        result.idxs1 = result.idxs2 = np.array([0], dtype=np.int32)
        return result


class MatcherDispatchTest(unittest.TestCase):
    def test_descriptor_matchers_short_circuit_on_missing_or_empty_descriptors(self):
        matcher = RecordingMatcher()
        for des in (None, np.empty((0, 32), np.uint8)):
            result = matcher.match(None, None, des, des)
            self.assertEqual(len(result.idxs1), 0)
            self.assertEqual(result.idxs1.dtype, np.int32)
        self.assertEqual(matcher.calls, [])

    def test_detector_free_matchers_reach_their_implementation_without_descriptors(self):
        matcher = RecordingMatcher(detector_free=True)
        matcher.match(np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8), None, None)
        self.assertEqual(matcher.calls, [matcher.ratio_test])  # default ratio test forwarded

    def test_explicit_ratio_test_is_forwarded(self):
        matcher = RecordingMatcher()
        des = np.zeros((3, 32), np.uint8)
        matcher.match(None, None, des, des, ratio_test=0.5)
        self.assertEqual(matcher.calls, [0.5])

    def test_shipped_matchers_override_and_flag_correctly(self):
        overrides = {fm.XFeatMatcher, fm.LightGlueMatcher, fm.LoFTRMatcher, fm.Mast3RMatcher}
        for cls in overrides:
            with self.subTest(cls=cls.__name__):
                self.assertIsNot(cls._match_impl, fm.FeatureMatcher._match_impl)
        for cls in (fm.BfFeatureMatcher, fm.FlannFeatureMatcher):
            self.assertIs(cls._match_impl, fm.FeatureMatcher._match_impl)
        self.assertEqual({cls for cls in overrides | {fm.BfFeatureMatcher} if cls.detector_free},
                         {fm.LoFTRMatcher, fm.Mast3RMatcher})

    def test_bf_matcher_matches_identical_orb_descriptors(self):
        rng = np.random.default_rng(5)
        des = rng.integers(0, 256, (50, 32), dtype=np.uint8)
        matcher = fm.feature_matcher_factory(norm_type=cv2.NORM_HAMMING, matcher_type=fm.FeatureMatcherTypes.BF,
                                             ratio_test=0.8)
        result = matcher.match(None, None, des, des[::-1].copy())
        self.assertEqual(len(result.idxs1), 50)
        np.testing.assert_array_equal(result.idxs2, 49 - result.idxs1)


if __name__ == "__main__":
    unittest.main()
