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
* Opt-in accelerations: GPU NN matcher, batched LightGlue, torch.compile (all default off).
"""

import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import pyslam.config  # noqa: F401  (native library paths)
from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_matcher import MatcherUtils
from pyslam.local_features.feature_matcher_gpu import gpu_ratio_one_to_one, one_to_one_from_knn


class OptionDefaultsTest(unittest.TestCase):
    def test_all_accelerations_are_off_by_default(self):
        environment = ("PYSLAM_GPU_NN_MATCHER", "PYSLAM_LIGHTGLUE_BATCHED", "PYSLAM_TORCH_COMPILE")
        if not any(os.environ.get(name) for name in environment):
            self.assertFalse(Parameters.kFeatureMatchUseGpuNN)
            self.assertFalse(Parameters.kLightGlueBatchedMatching)
            self.assertFalse(Parameters.kTorchCompile)


class GpuNNMatcherTest(unittest.TestCase):
    def reference(self, values, indices, ratio):
        # The production path: C++ pyslam_utils.good_matches_one_to_one on cv2.DMatch pairs.
        # (MatcherUtils.goodMatchesOneToOne_py does not update the stored distance on
        # replacement; it is not used by the matchers.)
        matches = [[cv2.DMatch(q, int(indices[q, 0]), float(values[q, 0])),
                    cv2.DMatch(q, int(indices[q, 1]), float(values[q, 1]))] for q in range(len(values))]
        return MatcherUtils.goodMatchesOneToOne(matches, None, None, ratio)

    def test_one_to_one_semantics_match_the_reference_implementation(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            n, m = 200, 60  # few train indices: many duplicates to resolve
            values = np.sort(rng.integers(1, 30, (n, 2)).astype(np.float32), axis=1)  # integer distances: ties
            indices = np.stack([rng.integers(0, m, n), rng.integers(0, m, n)], 1)
            expected = self.reference(values, indices, 0.8)
            actual = one_to_one_from_knn(values, indices, 0.8)
            with self.subTest(trial=trial):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

    def test_matches_opencv_brute_force_on_float_descriptors(self):
        rng = np.random.default_rng(1)
        des2 = rng.normal(size=(300, 64)).astype(np.float32)
        des1 = des2[rng.permutation(300)[:200]] + rng.normal(scale=0.05, size=(200, 64)).astype(np.float32)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
        expected = MatcherUtils.goodMatchesOneToOne(knn, des1, des2, 0.7)
        actual = gpu_ratio_one_to_one(des1, des2, 0.7, device="cpu")
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(len(gpu_ratio_one_to_one(des1, des2[:1], 0.7, device="cpu")[0]), 0)


class TorchCompileHelperTest(unittest.TestCase):
    def test_disabled_returns_the_function_unchanged(self):
        from pyslam.utilities.torch_compile import maybe_compile

        fn = lambda x: x + 1  # noqa: E731
        with patch.object(Parameters, "kTorchCompile", False):
            self.assertIs(maybe_compile(fn), fn)

    def test_missing_backend_and_compile_failure_fall_back_to_eager(self):
        import torch
        from pyslam.utilities import torch_compile

        fn = lambda x: x * 2  # noqa: E731
        with patch.object(Parameters, "kTorchCompile", True), \
             patch.object(Parameters, "kTorchCompileBackend", "tensorrt"), \
             patch.object(torch_compile.importlib.util, "find_spec", return_value=None):
            with self.assertWarns(UserWarning):
                self.assertIs(torch_compile.maybe_compile(fn), fn)

        def broken_compile(function, backend=None):
            def raise_on_call(*args, **kwargs):
                raise RuntimeError("compile failed")
            return raise_on_call

        with patch.object(Parameters, "kTorchCompile", True), \
             patch.object(Parameters, "kTorchCompileBackend", "inductor"), \
             patch.object(torch, "compile", broken_compile):
            wrapped = torch_compile.maybe_compile(fn, "test")
            with self.assertWarns(UserWarning):
                self.assertEqual(wrapped(3), 6)
            self.assertEqual(wrapped(4), 8)  # stays in eager mode


def _lightglue_superpoint_cached():
    import torch

    return os.path.isfile(os.path.join(torch.hub.get_dir(), "checkpoints", "superpoint_lightglue_v0-1_arxiv.pth"))


@unittest.skipUnless(_lightglue_superpoint_cached(), "LightGlue SuperPoint weights not cached")
class BatchedLightGlueTest(unittest.TestCase):
    def test_batched_matches_agree_with_single_pair_matching(self):
        from pyslam.local_features.feature_matcher import FeatureMatcherTypes, feature_matcher_factory
        from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes

        matcher = feature_matcher_factory(norm_type=cv2.NORM_L2, matcher_type=FeatureMatcherTypes.LIGHTGLUE,
                                          detector_type=FeatureDetectorTypes.SUPERPOINT,
                                          descriptor_type=FeatureDescriptorTypes.SUPERPOINT)
        rng = np.random.default_rng(3)
        image = np.zeros((240, 320, 3), np.uint8)
        kps0 = rng.uniform([0, 0], [320, 240], (150, 2)).astype(np.float32)
        des0 = rng.normal(size=(150, 256)).astype(np.float32)
        des0 /= np.linalg.norm(des0, axis=1, keepdims=True)
        others = []
        for n in (150, 90, 120):  # different keypoint counts per pair
            keep = rng.permutation(150)[:n]
            des = des0[keep] + rng.normal(scale=0.02, size=(n, 256)).astype(np.float32)
            others.append((image, des / np.linalg.norm(des, axis=1, keepdims=True), kps0[keep] + 2.0))
        batched = matcher.match_frames_batch(image, des0, kps0, others)
        for (img2, des2, kps2), (b1, b2) in zip(others, batched):
            single = matcher.match(image, img2, des0, des2, kps1=kps0, kps2=kps2)
            s = set(zip(np.asarray(single.idxs1).tolist(), np.asarray(single.idxs2).tolist()))
            b = set(zip(b1.tolist(), b2.tolist()))
            self.assertGreater(len(s), 0)
            self.assertGreaterEqual(len(s & b) / len(s), 0.95)


if __name__ == "__main__":
    unittest.main()
