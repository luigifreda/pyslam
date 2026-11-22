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
* Detector-free matcher pipeline (resize, sub-pixel rescale, filtering) with a fake model.
"""

import os
import unittest

import numpy as np

import pyslam.config  # noqa: F401  (native library paths)
from pyslam.local_features.feature_matcher import FeatureMatcherTypes
from pyslam.local_features.feature_matcher_detector_free import (
    DetectorFreeMatcher,
    keypoints_from_points,
    resized_shape,
)


class FakeShiftMatcher(DetectorFreeMatcher):
    """Returns a fixed grid in resized coordinates, shifted by (dx, dy) in the second image."""

    def __init__(self, shift=(4.25, 2.5), conf=None, **kwargs):
        super().__init__(matcher_type=FeatureMatcherTypes.LOFTR, device="cpu", **kwargs)
        self.shift = np.float32(shift)
        self.conf = conf
        self.seen_shapes = []

    def _infer(self, t0, t1):
        import torch

        self.seen_shapes.append((tuple(t0.shape), tuple(t1.shape)))
        height, width = t0.shape[-2:]
        xs, ys = np.meshgrid(np.linspace(8.5, width - 9.5, 6), np.linspace(8.5, height - 9.5, 4))
        k0 = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        conf = self.conf if self.conf is not None else np.linspace(0.3, 0.9, len(k0)).astype(np.float32)
        return torch.from_numpy(k0), torch.from_numpy(k0 + self.shift), torch.from_numpy(conf)


class DetectorFreePipelineTest(unittest.TestCase):
    def test_resized_shape_preserves_aspect_and_divisibility(self):
        self.assertEqual(resized_shape(376, 1241, 1024, 32), (288, 1024))
        self.assertEqual(resized_shape(480, 640, 1024, 32), (480, 640))  # never upsampled
        self.assertEqual(resized_shape(480, 752, None, 32), (480, 736))
        self.assertEqual(resized_shape(10, 10, 1024, 32), (32, 32))

    def test_subpixel_rescaling_to_original_coordinates(self):
        matcher = FakeShiftMatcher(max_side=620)  # KITTI 1241x376 -> (160, 608)
        image = np.zeros((376, 1241, 3), np.uint8)
        k0, k1, conf = matcher.match_points(image, image)
        self.assertEqual(matcher.seen_shapes[0], ((1, 1, 160, 608), (1, 1, 160, 608)))
        sx, sy = 1241 / 608, 376 / 160
        np.testing.assert_allclose(k1 - k0, np.tile([4.25 * sx, 2.5 * sy], (len(k0), 1)), rtol=1e-5)
        self.assertTrue(np.any(np.abs(k0 - np.round(k0)) > 1e-3), "keypoints must stay sub-pixel")
        self.assertEqual(k0.dtype, np.float32)

    def test_confidence_bounds_and_cap(self):
        conf = np.full(24, 0.9, np.float32)
        conf[:5] = 0.1
        matcher = FakeShiftMatcher(conf=conf, conf_threshold=0.2, max_num_matches=10)
        image = np.zeros((480, 640), np.uint8)
        k0, _, kept = matcher.match_points(image, image)
        self.assertEqual(len(k0), 10)
        self.assertTrue(np.all(kept >= 0.2))
        out_of_bounds = FakeShiftMatcher(shift=(5000, 0))
        self.assertEqual(len(out_of_bounds.match_points(image, image)[0]), 0)

    def test_matching_result_contract(self):
        matcher = FakeShiftMatcher()
        image = np.zeros((480, 640), np.uint8)
        result = matcher.match(image, image, None, None)
        self.assertEqual(len(result.kps1), len(result.idxs1))
        self.assertEqual(result.idxs1.dtype, np.int32)
        self.assertAlmostEqual(result.kps1[0].pt[0], float(matcher.match_points(image, image)[0][0, 0]), 4)
        self.assertIsNone(result.des1)
        self.assertEqual(len(keypoints_from_points(np.empty((0, 2)))), 0)

    def test_row_matching_uses_original_coordinates(self):
        matcher = FakeShiftMatcher(shift=(-3.0, 0.0))  # pure horizontal disparity
        image = np.zeros((480, 640), np.uint8)
        result = matcher.match(image, image, None, None, row_matching=True, max_disparity=100)
        self.assertEqual(len(result.idxs1), len(matcher.match_points(image, image)[0]))

class FakeRoMa:
    """Mimics romatch's match/sample/to_pixel_coordinates with a known pixel shift."""

    def __init__(self, shift):
        self.shift = np.float32(shift)

    def match(self, img1, img2, **kwargs):
        return (img1, img2), None

    def sample(self, warp, certainty, num):
        import torch

        rng = np.random.default_rng(0)
        pts = rng.uniform([20, 20], [300, 220], (num, 2)).astype(np.float32)
        return torch.from_numpy(np.concatenate([pts, pts + self.shift], 1)), torch.full((num,), 0.9)

    def to_pixel_coordinates(self, matches, h1, w1, h2, w2):
        return matches[:, :2], matches[:, 2:]


class RoMaPipelineTest(unittest.TestCase):
    def make(self, shift=(3.5, -1.25), **kwargs):
        from pyslam.local_features import feature_matcher_detector_free as dfm

        matcher = dfm.RoMaMatcher.__new__(dfm.RoMaMatcher)
        dfm.DetectorFreeMatcher.__init__(matcher, matcher_type=FeatureMatcherTypes.ROMA, max_side=None,
                                         device="cpu", conf_threshold=0.0, **kwargs)
        matcher.num_samples = kwargs.get("max_num_matches") or 100
        matcher.matcher = FakeRoMa(shift)
        matcher._match_warp = lambda a, b: matcher.matcher.match(a, b)
        return matcher

    def test_original_coordinates_are_kept_and_filtered(self):
        matcher = self.make(max_num_matches=50)
        image = np.zeros((240, 320, 3), np.uint8)
        k0, k1, conf = matcher.match_points(image, image)
        self.assertEqual(len(k0), 50)
        np.testing.assert_allclose(k1 - k0, np.tile([3.5, -1.25], (50, 1)), atol=1e-5)
        result = matcher.match(image, image, None, None)
        self.assertEqual(len(result.kps1), 50)

    def test_loguru_shim_only_when_missing(self):
        import sys
        import types
        from unittest.mock import patch
        from pyslam.local_features import feature_matcher_detector_free as dfm

        fake_romatch = types.ModuleType("romatch")
        # Save/restore only the two touched entries: patch.dict(sys.modules) would also drop
        # every module imported meanwhile (re-importing torchvision then fails).
        saved = {name: sys.modules.get(name) for name in ("romatch", "loguru")}
        try:
            sys.modules["romatch"] = fake_romatch
            sys.modules["loguru"] = None  # makes "import loguru" fail like a missing package
            with patch("pyslam.loop_closing.global_feature_torchhub.github_snapshot", return_value="/nonexistent"):
                self.assertIs(dfm._import_romatch(), fake_romatch)
            self.assertTrue(hasattr(sys.modules["loguru"], "logger"))
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            if "/nonexistent" in sys.path:
                sys.path.remove("/nonexistent")


def _tiny_roma_cached():
    import os
    import torch

    return os.path.isfile(os.path.join(torch.hub.get_dir(), "checkpoints", "tiny_roma_v1_outdoor.pth"))


@unittest.skipUnless(_tiny_roma_cached(), "tiny-RoMa weights not cached (no download in tests)")
class TinyRoMaRealTest(unittest.TestCase):
    def test_recovers_a_known_translation(self):
        # Guards the coordinate pipeline (scaling, pairing): the robust displacement estimate
        # must match the applied sub-pixel shift. Per-match accuracy is measured on real
        # frames in docs/visual_features.md, not on synthetic texture.
        import cv2
        from pyslam.local_features.feature_matcher import feature_matcher_factory

        os.environ.setdefault("PYSLAM_OFFLINE", "1")
        rng = np.random.default_rng(4)
        texture = np.zeros((240, 320), np.float32)
        for scale in (1, 2, 4, 8):  # multi-scale noise texture
            noise = rng.normal(0, 1, (240 // scale + 1, 320 // scale + 1)).astype(np.float32)
            texture += cv2.resize(noise, (320, 240), interpolation=cv2.INTER_CUBIC) * scale
        image = cv2.cvtColor(cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                             cv2.COLOR_GRAY2RGB)
        other = cv2.warpAffine(image, np.float32([[1, 0, 5.5], [0, 1, 2.25]]), (320, 240),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        matcher = feature_matcher_factory(matcher_type=FeatureMatcherTypes.TINY_ROMA, max_num_matches=500)
        k0, k1, _ = matcher.match_points(image, other)
        self.assertGreater(len(k0), 200)
        np.testing.assert_allclose(np.median(k1 - k0, axis=0), [5.5, 2.25], atol=0.5)


if __name__ == "__main__":
    unittest.main()
