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
* torch.hub global descriptors (SALAD, BoQ) with fake models, and loop-detector wiring.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

import pyslam.config  # noqa: F401  (native library paths)
from pyslam.loop_closing import global_feature_torchhub as gth


class FakeAggregatedModel(torch.nn.Module):
    """Mean color per channel, tiled to `dim`; optionally returns (descriptor, attentions) like BoQ."""

    def __init__(self, dim, tuple_output=False):
        super().__init__()
        self.dim, self.tuple_output = dim, tuple_output

    def forward(self, x):
        descriptor = x.mean(dim=(2, 3)).repeat(1, self.dim // 3 + 1)[:, : self.dim] + 1.0
        return (descriptor, None) if self.tuple_output else descriptor


def fake_extractor(spec, tuple_output=False, dim=None):
    with patch.object(gth.GlobalFeatureTorchHub, "_build_model",
                      lambda self: FakeAggregatedModel(dim or spec.dim, tuple_output)):
        return gth.GlobalFeatureTorchHub(spec, device="cpu")


class GlobalFeatureTorchHubTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.images = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(5)]

    def test_batch_equals_single_and_descriptors_are_normalized(self):
        for spec, tuple_output in ((gth.SALAD, False), (gth.BOQ, True)):
            with self.subTest(spec=spec.name):
                extractor = fake_extractor(spec, tuple_output)
                batch = extractor.compute_features(self.images)
                self.assertEqual(batch.shape, (5, spec.dim))
                self.assertEqual(batch.dtype, np.float32)
                np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, rtol=1e-5)
                single = extractor.compute_features_step(self.images[3])
                np.testing.assert_allclose(single[0], batch[3], rtol=1e-5, atol=1e-6)

    def test_grayscale_input_and_dimension_check(self):
        extractor = fake_extractor(gth.SALAD)
        self.assertEqual(extractor.compute_features_step(self.images[0][:, :, 0]).shape, (1, gth.SALAD.dim))
        wrong = fake_extractor(gth.SALAD, dim=16)
        with self.assertRaisesRegex(RuntimeError, "descriptor dim"):
            wrong.compute_features_step(self.images[0])

    def test_specs_are_pinned_to_commits(self):
        for spec in (gth.SALAD, gth.BOQ):
            repo, _, ref = spec.repo.partition(":")
            self.assertRegex(ref, r"^[0-9a-f]{40}$", spec.name)
            self.assertEqual(spec.image_size, (322, 322))  # multiple of the DINOv2 patch size 14

    def test_vendor_files_load_under_private_names(self):
        import os
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "models.py")
            with open(path, "w") as stream:
                stream.write("VALUE = 7\n")
            module = gth.load_module_from_file("pyslam_vendor_test_models", path)
            self.assertEqual(module.VALUE, 7)
            self.assertIs(sys.modules["pyslam_vendor_test_models"], module)
            self.assertNotIn("models", sys.modules)  # the generic vendor name is not registered
            self.assertNotIn(folder, sys.path)
            sys.modules.pop("pyslam_vendor_test_models", None)


class LoopDetectorWiringTest(unittest.TestCase):
    def test_vpr_factory_builds_loop_detectors_without_loading_models(self):
        from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs, loop_detector_factory

        for preset, expected in (("MEGALOC", "LoopDetectorMegaloc"), ("SALAD", "LoopDetectorSalad"),
                                 ("BOQ", "LoopDetectorBoQ")):
            with self.subTest(preset=preset):
                detector = loop_detector_factory(**getattr(LoopDetectorConfigs, preset))
                self.assertEqual(type(detector).__name__, expected)  # MEGALOC used to raise TypeError
                self.assertIsNone(detector.global_feature_extractor)  # built later, in the process

    def test_megaloc_dimension_matches_model_output(self):
        from pyslam.loop_closing.global_feature_megaloc import GlobalFeatureMegaloc
        import inspect

        self.assertIn("self.dim = 8448", inspect.getsource(GlobalFeatureMegaloc.__init__))


if __name__ == "__main__":
    unittest.main()
