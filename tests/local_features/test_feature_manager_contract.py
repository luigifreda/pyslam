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
* Registry-driven FeatureManager and tracker dispatch: CPU/no-network tests.
*
* The settings test compares against values recorded with the code before the registry
* refactor, so a change in a feature's options is caught even if the code agrees with itself.
"""

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import pyslam.config  # noqa: F401  (sets up the library paths)
from pyslam.config_parameters import Parameters
from pyslam.local_features import feature_tracker as ft
from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_matcher import FeatureMatcherTypes
from pyslam.local_features.feature_types import FeatureDescriptorTypes, FeatureDetectorTypes
from pyslam.visual_models import ModelUnavailable, list_models
from pyslam.visual_models import factory

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ["num_features", "num_levels", "scale_factor", "first_level", "oriented_features",
            "do_keypoints_size_rescaling", "keypoint_size_rescale_factor", "need_color_image",
            "keypoint_filter_type", "need_nms", "use_bock_adaptor", "use_pyramid_adaptor", "pyramid_type",
            "pyramid_do_parallel", "force_multiscale_detect_and_compute", "need_mask_management", "norm_type",
            "max_descriptor_distance", "is_detector_equal_to_descriptor", "orb_params"]


def build(detector, descriptor, **kwargs):
    # FeatureManager prints a lot; keep the test output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        return feature_manager_factory(detector_type=FeatureDetectorTypes[detector],
                                       descriptor_type=FeatureDescriptorTypes[descriptor], **kwargs)


class FeatureManagerSettingsTest(unittest.TestCase):
    def setUp(self):
        # The first constructed manager sets the global max descriptor distance; restore it.
        saved = Parameters.kMaxDescriptorDistance
        self.addCleanup(setattr, Parameters, "kMaxDescriptorDistance", saved)

    def test_classic_settings_match_the_pre_refactor_values(self):
        recorded = json.loads((ROOT / "tests/data/visual_models/feature_manager_settings.json").read_text())["combos"]
        self.assertGreater(len(recorded), 250)
        for combo, expected in recorded.items():
            detector, descriptor = combo.split("/")
            with self.subTest(combo=combo):
                manager = build(detector, descriptor, num_features=1000, scale_factor=1.2,
                                num_levels=8 if detector == "ORB2" else 4)
                actual = {name: repr(getattr(manager, name, None)) for name in SETTINGS}
                actual["detector_class"] = type(manager._feature_detector).__name__
                self.assertEqual(actual, {name: expected[name] for name in actual})

    def test_joint_descriptor_requires_its_own_detector(self):
        joint = [spec.name for spec in list_models("local_feature") if spec.joint]
        self.assertEqual(len(joint), 14)
        for name in joint:
            with self.subTest(descriptor=name):
                # FAST is built first and loads no model; the descriptor check must then fail.
                with self.assertRaisesRegex(ValueError, f"cannot use {name} descriptor without {name} detector"):
                    build("FAST", name)

    def test_missing_optional_package_raises_model_unavailable(self):
        factory.resolve_symbol.cache_clear()
        self.addCleanup(factory.resolve_symbol.cache_clear)
        with patch("pyslam.visual_models.factory.resolve_symbol", side_effect=ImportError("No module named 'torch'")):
            with self.assertRaisesRegex(ModelUnavailable, r"SUPERPOINT.*Dependency group: core.*torch"):
                build("SUPERPOINT", "SUPERPOINT")


class TrackerDispatchTest(unittest.TestCase):
    def create(self, tracker_type):
        with patch.object(ft, "feature_manager_factory", MagicMock()), \
             patch.object(ft, "feature_matcher_factory", MagicMock()) as matcher_factory, \
             contextlib.redirect_stdout(io.StringIO()):
            tracker = ft.feature_tracker_factory(tracker_type=tracker_type,
                                                 detector_type=FeatureDetectorTypes.NONE,
                                                 descriptor_type=FeatureDescriptorTypes.NONE)
        return tracker, matcher_factory

    def test_detector_free_types_use_one_tracker_with_the_same_named_matcher(self):
        for name in ("LOFTR", "MAST3R", "ROMA", "TINY_ROMA"):
            with self.subTest(tracker=name):
                tracker, matcher_factory = self.create(ft.FeatureTrackerTypes[name])
                self.assertIs(type(tracker), ft.DetectorFreeFeatureTracker)
                self.assertEqual(tracker.matcher_type, FeatureMatcherTypes[name])
                self.assertEqual(matcher_factory.call_args.kwargs["matcher_type"], FeatureMatcherTypes[name])
                self.assertIs(matcher_factory.call_args.kwargs["descriptor_type"], FeatureDescriptorTypes.NONE)

    def test_descriptor_and_lk_types(self):
        expected = {"DES_BF": FeatureMatcherTypes.BF, "DES_FLANN": FeatureMatcherTypes.FLANN,
                    "XFEAT": FeatureMatcherTypes.XFEAT, "LIGHTGLUE": FeatureMatcherTypes.LIGHTGLUE}
        for name, matcher_type in expected.items():
            with self.subTest(tracker=name):
                tracker, _ = self.create(ft.FeatureTrackerTypes[name])
                self.assertIs(type(tracker), ft.DescriptorFeatureTracker)
                self.assertEqual(tracker.matcher_type, matcher_type)
        tracker, matcher_factory = self.create(ft.FeatureTrackerTypes.LK)
        self.assertIs(type(tracker), ft.LkFeatureTracker)
        matcher_factory.assert_not_called()

    def test_every_tracker_type_is_dispatched(self):
        for value in ft.FeatureTrackerTypes:
            with self.subTest(tracker=value.name):
                self.assertIsNotNone(self.create(value)[0])

    def test_legacy_tracker_names(self):
        self.assertIs(ft.LoftrFeatureTracker, ft.DetectorFreeFeatureTracker)
        self.assertIs(ft.Mast3rFeatureTracker, ft.DetectorFreeFeatureTracker)


if __name__ == "__main__":
    unittest.main()
