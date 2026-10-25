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
* Fusion of per-view semantics into a map-point semantic descriptor.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyslam.config  # noqa: F401  (sets up the import paths)
from pyslam.semantics.semantic_fusion_methods import SemanticFusionMethods
from pyslam.semantics.semantic_types import SemanticFeatureType


class TestSemanticFusionMethods(unittest.TestCase):
    def test_count_labels_returns_majority(self):
        self.assertEqual(SemanticFusionMethods.count_labels(np.array([3, 1, 3, 2, 3, 1])), 3)

    def test_count_labels_tie_picks_smallest_label(self):
        self.assertEqual(SemanticFusionMethods.count_labels([4, 2, 4, 2]), 2)

    def test_bayesian_fusion_matches_iterative_update(self):
        rng = np.random.default_rng(0)
        probs = rng.random((5, 4))
        probs /= probs.sum(axis=1, keepdims=True)
        expected = np.full(4, 0.25)
        for p in probs:
            expected = expected * p
            expected /= expected.sum()
        fused = SemanticFusionMethods.bayesian_fusion(list(probs))
        np.testing.assert_allclose(fused, expected)
        np.testing.assert_allclose(SemanticFusionMethods.bayesian_fusion(probs), expected)
        self.assertAlmostEqual(fused.sum(), 1.0)

    def test_bayesian_fusion_sharpens_agreeing_views(self):
        view = np.array([0.6, 0.3, 0.1])
        fused = SemanticFusionMethods.bayesian_fusion([view, view, view])
        self.assertEqual(fused.argmax(), 0)
        self.assertGreater(fused[0], view[0])

    def test_bayesian_fusion_does_not_modify_input(self):
        probs = np.array([[0.5, 0.5], [0.9, 0.1]])
        before = probs.copy()
        SemanticFusionMethods.bayesian_fusion(probs)
        np.testing.assert_array_equal(probs, before)

    def test_average_fusion(self):
        features = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        np.testing.assert_allclose(SemanticFusionMethods.average_fusion(features), [2.0, 2.0, 2.0])

    def test_method_selection_by_feature_type(self):
        get = SemanticFusionMethods.get_semantic_fusion_method
        self.assertIs(get(SemanticFeatureType.LABEL), SemanticFusionMethods.count_labels)
        self.assertIs(
            get(SemanticFeatureType.PROBABILITY_VECTOR), SemanticFusionMethods.bayesian_fusion
        )
        self.assertIs(get(SemanticFeatureType.FEATURE_VECTOR), SemanticFusionMethods.average_fusion)
        with self.assertRaises(ValueError):
            get(SemanticFeatureType.NONE)


if __name__ == "__main__":
    unittest.main()
