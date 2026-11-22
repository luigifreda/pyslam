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
* Golden compatibility contract captured before the visual-model integration work.
"""

import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarking"))
from inventory_visual_models import contract_snapshot


class LegacyContractTest(unittest.TestCase):
    def test_vendor_patches_and_installer_pins_are_preserved(self):
        expected = json.loads((ROOT / "tests/data/visual_models/legacy_install_contract.json").read_text())
        for path, digest in expected.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_enum_and_feature_info_contract_is_unchanged(self):
        # Append-only: every recorded enum name/value and FeatureInfo entry must be
        # identical. New members are allowed only with values above the recorded maximum
        # (saved maps and the C++ core depend on the existing values).
        expected = json.loads((ROOT / "tests/data/visual_models/legacy_contract.json").read_text())
        actual = contract_snapshot()
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(set(actual["enums"]), set(expected["enums"]))
        for enum_name, members in expected["enums"].items():
            current = actual["enums"][enum_name]
            with self.subTest(enum=enum_name):
                self.assertEqual({name: current.get(name) for name in members}, members)
                added = {name: value for name, value in current.items() if name not in members}
                self.assertTrue(all(value > max(members.values()) for value in added.values()),
                                f"new {enum_name} members must be appended: {added}")
                self.assertEqual(len(set(current.values())), len(current), "duplicate enum values")
        for table, entries in expected["FeatureInfo"].items():
            current = actual["FeatureInfo"][table]
            with self.subTest(table=table):
                if isinstance(entries, dict):
                    self.assertEqual({key: current.get(key) for key in entries}, entries)
                else:  # ordered list: the recorded entries stay first, new ones are appended
                    self.assertEqual(current[: len(entries)], entries)
    def test_registry_covers_each_local_and_matcher_enum_without_changing_tables(self):
        from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes
        from pyslam.visual_models.model_registry import get_spec, spec_for_enum
        snapshot = contract_snapshot()
        for cls in (FeatureDetectorTypes, FeatureDescriptorTypes):
            for value in cls:
                spec = spec_for_enum(value)
                self.assertEqual(spec.name, value.name)
                self.assertEqual(spec.kind, "local_feature")
                values = spec.feature_info()
                if cls is FeatureDescriptorTypes:
                    from pyslam.local_features.feature_types import FeatureInfo
                    self.assertEqual(values["norm_type"], FeatureInfo.norm_type[value])
                    self.assertEqual(values["max_descriptor_distance"], FeatureInfo.max_descriptor_distance[value])
        for name in snapshot["enums"]["FeatureMatcherTypes"]:
            self.assertEqual(get_spec(name, "pair_matcher").name, name)
        for enum_name in ("GlobalDescriptorType", "LocalDescriptorAggregationType"):
            for name in snapshot["enums"][enum_name]:
                self.assertEqual(get_spec(name, "global_descriptor").name, name.replace("_", "-"))


if __name__ == "__main__":
    unittest.main()
