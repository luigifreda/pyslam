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
* CPU/no-network tests for metadata, lazy constructor resolution and weight integrity.
"""

import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pyslam.visual_models import get_local_feature, get_spec, list_models, validate_frontend
from pyslam.visual_models.factory import ModelUnavailable, resolve_symbol
from pyslam.visual_models.weights import (ChecksumError, WeightResolver, WeightSource,
                                         WeightsUnavailable, default_cache, registered_weights)


class RegistryTest(unittest.TestCase):
    def test_generated_model_documentation_matches_registry(self):
        from pyslam.visual_models.__main__ import markdown_table
        path = Path(__file__).resolve().parents[2] / "docs/visual_features.md"
        contents = path.read_text().split("<!-- BEGIN GENERATED VISUAL MODEL TABLE -->")[1]
        actual = contents.split("<!-- END GENERATED VISUAL MODEL TABLE -->")[0].strip()
        self.assertEqual(actual, markdown_table())

    def tearDown(self):
        resolve_symbol.cache_clear()

    def test_catalog_import_does_not_import_any_runtime_or_vendor(self):
        code = """
import sys
import pyslam.visual_models as vm
assert len(vm.list_models()) > 40
for name in ('torch', 'tensorflow', 'kornia', 'cv2', 'lightglue', 'pyslam.config', 'pyslam.local_features.feature_manager'):
    assert name not in sys.modules, name
"""
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_feature_manager_import_is_lazy(self):
        # Importing the manager must not import optional learned wrappers (TF/Keras,
        # kornia, LightGlue) nor put vendor repos on sys.path; constructing does.
        code = """
import sys
import pyslam.config
import pyslam.local_features.feature_manager as fm
import pyslam.local_features.feature_tracker
import pyslam.local_features.feature_matcher as matcher
for name in ('tensorflow', 'keras', 'kornia', 'lightglue', 'trimesh'):
    assert name not in sys.modules, name
assert matcher.K is None and matcher.KF is None
assert matcher.LightGlue is not None and 'lightglue' in sys.modules  # compat attribute loads on demand
for vendor in ('thirdparty/lfnet', 'thirdparty/contextdesc', 'thirdparty/geodesc', 'thirdparty/superpoint'):
    assert not any(path.rstrip('/').endswith(vendor) for path in sys.path), vendor
assert fm.Orbslam2Feature2D is fm._wrapper('ORB2') is not None
assert fm.XfeatFeature2D is fm._wrapper('XFEAT')
try:
    fm.NotAWrapper
except AttributeError:
    pass
else:
    raise AssertionError('unknown attribute resolved')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True,
                                timeout=120, env=env, cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])

    def test_registry_wrappers_point_to_existing_classes(self):
        # FeatureManager and the installer resolve classes only through the registry.
        import re
        root = Path(__file__).resolve().parents[2]
        for spec in list_models():
            if not (spec.wrapper or "").startswith("pyslam."):
                continue
            module, symbol = spec.wrapper.split(":")
            with self.subTest(name=spec.name, kind=spec.kind):
                source = (root / (module.replace(".", "/") + ".py")).read_text()
                self.assertRegex(source, rf"(?m)^class {symbol}\b")
        manager = (root / "pyslam/local_features/feature_manager.py").read_text()
        used = set(re.findall(r'_wrapper\("([A-Z0-9_]+)"\)', manager))
        self.assertGreater(len(used), 40)
        for name in used:
            with self.subTest(name=name):
                self.assertIsNotNone(get_spec(name).wrapper)

    def test_descriptor_metric_lives_in_the_registry(self):
        from pyslam.local_features.feature_types import FeatureDescriptorTypes, FeatureInfo
        for value in FeatureDescriptorTypes:
            spec = get_spec(value.name)
            with self.subTest(name=value.name):
                self.assertEqual(FeatureInfo.norm_type[value], spec.feature_info()["norm_type"])
                self.assertEqual(FeatureInfo.max_descriptor_distance[value], spec.max_descriptor_distance)
        self.assertTrue(get_spec("SUPERPOINT").joint)
        self.assertFalse(get_spec("ORB").joint)

    def test_constructors_are_lazy_but_stateful_instances_are_not_shared(self):
        with patch("pyslam.visual_models.factory.importlib.import_module") as importer:
            importer.return_value = SimpleNamespace(SIFT_create=lambda **kwargs: dict(kwargs))
            first = get_local_feature("SIFT", nfeatures=7)
            second = get_local_feature("SIFT", nfeatures=7)
            self.assertEqual(first, {"nfeatures": 7})
            self.assertIsNot(first, second)
            importer.assert_called_once_with("cv2")

    def test_unselected_dependency_does_not_fail_and_selected_error_is_actionable(self):
        with patch("pyslam.visual_models.factory.importlib.import_module", side_effect=ImportError("missing torch")):
            self.assertTrue(list_models())
            with self.assertRaisesRegex(ModelUnavailable, "SUPERPOINT.*core.*missing torch"):
                get_local_feature("SUPERPOINT")

    def test_dependency_imported_inside_constructor_has_model_context(self):
        constructor = SimpleNamespace(SIFT_create=lambda **kwargs: unavailable_constructor())
        with patch("pyslam.visual_models.factory.importlib.import_module", return_value=constructor):
            with self.assertRaisesRegex(ModelUnavailable, "SIFT.*missing optional package"):
                get_local_feature("SIFT")

    def test_constructor_bugs_are_not_reported_as_unavailable(self):
        # Only import failures mean "optional integration unavailable"; an AttributeError
        # raised while constructing is a wrapper bug and must propagate unchanged.
        def broken(**kwargs):
            raise AttributeError("wrapper bug")
        with patch("pyslam.visual_models.factory.importlib.import_module",
                   return_value=SimpleNamespace(SIFT_create=broken)):
            with self.assertRaisesRegex(AttributeError, "wrapper bug") as context:
                get_local_feature("SIFT")
            self.assertNotIsInstance(context.exception, ModelUnavailable)
        resolve_symbol.cache_clear()
        with patch("pyslam.visual_models.factory.importlib.import_module",
                   return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ModelUnavailable, "SIFT"):
                get_local_feature("SIFT")  # missing symbol is still "unavailable"

    def test_vpr_vendor_path_is_registered_once(self):
        from pyslam.visual_models import factory
        factory._set_lib_once.cache_clear()
        self.addCleanup(factory._set_lib_once.cache_clear)
        fake_config = SimpleNamespace(cfg=SimpleNamespace(set_lib=unittest.mock.MagicMock()))
        with patch.dict(sys.modules, {"pyslam.config": fake_config}), \
             patch("pyslam.visual_models.factory.resolve_symbol", return_value=lambda **kwargs: object()):
            factory.get_global_descriptor("COSPLACE")
            factory.get_global_descriptor("EIGENPLACES")
        fake_config.cfg.set_lib.assert_called_once_with("vpr", prepend=True)

    def test_check_reports_registered_weights(self):
        from pyslam.visual_models.__main__ import inspect
        from pyslam.visual_models import get_spec
        self.assertIn("LOFTR_OUTDOOR", inspect(get_spec("LOFTR", "pair_matcher"))["weights"])
        self.assertIn("ALIKED_N16", inspect(get_spec("ALIKED"))["weights"])

    def test_frontend_capability_and_sentinels(self):
        self.assertTrue(validate_frontend("SUPERPOINT").provides_descriptors)
        with self.assertRaisesRegex(ValueError, "per-keypoint descriptors"):
            validate_frontend("LOFTR", "pair_matcher")
        with self.assertRaisesRegex(ValueError, "sentinel"):
            get_local_feature("NONE")


class WeightResolverTest(unittest.TestCase):
    def test_registered_artifacts_have_pinned_sources_for_identical_content(self):
        entries = registered_weights()
        self.assertIn("LOFTR_OUTDOOR", entries)
        for entry in entries.values():
            sources = [WeightSource(**source) for source in entry["sources"]]
            self.assertEqual(len({source.sha256 for source in sources}), 1)
            self.assertIn(sources[0].kind, {"hf", "torch_hub"})
        self.assertEqual(entries["LOFTR_OUTDOOR"]["sources"][0]["kind"], "hf")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.content = b"a small, deterministic model fixture"
        self.digest = hashlib.sha256(self.content).hexdigest()
        self.source = WeightSource("url", "model.bin", self.digest, url="https://example.invalid/model.bin")
        self.resolver = WeightResolver(self.directory.name, offline=True)

    def cached(self, content):
        path = self.resolver.cached_path(self.source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_cache_hit_never_opens_network(self):
        expected = self.cached(self.content)
        with patch("pyslam.visual_models.weights.urlopen", side_effect=AssertionError("network")):
            self.assertEqual(self.resolver.resolve([self.source]), expected)

    def test_corrupt_cached_weights_are_rejected(self):
        self.cached(b"corrupt")
        with self.assertRaises(ChecksumError):
            self.resolver.resolve([self.source])

    def test_offline_miss_and_default_construction_never_download(self):
        with patch("pyslam.visual_models.weights.urlopen", side_effect=AssertionError("network")):
            with self.assertRaises(WeightsUnavailable):
                self.resolver.resolve([self.source], allow_download=True)
            with self.assertRaises(WeightsUnavailable):
                WeightResolver(self.directory.name, offline=False).resolve([self.source])

    def test_explicit_prefetch_verifies_and_publishes_atomically(self):
        resolver = WeightResolver(self.directory.name, offline=False)
        with patch.dict(os.environ, {"PYSLAM_OFFLINE": "0", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}), \
             patch("pyslam.visual_models.weights.urlopen", return_value=io.BytesIO(self.content)) as network:
            resolver = WeightResolver(self.directory.name, offline=False)
            result = resolver.resolve([self.source], allow_download=True)
        self.assertEqual(result.read_bytes(), self.content)
        network.assert_called_once()
        self.assertFalse(list(result.parent.glob("tmp*")))

    def test_bad_download_never_becomes_a_cache_entry(self):
        with patch.dict(os.environ, {"PYSLAM_OFFLINE": "0", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}), \
             patch("pyslam.visual_models.weights.urlopen", return_value=io.BytesIO(b"corrupt")):
            resolver = WeightResolver(self.directory.name, offline=False)
            with self.assertRaises(ChecksumError):
                resolver.resolve([self.source], allow_download=True)
        self.assertFalse(resolver.cached_path(self.source).exists())

    def test_legacy_weights_are_verified_without_moving_them(self):
        path = Path(self.directory.name) / "old-model.bin"
        path.write_bytes(self.content)
        self.assertEqual(self.resolver.resolve([self.source], legacy_paths=[path]), path)
        self.assertTrue(path.exists())

    def test_concurrent_prefetch_downloads_only_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with patch.dict(os.environ, {"PYSLAM_OFFLINE": "0", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}), \
             patch("pyslam.visual_models.weights.urlopen", side_effect=lambda *_args, **_kwargs: io.BytesIO(self.content)) as network:
            resolver = WeightResolver(self.directory.name, offline=False)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: resolver.resolve([self.source], allow_download=True), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(network.call_count, 1)

    def test_hf_is_pinned_and_paths_cannot_escape_cache(self):
        with self.assertRaises(ValueError):
            WeightSource("hf", "model.bin", self.digest, repo_id="a/b", revision="main")
        for name in ("../model.bin", "/model.bin", "a/../../model.bin", "..\\model.bin"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                WeightSource("url", name, self.digest, url="https://example.invalid/model.bin")

    def test_fallback_order_is_hf_then_torch_then_url(self):
        hf = WeightSource("hf", "model.bin", self.digest, repo_id="org/model", revision="a" * 40)
        hub = WeightSource("torch_hub", "model.bin", self.digest, url=self.source.url)
        path = Path(self.directory.name) / "torch/hub/checkpoints/model.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(self.content)
        provider = SimpleNamespace(hf_hub_download=unavailable_hf)
        with patch.dict(sys.modules, {"huggingface_hub": provider}), \
             patch.dict(os.environ, {"TORCH_HOME": str(Path(self.directory.name) / "torch")}), \
             patch("pyslam.visual_models.weights.urlopen", side_effect=AssertionError("network")):
            self.assertEqual(self.resolver.resolve([self.source, hub, hf]), path)

    def test_cache_override_and_environment_offline_win(self):
        with patch.dict(os.environ, {"PYSLAM_MODELS_DIR": self.directory.name, "PYSLAM_OFFLINE": "1"}):
            self.assertEqual(default_cache(), Path(self.directory.name))
            self.assertTrue(WeightResolver(offline=False).offline)


def unavailable_hf(**kwargs):
    if not kwargs["local_files_only"]:
        raise AssertionError("HF network enabled in offline mode")
    raise OSError("not cached")


def unavailable_constructor():
    raise ImportError("missing optional package")


if __name__ == "__main__":
    unittest.main()
