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
* Registry-driven installer (pyslam.visual_models install): plan, status, verification.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pyslam.visual_models import install as inst
from pyslam.visual_models.model_registry import get_spec

ROOT = Path(__file__).resolve().parents[2]


class InstallPlanTest(unittest.TestCase):
    def test_every_installer_maps_to_a_registered_spec(self):
        self.assertEqual(set(inst.INSTALLERS), {"DEDODE", "RDD", "ROMA", "TINY_ROMA", "SALAD", "BOQ", "MEGALOC", "LOFTR"})
        for name, spec in inst.INSTALLERS.items():
            with self.subTest(name=name):
                self.assertIs(spec, get_spec(spec.name, spec.kind))
                self.assertIsNotNone(spec.wrapper)
                module = spec.wrapper.split(":")[0]
                self.assertTrue((ROOT / (module.replace(".", "/") + ".py")).is_file(), module)

    def test_group_plan_reports_legacy_members(self):
        items = inst.plan(group="vpr")
        installers = {i.name for i in items if i.install_kwargs is not None}
        legacy = {i.name for i in items if i.install_kwargs is None}
        self.assertTrue({"SALAD", "BOQ", "MEGALOC"} <= installers)
        self.assertIn("COSPLACE", legacy)
        self.assertEqual(inst.legacy_report(get_spec("COSPLACE", "global_descriptor"))["status"], "legacy installer")

    def test_new_integrations_have_registered_checksums(self):
        for name in ("DEDODE", "RDD", "ROMA", "TINY_ROMA", "SALAD", "BOQ"):
            with self.subTest(name=name):
                self.assertTrue(inst.artifacts_for(inst.INSTALLERS[name]))


class ArtifactStatusTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.content = b"weights"
        self.entry = {"sources": [{"filename": "model.pth", "kind": "url", "url": "https://example.invalid/m",
                                   "sha256": hashlib.sha256(self.content).hexdigest()}]}

    def status(self, content=None, verify=False):
        path = Path(self.folder.name) / "model.pth"
        if content is not None:
            path.write_bytes(content)
        with patch.object(inst, "artifact_locations", return_value=[path]):
            return inst.artifact_status(self.entry, verify_checksum=verify)

    def test_missing_present_verified_and_mismatch(self):
        self.assertEqual(self.status(), "missing")
        self.assertEqual(self.status(self.content), "present")
        self.assertEqual(self.status(self.content, verify=True), "verified")
        self.assertEqual(self.status(b"corrupt", verify=True), "checksum mismatch")

    def test_install_fails_when_an_artifact_is_missing_after_construction(self):
        installer = inst.INSTALLERS["DEDODE"]
        with patch("pyslam.visual_models.factory.load_constructor", return_value=lambda **kwargs: object()) as load, \
             patch.object(inst, "artifacts_for", return_value={"W": self.entry}), \
             patch.object(inst, "artifact_status", return_value="missing"):
            self.assertEqual(inst.install(installer)["status"], "failed")
        load.assert_called_once_with("DEDODE", "local_feature")
        with patch.object(inst, "artifacts_for", return_value={"W": self.entry}), \
             patch.object(inst, "artifact_status", return_value="present"):
            self.assertEqual(inst.install(installer, dry_run=True)["status"], "planned")


class InstallCliTest(unittest.TestCase):
    def test_list_and_wrapper_script(self):
        result = subprocess.run([sys.executable, "-m", "pyslam.visual_models", "install", "--list"], cwd=ROOT,
                                text=True, capture_output=True, timeout=60, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        for name in ("DEDODE", "RDD", "ROMA", "SALAD", "BOQ"):
            self.assertIn(name, result.stdout)
        script = ROOT / "scripts/install_visual_model.sh"
        self.assertTrue(os.access(script, os.X_OK))
        self.assertIn("pyslam.visual_models install", script.read_text())

    def test_named_install_resolves_every_model_kind(self):
        # Registered names that are not self-installing are reported with their legacy installer.
        result = subprocess.run([sys.executable, "-m", "pyslam.visual_models", "install", "--dry-run",
                                 "COSPLACE", "LIGHTGLUE", "MAST3R", "ORB"], cwd=ROOT, text=True,
                                capture_output=True, timeout=60, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        reports = {r["name"]: r for r in json.loads(result.stdout)}
        self.assertEqual(set(reports), {"COSPLACE", "LIGHTGLUE", "MAST3R", "ORB"})
        self.assertEqual({r["status"] for r in reports.values()}, {"legacy installer"})
        self.assertEqual(reports["COSPLACE"]["kind"], "global_descriptor")
        self.assertEqual(reports["MAST3R"]["kind"], "pair_matcher")
        unknown = subprocess.run([sys.executable, "-m", "pyslam.visual_models", "install", "--dry-run", "NO_SUCH_MODEL"],
                                 cwd=ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("Unknown model 'NO_SUCH_MODEL'", unknown.stderr)


if __name__ == "__main__":
    unittest.main()
