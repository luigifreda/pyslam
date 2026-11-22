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
* A benchmark with missing measurements must never pass the performance gate.
"""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking"))
from compare_benchmarks import compare


def measurement():
    return {"schema_version": 1, "kind": "synthetic_pair_microbenchmark", "machine": {}, "settings": {},
            "cases": [{"preset": "ORB2", "resolution": [640, 480], "core": "cpp", "device": "cpu",
                       "status": "ok", "import_ms": 10, "model_load_ms": 2, "peak_rss_bytes": 100,
                       "peak_gpu_bytes": 0, "fps": 100, "keypoints": [50, 50], "matches": 42,
                       "synthetic_inliers": 40, "stages_ms": {stage: {"median": 5, "p95": 6}
                       for stage in ("extraction", "matching", "total")}}]}


class BenchmarkComparisonTest(unittest.TestCase):
    def test_sequence_startup_without_accepted_poses_does_not_pass(self):
        import tempfile
        from benchmark_visual_slam import collect_case_metrics
        with tempfile.TemporaryDirectory() as folder:
            artifacts = Path(folder)
            case = {"playback": {"all_frames": {"frames": 60}}}
            collect_case_metrics(case, artifacts, 60)
            self.assertEqual(case["status"], "incomplete")
            (artifacts / "trajectory_final.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
            collect_case_metrics(case, artifacts, 60)
            self.assertEqual(case["status"], "ok")
            self.assertNotIn("reason", case)
            self.assertEqual(case["accepted_trajectory_rows"], 1)

    def test_sequence_config_overrides_do_not_mutate_originals(self):
        import tempfile
        from benchmark_visual_slam import prepare_config
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            settings = directory / "source.yaml"
            settings.write_text("FeatureTrackerConfig.name: OLD\nLoopDetectionConfig.name: OLD\nCamera.fx: 123\n")
            config = {"DATASET": {"type": "OLD"}, "KITTI_DATASET": {
                "settings": str(settings), "sensor_type": "stereo", "base_path": "/datasets"},
                "GLOBAL_PARAMETERS": None}
            original = copy.deepcopy(config)
            output = prepare_config(config, "KITTI_DATASET", "ORB2", "SAD", directory, directory)
            result = yaml.safe_load(output.read_text())
            self.assertEqual(config, original)
            self.assertIn("OLD", settings.read_text())
            self.assertIn("FeatureTrackerConfig.name: ORB2", Path(result["KITTI_DATASET"]["settings"]).read_text())
            self.assertEqual(result["KITTI_DATASET"]["base_path"], "/datasets")

    def test_equal_measurements_pass(self):
        self.assertEqual(compare(measurement(), measurement())["status"], "passed")

    def test_three_percent_time_gate_and_exact_accuracy(self):
        for change in (lambda case: case["stages_ms"]["total"].update(p95=6.19),
                       lambda case: case.update(matches=41),
                       lambda case: case.update(peak_rss_bytes=103),
                       lambda case: case.update(import_ms=11.5),
                       lambda case: case.update(model_load_ms=2.5)):
            baseline = measurement()
            candidate = copy.deepcopy(baseline)
            change(candidate["cases"][0])
            self.assertEqual(compare(baseline, candidate)["status"], "regression")

    def test_nondeterministic_cases_use_count_tolerance(self):
        # ORB2's default extractor gives slightly different matches on every run.
        baseline = measurement()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0].update(matches=43, deterministic=False)  # 42 -> 43: ~2.4%
        self.assertEqual(compare(baseline, candidate)["status"], "passed")
        candidate["cases"][0].update(matches=50)  # ~19%: still a regression
        self.assertEqual(compare(baseline, candidate)["status"], "regression")
        deterministic = copy.deepcopy(baseline)
        deterministic["cases"][0].update(matches=43, deterministic=True)
        self.assertEqual(compare(baseline, deterministic)["status"], "regression")

    def test_startup_and_memory_noise_within_tolerance_passes(self):
        # One-shot import/load wall-clock and RSS jitter must not flag a regression.
        baseline = measurement()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0].update(import_ms=10.9, model_load_ms=2.1, peak_rss_bytes=101)
        self.assertEqual(compare(baseline, candidate)["status"], "passed")
        self.assertEqual(compare(baseline, candidate, startup_tolerance=0.0)["status"], "regression")
        with self.assertRaises(ValueError):
            compare(baseline, candidate, memory_tolerance=1.5)

    def test_missing_failed_and_duplicate_cases_do_not_pass(self):
        for change in (lambda value: value.update(cases=[]),
                       lambda value: value["cases"][0].update(status="unavailable"),
                       lambda value: value["cases"].append(copy.deepcopy(value["cases"][0]))):
            candidate = measurement()
            change(candidate)
            self.assertEqual(compare(measurement(), candidate)["status"], "incomplete")

    def test_different_threads_hardware_and_missing_metrics_do_not_pass(self):
        for change in (lambda value: value["settings"].update(threads=2),
                       lambda value: value["machine"].update(gpu="different GPU"),
                       lambda value: value["cases"][0].update(fps=float("nan"))):
            candidate = measurement()
            change(candidate)
            self.assertEqual(compare(measurement(), candidate)["status"], "incomplete")

    def test_unsupported_schema_and_null_stages_do_not_pass(self):
        for change in (lambda value: value.update(schema_version=999),
                       lambda value: value["cases"][0]["stages_ms"].update(total=None)):
            candidate = measurement()
            change(candidate)
            self.assertEqual(compare(measurement(), candidate)["status"], "incomplete")

    def test_observed_gpu_clock_does_not_change_hardware_identity(self):
        before, after = measurement(), measurement()
        before["machine"]["gpu"] = "GPU, uuid, driver, 300 MHz, 100 W"
        after["machine"]["gpu"] = "GPU, uuid, driver, 1500 MHz, 100 W"
        self.assertEqual(compare(before, after)["status"], "passed")


class OutputEquivalenceTest(unittest.TestCase):
    def test_exact_indices_and_descriptor_dtype_are_preserved(self):
        import tempfile
        import numpy as np
        from compare_visual_model_outputs import compare_arrays
        with tempfile.TemporaryDirectory() as folder:
            before, after = Path(folder) / "a.npz", Path(folder) / "b.npz"
            np.savez(before, idxs=np.array([0, 1], dtype=np.int32), kps=np.array([[1., 2.]]))
            np.savez(after, idxs=np.array([0, 1], dtype=np.int32), kps=np.array([[1., 2.]]))
            self.assertEqual(compare_arrays(before, after), [])
            np.savez(after, idxs=np.array([0, 2], dtype=np.int32), kps=np.array([[1., 2.]]))
            self.assertTrue(compare_arrays(before, after, atol=10))
            np.savez(after, idxs=np.array([0, 1], dtype=np.int64), kps=np.array([[1., 2.]]))
            self.assertTrue(compare_arrays(before, after))


if __name__ == "__main__":
    unittest.main()
