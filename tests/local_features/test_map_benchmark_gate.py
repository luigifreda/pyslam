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
* Map timing comparisons remain informative without inventing a relative-speed guarantee.
"""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarking"))
sys.path.insert(0, str(ROOT / "cpp/core/slam/tests_py"))
from compare_map_benchmarks import compare
from test_tools import PerformanceTestUnit


class FakeBenchmark(PerformanceTestUnit):
    def __init__(self, invalid=False):
        super().__init__("fake", num_iterations=4)
        self.calls = []
        self.invalid = invalid

    def setup(self):
        pass

    def run_benchmark(self, module_type, module):
        self.calls.append(module_type)
        if self.invalid and len(self.calls) == 3:
            raise RuntimeError("a failed sample must not be hidden by its median")
        return 2.0 if module_type == "C++" else 1.0


def report():
    machine = dict(system="Linux", release="test", machine="x86_64", processor="", cpu_count=8,
                   python="3.11", packages={"numpy": "test"})
    return dict(schema_version=1, kind="map_operations", machine=machine, workload={"points": 1000},
                cases=[dict(name="Map", cpp_time=2.0, python_time=1.0)])


class MapBenchmarkGateTest(unittest.TestCase):
    def test_relative_comparison_is_not_a_regression_gate(self):
        self.assertTrue(FakeBenchmark().execute(None, None, require_speedup=False).passed)
        self.assertFalse(FakeBenchmark().execute(None, None, require_speedup=True).passed)
        self.assertEqual(compare(report(), report())["status"], "passed")

    def test_measurements_are_paired_with_alternating_order(self):
        benchmark = FakeBenchmark()
        benchmark.execute(None, None, require_speedup=False)
        self.assertEqual(benchmark.calls[2:], ["C++", "Python", "Python", "C++"] * 2)

    def test_failed_samples_still_fail_advisory_comparisons(self):
        result = FakeBenchmark(invalid=True).execute(None, None, require_speedup=False)
        self.assertFalse(result.passed)
        self.assertIn("failed sample", result.error_message)

    def test_same_backend_regressions_still_fail(self):
        for backend in ("cpp_time", "python_time"):
            before, after = report(), report()
            after["cases"][0][backend] *= 1.04
            self.assertEqual(compare(before, after)["status"], "regression")

    def test_incomplete_or_changed_workload_does_not_pass(self):
        for mutate in (lambda value: value["cases"].clear(),
                       lambda value: value["machine"].clear(),
                       lambda value: value["cases"][0].update(cpp_time=float("nan")),
                       lambda value: value["workload"].update(points=2000)):
            candidate = copy.deepcopy(report())
            mutate(candidate)
            self.assertEqual(compare(report(), candidate)["status"], "incomplete")
