#!/usr/bin/env python3
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
* Test script for pybind11 SLAM C++ module
* Tests core C++ classes against Python implementations with systematic comparison
"""

import sys
import os
import numpy as np
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union
from abc import ABC, abstractmethod
import cv2


@dataclass
class TestResult:
    """Result of a single test unit"""

    name: str
    passed: bool
    cpp_result: Any = None
    python_result: Any = None
    cpp_time: float = 0.0
    python_time: float = 0.0
    error_message: str = ""
    tolerance: float = 1e-6


@dataclass
class PerformanceResult:
    """Result of a performance test"""

    name: str
    cpp_time: float
    python_time: float
    speedup: float
    passed: bool
    require_speedup: bool = True
    error_message: str = ""


# ============================================================================
# Test Units
# ============================================================================


class TestUnit(ABC):
    """Abstract base class for test units"""

    def __init__(self, name: str, tolerance: float = 1e-6, detailed: bool = False):
        self.name = name
        self.tolerance = tolerance
        self.detailed = detailed

    @abstractmethod
    def run_test(self, module_type: str, module: Any) -> Any:
        f"""Run test using {module_type} module"""
        pass

    def compare_results(self, cpp_result: Any, python_result: Any) -> bool:
        """Compare C++ and Python results"""
        try:
            if isinstance(cpp_result, dict) and isinstance(python_result, dict):
                if set(cpp_result.keys()) != set(python_result.keys()):
                    return False
                for key in cpp_result.keys():
                    if not self.compare_results(cpp_result[key], python_result[key]):
                        return False
                return True
            elif isinstance(cpp_result, np.ndarray) and isinstance(python_result, np.ndarray):
                # Handle different dtypes
                if cpp_result.dtype != python_result.dtype:
                    try:
                        # Try to convert to same dtype for comparison
                        if cpp_result.dtype.kind in "uif" and python_result.dtype.kind in "uif":
                            return np.allclose(
                                cpp_result.astype(np.float64),
                                python_result.astype(np.float64),
                                atol=self.tolerance,
                            )
                    except:
                        pass
                return np.allclose(cpp_result, python_result, atol=self.tolerance)
            elif isinstance(cpp_result, (list, tuple)) and isinstance(python_result, (list, tuple)):
                if len(cpp_result) != len(python_result):
                    return False
                for c, p in zip(cpp_result, python_result):
                    if not self.compare_results(c, p):
                        return False
                return True
            elif isinstance(cpp_result, (int, float)) and isinstance(python_result, (int, float)):
                return abs(cpp_result - python_result) <= self.tolerance
            else:
                return cpp_result == python_result
        except Exception:
            return False

    def compare_results_detailed(
        self, cpp_result: Any, python_result: Any, path: str = "root"
    ) -> bool:
        """Enhanced comparison with detailed debugging information"""
        print(f"\n🔍 Comparing: {path}")

        try:
            # Handle None values
            if cpp_result is None and python_result is None:
                print(f"  ✅ Both None")
                return True
            elif cpp_result is None or python_result is None:
                print(f"  ❌ One is None: C++={cpp_result is None}, Python={python_result is None}")
                return False

            # Handle dictionaries
            if isinstance(cpp_result, dict) and isinstance(python_result, dict):
                print(f"  📁 Comparing dictionaries with {len(cpp_result)} keys")

                cpp_keys = set(cpp_result.keys())
                python_keys = set(python_result.keys())

                if cpp_keys != python_keys:
                    print(f"  ❌ Key mismatch!")
                    print(f"    C++ keys: {sorted(cpp_keys)}")
                    print(f"    Python keys: {sorted(python_keys)}")
                    print(f"    Missing in C++: {sorted(python_keys - cpp_keys)}")
                    print(f"    Missing in Python: {sorted(cpp_keys - python_keys)}")
                    return False

                print(f"  ✅ Keys match: {sorted(cpp_keys)}")

                all_passed = True
                for key in sorted(cpp_keys):
                    key_path = f"{path}.{key}"
                    if not self.compare_results_detailed(
                        cpp_result[key], python_result[key], key_path
                    ):
                        all_passed = False

                return all_passed

            # Handle numpy arrays
            elif isinstance(cpp_result, np.ndarray) and isinstance(python_result, np.ndarray):
                print(f"  🔢 Comparing numpy arrays")
                print(f"    C++ shape: {cpp_result.shape}, dtype: {cpp_result.dtype}")
                print(f"    Python shape: {python_result.shape}, dtype: {python_result.dtype}")

                # Shape comparison
                if cpp_result.shape != python_result.shape:
                    print(f"  ❌ Shape mismatch!")
                    return False
                print(f"  ✅ Shape match: {cpp_result.shape}")

                # Data type comparison
                if cpp_result.dtype != python_result.dtype:
                    print(
                        f"  ⚠️  Dtype mismatch: C++={cpp_result.dtype}, Python={python_result.dtype}"
                    )

                    # Try to convert to same dtype for comparison
                    try:
                        if cpp_result.dtype.kind in "uif" and python_result.dtype.kind in "uif":
                            cpp_converted = cpp_result.astype(np.float64)
                            python_converted = python_result.astype(np.float64)

                            if np.allclose(cpp_converted, python_converted, atol=self.tolerance):
                                print(f"  ✅ Values match after dtype conversion")
                                return True
                            else:
                                print(f"  ❌ Values differ after dtype conversion")
                                self._analyze_array_differences(cpp_converted, python_converted)
                                return False
                        else:
                            print(f"  ❌ Cannot convert dtypes for comparison")
                            return False
                    except Exception as e:
                        print(f"  ❌ Dtype conversion failed: {e}")
                        return False
                else:
                    print(f"  ✅ Dtype match: {cpp_result.dtype}")

                # Value comparison
                if np.allclose(cpp_result, python_result, atol=self.tolerance):
                    print(f"  ✅ Values match within tolerance {self.tolerance}")
                    return True
                else:
                    print(f"  ❌ Values differ beyond tolerance {self.tolerance}")
                    self._analyze_array_differences(cpp_result, python_result)
                    return False

            # Handle lists/tuples
            elif isinstance(cpp_result, (list, tuple)) and isinstance(python_result, (list, tuple)):
                print(f"  📋 Comparing sequences")
                print(f"    C++ length: {len(cpp_result)}, type: {type(cpp_result).__name__}")
                print(
                    f"    Python length: {len(python_result)}, type: {type(python_result).__name__}"
                )

                if len(cpp_result) != len(python_result):
                    print(f"  ❌ Length mismatch!")
                    return False
                print(f"  ✅ Length match: {len(cpp_result)}")

                all_passed = True
                for i, (c, p) in enumerate(zip(cpp_result, python_result)):
                    item_path = f"{path}[{i}]"
                    if not self.compare_results_detailed(c, p, item_path):
                        all_passed = False
                        # Only show first few failures to avoid spam
                        if i < 3:
                            continue
                        else:
                            print(f"  ⚠️  More failures in sequence (showing first 3)...")
                            break

                return all_passed

            # Handle numeric values
            elif isinstance(cpp_result, (int, float)) and isinstance(python_result, (int, float)):
                print(f"  🔢 Comparing numbers")
                print(f"    C++: {cpp_result} ({type(cpp_result).__name__})")
                print(f"    Python: {python_result} ({type(python_result).__name__})")

                diff = abs(cpp_result - python_result)
                if diff <= self.tolerance:
                    print(f"  ✅ Values match (diff: {diff:.2e} <= {self.tolerance:.2e})")
                    return True
                else:
                    print(f"  ❌ Values differ (diff: {diff:.2e} > {self.tolerance:.2e})")
                    return False

            # Handle boolean values
            elif isinstance(cpp_result, bool) and isinstance(python_result, bool):
                print(f"  ✅ Comparing booleans: C++={cpp_result}, Python={python_result}")
                return cpp_result == python_result

            # Handle strings
            elif isinstance(cpp_result, str) and isinstance(python_result, str):
                print(f"  📝 Comparing strings")
                print(f"    C++ length: {len(cpp_result)}")
                print(f"    Python length: {len(python_result)}")
                if cpp_result == python_result:
                    print(f"  ✅ Strings match")
                    return True
                else:
                    print(f"  ❌ Strings differ")
                    # Show first 100 chars if they're long
                    if len(cpp_result) > 100 or len(python_result) > 100:
                        print(f"    C++ (first 100): {cpp_result[:100]}...")
                        print(f"    Python (first 100): {python_result[:100]}...")
                    else:
                        print(f"    C++: {cpp_result}")
                        print(f"    Python: {python_result}")
                    return False

            # Handle other types
            else:
                print(f"  🔍 Comparing other types")
                print(f"    C++ type: {type(cpp_result).__name__}, value: {cpp_result}")
                print(f"    Python type: {type(python_result).__name__}, value: {python_result}")

                if cpp_result == python_result:
                    print(f"  ✅ Values match exactly")
                    return True
                else:
                    print(f"  ❌ Values differ")
                    return False

        except Exception as e:
            print(f"  ❌ Comparison failed with exception: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _analyze_array_differences(self, cpp_array: np.ndarray, python_array: np.ndarray):
        """Analyze differences between two numpy arrays"""
        try:
            # Convert to float64 for analysis
            cpp_f64 = cpp_array.astype(np.float64)
            python_f64 = python_array.astype(np.float64)

            # Calculate differences
            diff = np.abs(cpp_f64 - python_f64)

            # Statistics
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            num_different = np.sum(diff > self.tolerance)
            total_elements = diff.size

            print(f"    📊 Difference Statistics:")
            print(f"      Max difference: {max_diff:.6e}")
            print(f"      Mean difference: {mean_diff:.6e}")
            print(f"      Std difference: {std_diff:.6e}")
            print(
                f"      Elements exceeding tolerance: {num_different}/{total_elements} ({100*num_different/total_elements:.1f}%)"
            )

            # Show sample differences
            if total_elements <= 20:
                print(f"    📋 All differences:")
                for i in range(total_elements):
                    flat_idx = np.unravel_index(i, diff.shape)
                    cpp_val = cpp_f64[flat_idx]
                    python_val = python_f64[flat_idx]
                    diff_val = diff[flat_idx]
                    print(
                        f"      [{flat_idx}]: C++={cpp_val:.6e}, Python={python_val:.6e}, diff={diff_val:.6e}"
                    )
            else:
                print(f"    📋 Sample differences (first 10):")
                flat_diff = diff.flatten()
                flat_cpp = cpp_f64.flatten()
                flat_python = python_f64.flatten()

                # Show largest differences
                largest_indices = np.argsort(flat_diff)[-10:]
                for idx in reversed(largest_indices):
                    print(
                        f"      [{idx}]: C++={flat_cpp[idx]:.6e}, Python={flat_python[idx]:.6e}, diff={flat_diff[idx]:.6e}"
                    )

            # Check for patterns
            if max_diff > 100:
                print(f"    ⚠️  Large differences detected - possible algorithm mismatch")
            elif num_different / total_elements > 0.5:
                print(f"    ⚠️  Many elements differ - possible systematic difference")
            elif max_diff < 1e-10:
                print(f"    ℹ️  Very small differences - likely precision/rounding issues")

        except Exception as e:
            print(f"    ❌ Analysis failed: {e}")

    def execute(self, cpp_module, python_module) -> TestResult:
        """Execute the test unit on both modules"""
        result = TestResult(name=self.name, passed=False, tolerance=self.tolerance)

        try:
            # Run C++ version
            start_time = time.perf_counter()
            result.cpp_result = self.run_test("C++", cpp_module)
            result.cpp_time = time.perf_counter() - start_time
        except Exception as e:
            result.error_message = str(e)
            result.passed = False
            print(f"C++ test failed: {e}")
            traceback.print_exc()
            return result

        try:
            # Run Python version
            start_time = time.perf_counter()
            result.python_result = self.run_test("Python", python_module)
            result.python_time = time.perf_counter() - start_time

        except Exception as e:
            result.error_message = str(e)
            result.passed = False
            print(f"Python test failed: {e}")
            traceback.print_exc()
            return result

        # Compare results
        try:
            if self.detailed:
                result.passed = self.compare_results_detailed(
                    result.cpp_result, result.python_result
                )
            else:
                result.passed = self.compare_results(result.cpp_result, result.python_result)
        except Exception as e:
            result.error_message = str(e)
            result.passed = False
            print(f"Comparison failed: {e}")
            traceback.print_exc()
            return result

        return result


# ============================================================================
# Performance Test Units
# ============================================================================


class PerformanceTestUnit(ABC):
    """Abstract base class for performance test units"""

    def __init__(self, name: str, num_iterations: int = 10):
        self.name = name
        self.num_iterations = num_iterations

    @abstractmethod
    def setup(self):
        """Setup test data"""
        pass

    @abstractmethod
    def run_benchmark(self, module_type: str, module: Any) -> float:
        """Run C++ benchmark, return execution time"""
        pass

    def execute(self, cpp_module, python_module, *, require_speedup=True) -> PerformanceResult:
        """Execute performance test"""
        self.setup()

        errors = []
        for label, module in (("C++", cpp_module), ("Python", python_module)):
            try:
                self.run_benchmark(label, module)
            except Exception as error:
                errors.append(f"{label} warmup: {error}")

        samples = {"C++": [], "Python": []}
        for iteration in range(self.num_iterations):
            # Pair measurements and counterbalance order to reduce clock/cache drift.
            pair = (("C++", cpp_module), ("Python", python_module))
            for label, module in pair if iteration % 2 == 0 else reversed(pair):
                try:
                    elapsed = float(self.run_benchmark(label, module))
                    if not np.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError(f"invalid elapsed time: {elapsed}")
                    samples[label].append(elapsed)
                except Exception as error:
                    errors.append(f"{label} sample {iteration}: {error}")
                    samples[label].append(float("inf"))
        cpp_times, python_times = samples["C++"], samples["Python"]

        cpp_median = np.median(cpp_times)
        python_median = np.median(python_times)
        speedup = python_median / cpp_median if cpp_median > 0 else 0
        valid = not errors and bool(cpp_times) and np.isfinite(cpp_median) and np.isfinite(python_median)
        passed = bool(valid and (not require_speedup or cpp_median < python_median))

        return PerformanceResult(
            name=self.name,
            cpp_time=cpp_median,
            python_time=python_median,
            speedup=speedup,
            passed=passed,
            require_speedup=require_speedup,
            error_message="; ".join(errors),
        )


# ============================================================================
# Test Suite Manager
# ============================================================================


class TestSuite:
    """Manages and executes all test units"""

    def __init__(self):
        self.functional_tests = []
        self.performance_tests = []
        self.setup_complete = False

    # def setup_environment(self) -> bool:
    #     """Setup test environment"""
    #     try:
    #         # Setup FeatureTrackerShared
    #         tracker_config = FeatureTrackerConfigs.ORB2
    #         tracker_config["num_features"] = 1000
    #         feature_tracker = feature_tracker_factory(**tracker_config)
    #         FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

    #         print("✅ Test environment setup complete")
    #         self.setup_complete = True
    #         return True
    #     except Exception as e:
    #         print(f"❌ Failed to setup test environment: {e}")
    #         traceback.print_exc()
    #         return False

    def add_functional_test(self, test_unit: TestUnit):
        """Add a functional test unit"""
        self.functional_tests.append(test_unit)

    def add_performance_test(self, test_unit: PerformanceTestUnit):
        """Add a performance test unit"""
        self.performance_tests.append(test_unit)

    def run_functional_tests(self, cpp_module, python_module) -> List[TestResult]:
        """Run all functional tests"""
        print("\n" + "=" * 100)
        print("Running Functional Tests")
        print("=" * 60)

        results = []
        for test in self.functional_tests:
            print("\n" + "-" * 60)
            print(f"Running {test.name}...", end=" ")
            result = test.execute(cpp_module, python_module)
            results.append(result)

            # Fix the numpy array boolean issue
            passed = result.passed
            if isinstance(passed, np.ndarray):
                passed = bool(np.all(passed))
            else:
                passed = bool(passed)

            if passed:
                print(f"✅ PASSED (C++: {result.cpp_time:.4f}s, Python: {result.python_time:.4f}s)")
            else:
                print(f"❌ FAILED - {result.error_message}")
                if hasattr(result, "cpp_result") and hasattr(result, "python_result"):
                    print(f"   C++ result: {result.cpp_result}")
                    print(f"   Python result: {result.python_result}")

        return results

    def run_performance_tests(self, cpp_module, python_module, *, require_speedup=True) -> List[PerformanceResult]:
        """Run all performance tests"""
        print("\n" + "=" * 100)
        print("Running Performance Tests")
        print("=" * 60)

        results = []
        for test in self.performance_tests:
            print("\n" + "-" * 60)
            print(f"Running {test.name}...", end=" ")
            result = test.execute(cpp_module, python_module, require_speedup=require_speedup)
            results.append(result)

            # Fix the numpy array boolean issue
            passed = result.passed
            if isinstance(passed, np.ndarray):
                passed = bool(np.all(passed))
            else:
                passed = bool(passed)

            if passed:
                status = "PASSED" if result.require_speedup else "MEASURED (comparison only)"
                ratio_label = "Speedup" if result.require_speedup else "Python/C++ time ratio"
                print(
                    f"✅ {status} - {ratio_label}: {result.speedup:.2f}x (C++: {result.cpp_time:.6f}s, Python: {result.python_time:.6f}s)"
                )
            else:
                print(
                    f"❌ FAILED - {result.error_message or 'C++ slower'} (C++: {result.cpp_time:.6f}s, Python: {result.python_time:.6f}s)"
                )

        return results

    def print_summary(
        self, functional_results: List[TestResult], performance_results: List[PerformanceResult]
    ):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        # Functional tests summary - Fix the numpy array boolean issue
        passed_functional = sum(
            1
            for r in functional_results
            if (bool(np.all(r.passed)) if isinstance(r.passed, np.ndarray) else bool(r.passed))
        )
        total_functional = len(functional_results)
        print(f"Functional Tests: {passed_functional}/{total_functional} passed")

        # Performance tests summary - Fix the numpy array boolean issue
        passed_performance = sum(
            1
            for r in performance_results
            if (bool(np.all(r.passed)) if isinstance(r.passed, np.ndarray) else bool(r.passed))
        )
        total_performance = len(performance_results)
        comparisons_only = bool(performance_results) and all(not r.require_speedup for r in performance_results)
        label = "Performance measurements" if comparisons_only else "Performance Tests"
        print(f"{label}: {passed_performance}/{total_performance} {'valid' if comparisons_only else 'passed'}")

        # Functional details
        if functional_results:
            print(f"\nFunctional Details:")
            for result in functional_results:
                passed = (
                    bool(np.all(result.passed))
                    if isinstance(result.passed, np.ndarray)
                    else bool(result.passed)
                )
                status = "✅" if passed else "❌"
                print(f"  {status} {result.name}")

        # Performance tests summary - Fix the numpy array boolean issue

        # Performance details
        if performance_results:
            print(f"\nPerformance Details:")
            for result in performance_results:
                passed = (
                    bool(np.all(result.passed))
                    if isinstance(result.passed, np.ndarray)
                    else bool(result.passed)
                )
                status = "✅" if passed else "❌"
                ratio_label = "speedup" if result.require_speedup else "Python/C++ time ratio"
                print(f"  {status} {result.name:30} {result.speedup:6.1f}x {ratio_label}")

        total_passed = passed_functional + passed_performance
        total_tests = total_functional + total_performance

        print(f"\nOverall: {total_passed}/{total_tests} {'checks' if comparisons_only else 'tests'} passed")

        return total_passed == total_tests
