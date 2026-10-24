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
"""
"""
Test that compares C++ and Python KDTree implementations using identical interfaces
Supports multiple dimensions (2D, 3D, Dynamic)
"""

import sys
import os
import numpy as np
import time
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass


import pyslam.config as config
from pyslam.config_parameters import Parameters

USE_CPP = True
Parameters.USE_CPP_CORE = USE_CPP

import pyslam.slam as python_module

try:
    # Import C++ module
    import pyslam.slam.cpp as cpp_module

    if not cpp_module.CPP_AVAILABLE:
        print("❌ C++ module imported but core not available")
        sys.exit(1)
    print("✅ C++ module available for testing")
except ImportError as e:
    print(f"❌ Failed to import C++ module: {e}")
    sys.exit(1)


@dataclass
class TestResult:
    """Container for test results"""

    test_name: str
    passed: bool
    error_msg: str = ""
    performance_data: Dict[str, float] = None
    warnings: List[str] = None


class CKDTreeMultiDimComparison:
    """Compare C++ and Python KDTree implementations for multiple dimensions"""

    def __init__(self):
        self.tolerance = 1e-10

    def _create_trees_2d(self, points: np.ndarray) -> Tuple[Any, Any]:
        """Create both C++ and Python KDTree instances for 2D"""
        try:
            # Create C++ tree
            cpp_tree = cpp_module.CKDTree2d(points)

            # Create Python wrapper tree with identical interface
            python_tree = python_module.CKDTree2d(points)

            return cpp_tree, python_tree
        except Exception as e:
            raise RuntimeError(f"Failed to create 2D trees: {e}")

    def _create_trees_3d(self, points: np.ndarray) -> Tuple[Any, Any]:
        """Create both C++ and Python KDTree instances for 3D"""
        try:
            # Create C++ tree
            cpp_tree = cpp_module.CKDTree3d(points)

            # Create Python wrapper tree with identical interface
            python_tree = python_module.CKDTree3d(points)

            return cpp_tree, python_tree
        except Exception as e:
            raise RuntimeError(f"Failed to create 3D trees: {e}")

    def _create_trees_dyn(self, points: np.ndarray) -> Tuple[Any, Any]:
        """Create both C++ and Python KDTree instances for dynamic dimension"""
        try:
            # Create C++ tree
            cpp_tree = cpp_module.CKDTreeDyn(points)

            # Create Python wrapper tree with identical interface
            python_tree = python_module.CKDTreeDyn(points)

            return cpp_tree, python_tree
        except Exception as e:
            raise RuntimeError(f"Failed to create dynamic trees: {e}")

    def _compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray, tolerance: float = None) -> bool:
        """Compare two arrays with tolerance"""
        if tolerance is None:
            tolerance = self.tolerance

        if arr1.shape != arr2.shape:
            return False

        if arr1.dtype != arr2.dtype:
            return False

        return np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance)

    def _compare_lists(self, list1: List, list2: List) -> bool:
        """Compare two lists (order-independent)"""
        if len(list1) != len(list2):
            return False

        # Convert to sets for order-independent comparison
        set1 = set(list1)
        set2 = set(list2)
        return set1 == set2

    def _compare_knn_results(
        self, cpp_dists, cpp_indices, python_dists, python_indices, tolerance=1e-8
    ):
        """Compare k-NN results with tolerance for tie-breaking differences"""
        # Compare distances first
        if not self._compare_arrays(cpp_dists, python_dists, tolerance=tolerance):
            return False, "Distance mismatch"

        # For indices, check if the sets are the same (allowing for tie-breaking differences)
        cpp_set = set(cpp_indices.tolist())
        python_set = set(python_indices.tolist())

        if cpp_set != python_set:
            return False, "Index set mismatch (different tie-breaking)"

        return True, ""

    def _compare_radius_results(self, cpp_indices, python_indices, query_point, radius, points):
        """Compare radius query results with tolerance for boundary conditions"""
        cpp_list = cpp_indices.tolist() if hasattr(cpp_indices, "tolist") else list(cpp_indices)
        python_list = (
            python_indices.tolist() if hasattr(python_indices, "tolist") else list(python_indices)
        )

        # Check if the sets are the same
        cpp_set = set(cpp_list)
        python_set = set(python_list)

        if cpp_set == python_set:
            return True, ""

        # Check for boundary condition differences
        distances = np.linalg.norm(points - query_point, axis=1)
        boundary_points = np.abs(distances - radius) < 1e-10

        if np.any(boundary_points):
            return False, f"Boundary condition difference (radius={radius})"

        return False, "Index set mismatch"

    def test_basic_properties_2d(self, points: np.ndarray) -> TestResult:
        """Test basic tree properties for 2D"""
        try:
            cpp_tree, python_tree = self._create_trees_2d(points)

            # Test n (number of points)
            if cpp_tree.n != python_tree.n:
                return TestResult(
                    "basic_properties_2d",
                    False,
                    f"n mismatch: C++={cpp_tree.n}, Python={python_tree.n}",
                )

            # Test d (dimensions)
            if cpp_tree.d != python_tree.d:
                return TestResult(
                    "basic_properties_2d",
                    False,
                    f"d mismatch: C++={cpp_tree.d}, Python={python_tree.d}",
                )

            return TestResult("basic_properties_2d", True)

        except Exception as e:
            return TestResult("basic_properties_2d", False, str(e))

    def test_basic_properties_3d(self, points: np.ndarray) -> TestResult:
        """Test basic tree properties for 3D"""
        try:
            cpp_tree, python_tree = self._create_trees_3d(points)

            # Test n (number of points)
            if cpp_tree.n != python_tree.n:
                return TestResult(
                    "basic_properties_3d",
                    False,
                    f"n mismatch: C++={cpp_tree.n}, Python={python_tree.n}",
                )

            # Test d (dimensions)
            if cpp_tree.d != python_tree.d:
                return TestResult(
                    "basic_properties_3d",
                    False,
                    f"d mismatch: C++={cpp_tree.d}, Python={python_tree.d}",
                )

            return TestResult("basic_properties_3d", True)

        except Exception as e:
            return TestResult("basic_properties_3d", False, str(e))

    def test_basic_properties_dyn(self, points: np.ndarray) -> TestResult:
        """Test basic tree properties for dynamic dimension"""
        try:
            cpp_tree, python_tree = self._create_trees_dyn(points)

            # Test n (number of points)
            if cpp_tree.n != python_tree.n:
                return TestResult(
                    "basic_properties_dyn",
                    False,
                    f"n mismatch: C++={cpp_tree.n}, Python={python_tree.n}",
                )

            # Test d (dimensions)
            if cpp_tree.d != python_tree.d:
                return TestResult(
                    "basic_properties_dyn",
                    False,
                    f"d mismatch: C++={cpp_tree.d}, Python={python_tree.d}",
                )

            return TestResult("basic_properties_dyn", True)

        except Exception as e:
            return TestResult("basic_properties_dyn", False, str(e))

    def test_query_method_2d(self, points: np.ndarray) -> TestResult:
        """Test query method for 2D"""
        try:
            cpp_tree, python_tree = self._create_trees_2d(points)

            # Test query points
            query_points = [
                np.array([0.5, 0.5]),
                np.array([2.0, 2.0]),
                np.array([5.0, 5.0]),  # Outside dataset bounds
            ]

            k_values = [1, 3, min(5, len(points))]
            warnings = []

            for query_point in query_points:
                for k in k_values:
                    if k > len(points):
                        continue

                    # Both trees now have identical interfaces!
                    cpp_dists, cpp_indices = cpp_tree.query(query_point, k=k, return_distance=True)
                    python_dists, python_indices = python_tree.query(
                        query_point, k=k, return_distance=True
                    )

                    # Compare results with tolerance for tie-breaking
                    is_match, error_msg = self._compare_knn_results(
                        cpp_dists, cpp_indices, python_dists, python_indices
                    )

                    if not is_match:
                        # Check if this is a tie-breaking difference we can accept
                        if "tie-breaking" in error_msg:
                            warnings.append(
                                f"Tie-breaking difference for query {query_point}, k={k}"
                            )
                        else:
                            return TestResult(
                                "query_method_2d",
                                False,
                                f"{error_msg} for query {query_point}, k={k}",
                            )

            return TestResult("query_method_2d", True, warnings=warnings)

        except Exception as e:
            return TestResult("query_method_2d", False, str(e))

    def test_query_method_3d(self, points: np.ndarray) -> TestResult:
        """Test query method for 3D"""
        try:
            cpp_tree, python_tree = self._create_trees_3d(points)

            # Test query points
            query_points = [
                np.array([0.5, 0.5, 0.5]),
                np.array([2.0, 2.0, 2.0]),
                np.array([5.0, 5.0, 5.0]),  # Outside dataset bounds
            ]

            k_values = [1, 3, min(5, len(points))]
            warnings = []

            for query_point in query_points:
                for k in k_values:
                    if k > len(points):
                        continue

                    # Both trees now have identical interfaces!
                    cpp_dists, cpp_indices = cpp_tree.query(query_point, k=k, return_distance=True)
                    python_dists, python_indices = python_tree.query(
                        query_point, k=k, return_distance=True
                    )

                    # Compare results with tolerance for tie-breaking
                    is_match, error_msg = self._compare_knn_results(
                        cpp_dists, cpp_indices, python_dists, python_indices
                    )

                    if not is_match:
                        # Check if this is a tie-breaking difference we can accept
                        if "tie-breaking" in error_msg:
                            warnings.append(
                                f"Tie-breaking difference for query {query_point}, k={k}"
                            )
                        else:
                            return TestResult(
                                "query_method_3d",
                                False,
                                f"{error_msg} for query {query_point}, k={k}",
                            )

            return TestResult("query_method_3d", True, warnings=warnings)

        except Exception as e:
            return TestResult("query_method_3d", False, str(e))

    def test_performance_comparison_2d(self, points: np.ndarray) -> TestResult:
        """Compare performance for 2D"""
        try:
            cpp_tree, python_tree = self._create_trees_2d(points)

            # Generate test queries
            n_queries = 1000
            query_points = np.random.rand(n_queries, 2) * 10

            performance_data = {}

            # Test k-NN queries
            k = 10
            times_cpp = []
            times_python = []

            for query_point in query_points:
                # C++ timing
                start = time.perf_counter()
                cpp_tree.query(query_point, k=k, return_distance=True)
                times_cpp.append(time.perf_counter() - start)

                # Python timing (now with identical interface!)
                start = time.perf_counter()
                python_tree.query(query_point, k=k, return_distance=True)
                times_python.append(time.perf_counter() - start)

            performance_data["knn_cpp_mean"] = np.mean(times_cpp)
            performance_data["knn_python_mean"] = np.mean(times_python)
            performance_data["knn_speedup"] = np.mean(times_python) / np.mean(times_cpp)

            return TestResult("performance_comparison_2d", True, performance_data=performance_data)

        except Exception as e:
            return TestResult("performance_comparison_2d", False, str(e))

    def test_performance_comparison_3d(self, points: np.ndarray) -> TestResult:
        """Compare performance for 3D"""
        try:
            cpp_tree, python_tree = self._create_trees_3d(points)

            # Generate test queries
            n_queries = 1000
            query_points = np.random.rand(n_queries, 3) * 10

            performance_data = {}

            # Test k-NN queries
            k = 10
            times_cpp = []
            times_python = []

            for query_point in query_points:
                # C++ timing
                start = time.perf_counter()
                cpp_tree.query(query_point, k=k, return_distance=True)
                times_cpp.append(time.perf_counter() - start)

                # Python timing (now with identical interface!)
                start = time.perf_counter()
                python_tree.query(query_point, k=k, return_distance=True)
                times_python.append(time.perf_counter() - start)

            performance_data["knn_cpp_mean"] = np.mean(times_cpp)
            performance_data["knn_python_mean"] = np.mean(times_python)
            performance_data["knn_speedup"] = np.mean(times_python) / np.mean(times_cpp)

            return TestResult("performance_comparison_3d", True, performance_data=performance_data)

        except Exception as e:
            return TestResult("performance_comparison_3d", False, str(e))

    def run_all_tests(self) -> List[TestResult]:
        """Run all tests for multiple dimensions"""
        results = []

        # Generate test datasets for different dimensions
        test_datasets_2d = {
            "grid_5x5_2d": self._create_grid_dataset(5, 5),
            "random_100_2d": self._create_random_dataset(100, 2),
            "random_1000_2d": self._create_random_dataset(1000, 2),
        }

        test_datasets_3d = {
            "grid_3x3x3_3d": self._create_grid_dataset_3d(3, 3, 3),
            "random_100_3d": self._create_random_dataset(100, 3),
            "random_1000_3d": self._create_random_dataset(1000, 3),
        }

        test_datasets_dyn = {
            "random_100_4d": self._create_random_dataset(100, 4),
            "random_100_5d": self._create_random_dataset(100, 5),
        }

        print("Running CKDTree multi-dimensional comparison tests...")
        print("=" * 60)

        # Test 2D
        print("\nTesting 2D implementations:")
        for dataset_name, points in test_datasets_2d.items():
            print(f"\n  Dataset: {dataset_name}")

            result = self.test_basic_properties_2d(points)
            result.test_name = f"{result.test_name}_{dataset_name}"
            results.append(result)
            self._print_result(result)

            result = self.test_query_method_2d(points)
            result.test_name = f"{result.test_name}_{dataset_name}"
            results.append(result)
            self._print_result(result)

        # Test 3D
        print("\nTesting 3D implementations:")
        for dataset_name, points in test_datasets_3d.items():
            print(f"\n  Dataset: {dataset_name}")

            result = self.test_basic_properties_3d(points)
            result.test_name = f"{result.test_name}_{dataset_name}"
            results.append(result)
            self._print_result(result)

            result = self.test_query_method_3d(points)
            result.test_name = f"{result.test_name}_{dataset_name}"
            results.append(result)
            self._print_result(result)

        # Test Dynamic dimension
        print("\nTesting Dynamic dimension implementations:")
        for dataset_name, points in test_datasets_dyn.items():
            print(f"\n  Dataset: {dataset_name}")

            result = self.test_basic_properties_dyn(points)
            result.test_name = f"{result.test_name}_{dataset_name}"
            results.append(result)
            self._print_result(result)

        # Performance tests
        print(f"\n⚡ Performance comparison:")
        large_points_2d = self._create_random_dataset(10000, 2)
        result = self.test_performance_comparison_2d(large_points_2d)
        results.append(result)
        self._print_result(result)

        large_points_3d = self._create_random_dataset(10000, 3)
        result = self.test_performance_comparison_3d(large_points_3d)
        results.append(result)
        self._print_result(result)

        return results

    def _create_grid_dataset(self, nx: int, ny: int) -> np.ndarray:
        """Create a 2D grid dataset"""
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        X, Y = np.meshgrid(x, y)
        return np.column_stack([X.ravel(), Y.ravel()])

    def _create_grid_dataset_3d(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """Create a 3D grid dataset"""
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        z = np.linspace(0, nz - 1, nz)
        X, Y, Z = np.meshgrid(x, y, z)
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def _create_random_dataset(self, n_points: int, n_dims: int) -> np.ndarray:
        """Create a random dataset with specified dimensions"""
        np.random.seed(42)  # For reproducibility
        return np.random.rand(n_points, n_dims) * 10

    def _print_result(self, result: TestResult):
        """Print test result"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"    {status} {result.test_name}")

        if not result.passed and result.error_msg:
            print(f"      Error: {result.error_msg}")

        if result.warnings:
            print(f"      Warnings:")
            for warning in result.warnings:
                print(f"        ⚠️  {warning}")

        if result.performance_data:
            print(f"      Performance data:")
            for key, value in result.performance_data.items():
                if "speedup" in key:
                    print(f"        {key}: {value:.2f}x")
                else:
                    print(f"        {key}: {value:.6f}s")


def main():
    """Main test runner"""
    print("CKDTree Multi-Dimensional Interface Comparison Test")
    print("=" * 60)
    print("Comparing C++ and Python implementations for 2D, 3D, and Dynamic dimensions")
    print("Using generalized CKDTreeWrapper around scipy.spatial.cKDTree")
    print()

    comparator = CKDTreeMultiDimComparison()
    results = comparator.run_all_tests()

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Both implementations are consistent across all dimensions.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        failed_tests = [r for r in results if not r.passed]
        print(f"\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test.test_name}: {test.error_msg}")

    # Count warnings
    total_warnings = sum(len(r.warnings) if r.warnings else 0 for r in results)
    if total_warnings > 0:
        print(f"\n⚠️  Total warnings: {total_warnings}")
        print("These are expected differences due to implementation-specific behaviors:")
        print("  - Tie-breaking differences in k-NN queries")
        print("  - Boundary condition handling in radius queries")
        print("  - Different ordering of results with equal distances")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
