#!/usr/bin/env python3
"""
Test script for pybind11 SLAM C++ module
Tests core C++ classes against Python implementations with systematic comparison
"""

import sys
import os
import numpy as np
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union
from abc import ABC, abstractmethod

from pyslam.config import Config

config = Config()

import g2o
import cv2

try:
    import pyslam.slam.cpp as cpp_module

    if not cpp_module.CPP_AVAILABLE:
        print("❌ cpp_module imported successfully but C++ core is not available")
        sys.exit(1)
    print("✅ cpp_module imported successfully, C++ core is available")
except ImportError as e:
    print(f"❌ Failed to import C++ module: {e}")
    sys.exit(1)

import pyslam.slam as python_module
from pyslam.slam.frame import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


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


class TestUnit(ABC):
    """Abstract base class for test units"""

    def __init__(self, name: str, tolerance: float = 1e-6):
        self.name = name
        self.tolerance = tolerance

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
            result.passed = self.compare_results(result.cpp_result, result.python_result)
        except Exception as e:
            result.error_message = str(e)
            result.passed = False
            print(f"Comparison failed: {e}")
            traceback.print_exc()
            return result

        return result


# ============================================================================
# Test Units for MapPoint
# ============================================================================


class MapPointCreationTest(TestUnit):
    """Test MapPoint creation and basic properties"""

    def __init__(self):
        super().__init__("MapPoint Creation")
        self.position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.color = np.array([255, 128, 64], dtype=np.uint8)

    def run_test(self, module_type: str, module: Any) -> Dict:
        mp = module.MapPoint(self.position, self.color)
        return {
            "position": mp.pt,
            "color": mp.color,
            "is_bad": mp.is_bad,
            "id": mp.id,
        }


class MapPointUpdatePositionTest(TestUnit):
    """Test MapPoint position update"""

    def __init__(self):
        super().__init__("MapPoint Update Position")
        self.initial_pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.new_pos = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        self.color = np.array([255, 128, 64], dtype=np.uint8)

    def run_test(self, module_type: str, module: Any) -> np.ndarray:
        mp = module.MapPoint(self.initial_pos, self.color)
        mp.update_position(self.new_pos)
        return mp.pt


# ============================================================================
# Test Units for CameraPose
# ============================================================================


class CameraPoseIdentityTest(TestUnit):
    """Test CameraPose identity initialization"""

    def __init__(self):
        super().__init__("CameraPose Identity")

    def run_test(self, module_type: str, module: Any) -> np.ndarray:
        pose = module.CameraPose()
        return pose.get_matrix()


class CameraPoseRotationTranslationTest(TestUnit):
    """Test CameraPose from rotation and translation"""

    def __init__(self):
        super().__init__("CameraPose from R,t")
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def run_test(self, module_type: str, module: Any) -> Dict:
        pose = module.CameraPose()
        pose.set_from_rotation_and_translation(self.R, self.t)
        return {
            "rotation": pose.get_rotation_matrix(),
            "position": pose.position,
            "matrix": pose.get_matrix(),
        }


class CameraPoseQuaternionTest(TestUnit):
    """Test CameraPose quaternion operations"""

    def __init__(self):
        super().__init__("CameraPose Quaternion")
        self.position = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def run_test(self, module_type: str, module: Any) -> Dict:
        pose = module.CameraPose()
        # Set identity rotation and given position
        pose.set_from_rotation_and_translation(np.eye(3), self.position)
        q = pose.quaternion
        # Create new pose from quaternion
        pose2 = module.CameraPose()
        pose2.set_from_quaternion_and_position(q, self.position)
        return {
            "quaternion": np.array([q.w(), q.x(), q.y(), q.z()]),  # Convert to numpy array
            "rotation": pose2.get_rotation_matrix(),
            "position": pose2.position,
        }


# ============================================================================
# Test Units for Camera
# ============================================================================


def _make_test_pinhole_camera(module_type: str, module: Any) -> Any:
    """Helper to create test camera"""
    cam_settings = {
        "Camera.width": 640,
        "Camera.height": 480,
        "Camera.fx": 500.0,
        "Camera.fy": 500.0,
        "Camera.cx": 320.0,
        "Camera.cy": 240.0,
        "Camera.fps": 30,
        "Camera.sensor_type": "monocular",
    }
    dataset_settings = {
        "sensor_type": "monocular",
    }
    config = {"cam_settings": cam_settings, "dataset_settings": dataset_settings}
    return module.PinholeCamera(config)


class CameraProjectionTest(TestUnit):
    """Test camera point projection"""

    def __init__(self):
        super().__init__("Camera Projection")
        self.test_points = np.array(
            [[0.0, 0.0, 2.0], [1.0, 1.0, 3.0], [-1.0, -1.0, 4.0]], dtype=np.float64
        )

    def run_test(self, module_type: str, module: Any) -> Tuple[np.ndarray, np.ndarray]:
        cam = _make_test_pinhole_camera(module_type, module)
        uv, zs = cam.project(self.test_points)
        return uv, zs


class CameraUnprojectionTest(TestUnit):
    """Test camera point unprojection"""

    def __init__(self):
        super().__init__("Camera Unprojection")
        self.uv_points = np.array(
            [[320.0, 240.0], [420.0, 340.0], [520.0, 440.0], [620.0, 540.0]], dtype=np.float64
        )
        self.depths = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    def run_test(self, module_type: str, module: Any) -> np.ndarray:
        cam = _make_test_pinhole_camera(module_type, module)
        return cam.unproject_points_3d(self.uv_points, self.depths)


class CameraInImageTest(TestUnit):
    """Test camera in-image bounds checking"""

    def __init__(self):
        super().__init__("Camera In Image")
        self.test_coords = [
            (320.0, 240.0),  # Center, should be in
            (0.0, 0.0),  # Corner, should be in
            (-10.0, 240.0),  # Outside, should be out
            (700.0, 240.0),  # Outside, should be out
        ]
        self.depth = 1.0

    def run_test(self, module_type: str, module: Any) -> List[bool]:
        cam = _make_test_pinhole_camera(module_type, module)
        return [cam.is_in_image(coord, self.depth) for coord in self.test_coords]


# ============================================================================
# Test Units for Frame
# ============================================================================


class FrameProjectionTest(TestUnit):
    """Test frame point projection"""

    def __init__(self):
        super().__init__("Frame Projection")
        self.world_point = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def run_test(self, module_type: str, module: Any) -> Tuple[np.ndarray, float]:
        cam = _make_test_pinhole_camera(module_type, module)
        # Create a dummy image for the Frame constructor
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = module.Frame(cam, dummy_img)
        frame.update_rotation_and_translation(self.R, self.t)
        return frame.project_point(self.world_point)


class FrameBatchProjectionTest(TestUnit):
    """Test frame batch point projection"""

    def __init__(self):
        super().__init__("Frame Batch Projection")
        self.world_points = np.random.uniform(-2, 2, (100, 3)).astype(np.float64)
        self.world_points[:, 2] = np.abs(self.world_points[:, 2]) + 1.0  # Ensure positive depth

    def run_test(self, module_type: str, module: Any) -> Tuple[np.ndarray, np.ndarray]:
        cam = _make_test_pinhole_camera(module_type, module)
        # Create a dummy image for the Frame constructor
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = module.Frame(cam, dummy_img)
        return frame.project_points(self.world_points)


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

    def execute(self, cpp_module, python_module) -> PerformanceResult:
        """Execute performance test"""
        self.setup()

        # Warmup
        try:
            self.run_benchmark("C++", cpp_module)
            self.run_benchmark("Python", python_module)
        except:
            pass

        # Benchmark C++
        cpp_times = []
        for _ in range(self.num_iterations):
            try:
                cpp_time = self.run_benchmark("C++", cpp_module)
                cpp_times.append(cpp_time)
            except Exception as e:
                print(f"C++ benchmark failed: {e}")
                cpp_times.append(float("inf"))
                traceback.print_exc()

        # Benchmark Python
        python_times = []
        for _ in range(self.num_iterations):
            try:
                python_time = self.run_benchmark("Python", python_module)
                python_times.append(python_time)
            except Exception as e:
                print(f"Python benchmark failed: {e}")
                python_times.append(float("inf"))
                traceback.print_exc()

        cpp_median = np.median(cpp_times)
        python_median = np.median(python_times)
        speedup = python_median / cpp_median if cpp_median > 0 else 0
        passed = cpp_median < python_median

        return PerformanceResult(
            name=self.name,
            cpp_time=cpp_median,
            python_time=python_median,
            speedup=speedup,
            passed=passed,
        )


class CameraProjectionPerformance(PerformanceTestUnit):
    """Performance test for camera projection"""

    def __init__(self):
        super().__init__("Camera Projection Performance")
        self.test_points = None

    def setup(self):
        np.random.seed(42)
        self.test_points = np.random.uniform(-5, 5, (5000, 3)).astype(np.float64)
        self.test_points[:, 2] = np.abs(self.test_points[:, 2]) + 1.0

    def run_benchmark(self, module_type: str, module: Any) -> float:
        cam = _make_test_pinhole_camera(module_type, module)
        start_time = time.perf_counter()
        uv, zs = cam.project(self.test_points)
        return time.perf_counter() - start_time


class CameraUnprojectionPerformance(PerformanceTestUnit):
    """Performance test for camera unprojection"""

    def __init__(self):
        super().__init__("Camera Unprojection Performance")
        self.uv_points = None
        self.depths = None

    def setup(self):
        np.random.seed(42)
        self.uv_points = np.random.uniform(0, 640, (5000, 2)).astype(np.float64)
        self.depths = np.random.uniform(0.5, 10.0, 5000).astype(np.float64)

    def run_benchmark(self, module_type: str, module: Any) -> float:
        cam = _make_test_pinhole_camera(module_type, module)
        start_time = time.perf_counter()
        xyz = cam.unproject_points_3d(self.uv_points, self.depths)
        return time.perf_counter() - start_time


class CameraPoseOperationsPerformance(PerformanceTestUnit):
    """Performance test for camera pose operations"""

    def __init__(self):
        super().__init__("CameraPose Operations Performance")
        self.num_poses = 1000

    def setup(self):
        np.random.seed(42)

    def run_benchmark(self, module_type: str, module: Any) -> float:
        start_time = time.perf_counter()
        for i in range(self.num_poses):
            pose = module.CameraPose()
            R = np.eye(3) + np.random.normal(0, 0.01, (3, 3))
            t = np.random.normal(0, 1, 3)
            pose.set_from_rotation_and_translation(R, t)
            _ = pose.get_matrix()
        return time.perf_counter() - start_time


class MapPointOperationsPerformance(PerformanceTestUnit):
    """Performance test for MapPoint operations"""

    def __init__(self):
        super().__init__("MapPoint Operations Performance")
        self.num_points = 1000

    def setup(self):
        np.random.seed(42)

    def run_benchmark(self, module_type: str, module: Any) -> float:
        start_time = time.perf_counter()
        for i in range(self.num_points):
            position = np.random.uniform(-10, 10, 3).astype(np.float64)
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            mp = module.MapPoint(position, color)
            new_pos = position + np.random.normal(0, 0.1, 3)
            mp.update_position(new_pos)
        return time.perf_counter() - start_time


class FrameProjectionPerformance(PerformanceTestUnit):
    """Performance test for frame projection operations"""

    def __init__(self):
        super().__init__("Frame Projection Performance")
        self.world_points = None

    def setup(self):
        np.random.seed(42)
        self.world_points = np.random.uniform(-5, 5, (5000, 3)).astype(np.float64)
        self.world_points[:, 2] = np.abs(self.world_points[:, 2]) + 1.0

    def run_benchmark(self, module_type: str, module: Any) -> float:
        cam = _make_test_pinhole_camera(module_type, module)
        random_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        frame = module.Frame(cam, random_img)
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        frame.update_rotation_and_translation(R, t)

        start_time = time.perf_counter()
        uv, depths = frame.project_points(self.world_points)
        return time.perf_counter() - start_time


# ============================================================================
# Test Suite Manager
# ============================================================================


class TestSuite:
    """Manages and executes all test units"""

    def __init__(self):
        self.functional_tests = []
        self.performance_tests = []
        self.setup_complete = False

    def setup_environment(self) -> bool:
        """Setup test environment"""
        try:
            # Setup FeatureTrackerShared
            tracker_config = FeatureTrackerConfigs.ORB2
            tracker_config["num_features"] = 1000
            feature_tracker = feature_tracker_factory(**tracker_config)
            FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

            print("✅ Test environment setup complete")
            self.setup_complete = True
            return True
        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            traceback.print_exc()
            return False

    def add_functional_test(self, test_unit: TestUnit):
        """Add a functional test unit"""
        self.functional_tests.append(test_unit)

    def add_performance_test(self, test_unit: PerformanceTestUnit):
        """Add a performance test unit"""
        self.performance_tests.append(test_unit)

    def run_functional_tests(self, cpp_module, python_module) -> List[TestResult]:
        """Run all functional tests"""
        print("\n" + "=" * 60)
        print("Running Functional Tests")
        print("=" * 60)

        results = []
        for test in self.functional_tests:
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

    def run_performance_tests(self, cpp_module, python_module) -> List[PerformanceResult]:
        """Run all performance tests"""
        print("\n" + "=" * 60)
        print("Running Performance Tests")
        print("=" * 60)

        results = []
        for test in self.performance_tests:
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
                print(
                    f"✅ PASSED - Speedup: {result.speedup:.1f}x (C++: {result.cpp_time:.4f}s, Python: {result.python_time:.4f}s)"
                )
            else:
                print(
                    f"❌ FAILED - C++ slower (C++: {result.cpp_time:.4f}s, Python: {result.python_time:.4f}s)"
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
        print(f"Performance Tests: {passed_performance}/{total_performance} passed")

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
                print(f"  {status} {result.name:30} {result.speedup:6.1f}x speedup")

        total_passed = passed_functional + passed_performance
        total_tests = total_functional + total_performance

        print(f"\nOverall: {total_passed}/{total_tests} tests passed")

        return total_passed == total_tests


def setup_test_suite() -> TestSuite:
    """Setup and configure the test suite"""
    suite = TestSuite()

    # Add functional tests
    suite.add_functional_test(MapPointCreationTest())
    suite.add_functional_test(MapPointUpdatePositionTest())
    suite.add_functional_test(CameraPoseIdentityTest())
    suite.add_functional_test(CameraPoseRotationTranslationTest())
    suite.add_functional_test(CameraPoseQuaternionTest())
    suite.add_functional_test(CameraProjectionTest())
    suite.add_functional_test(CameraUnprojectionTest())
    suite.add_functional_test(CameraInImageTest())
    suite.add_functional_test(FrameProjectionTest())
    suite.add_functional_test(FrameBatchProjectionTest())

    # Add performance tests
    suite.add_performance_test(CameraProjectionPerformance())
    suite.add_performance_test(CameraUnprojectionPerformance())
    suite.add_performance_test(CameraPoseOperationsPerformance())
    suite.add_performance_test(MapPointOperationsPerformance())
    suite.add_performance_test(FrameProjectionPerformance())

    return suite


def main():
    """Main test function"""
    print("PYSLAM C++ vs Python Systematic Test Suite")
    print("=" * 60)

    # Setup test suite
    suite = setup_test_suite()
    if not suite.setup_environment():
        print("❌ Cannot proceed without proper environment setup")
        return False

    try:
        # Run tests
        start_time = time.time()

        functional_results = suite.run_functional_tests(cpp_module, python_module)
        performance_results = suite.run_performance_tests(cpp_module, python_module)

        total_time = time.time() - start_time

        # Print summary
        all_passed = suite.print_summary(functional_results, performance_results)
        print(f"Total execution time: {total_time:.2f}s")

        if all_passed:
            print("\n🎉 All tests passed!")
            return True
        else:
            print("\n⚠️  Some tests failed!")
            return False

        print("✅ Test environment cleanup complete")
    except Exception as e:
        print(f"⚠️  Warning: Failed to cleanup test environment: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
