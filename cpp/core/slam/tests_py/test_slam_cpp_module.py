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
import cv2

import pyslam.config as config
from pyslam.config_parameters import Parameters

Parameters.USE_CPP_CORE = True

from pyslam.slam.cpp import cpp_module, python_module, CPP_AVAILABLE

if not CPP_AVAILABLE:
    print("❌ cpp_module imported successfully but C++ core is not available")
    sys.exit(1)
else:
    print("✅ cpp_module imported successfully")

from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from test_tools import TestSuite, TestUnit, PerformanceTestUnit, TestResult, PerformanceResult


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
            "position": mp.pt(),
            "color": mp.color,
            "is_bad": mp.is_bad(),
            # "id": mp.id,  # Remove ID from comparison as it's module-specific
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
        return mp.pt()


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
        "Camera.sensor_type": "mono",
    }
    dataset_settings = {
        "sensor_type": "mono",
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
# Test Units for Frame Matching
# ============================================================================


class FrameExtractionTest(TestUnit):
    """Test frame extraction"""

    kScriptPath = os.path.realpath(__file__)
    kScriptFolder = os.path.dirname(kScriptPath)
    kRootFolder = kScriptFolder + "/../../../.."
    kDataFolder = kRootFolder + "/test/data"

    def __init__(self):
        super().__init__("Frame Extraction", detailed=True)

        from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
        from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
        from pyslam.local_features.feature_tracker import feature_tracker_factory

        self.config = config.cfg

        # forced camera settings to be kept coherent with the input file below
        self.config.config[self.config.dataset_type]["settings"] = "settings/KITTI04-12.yaml"
        self.config.sensor_type = "mono"
        self.config.get_general_system_settings()  # parse again the settings file

        tracker_config = FeatureTrackerConfigs.ORB2
        tracker_config["num_features"] = 1000
        tracker_config["deterministic"] = True
        feature_tracker = feature_tracker_factory(**tracker_config)
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

        self.img_ref = cv2.imread(f"{self.kDataFolder}/kitti06-12.png", cv2.IMREAD_COLOR)
        self.img_cur = cv2.imread(f"{self.kDataFolder}/kitti06-13.png", cv2.IMREAD_COLOR)

    def run_test(self, module_type: str, module: Any) -> Tuple[np.ndarray, np.ndarray]:
        cam = module.PinholeCamera(self.config)
        f_ref = module.Frame(cam, self.img_ref.copy(), img_id=0)
        f_cur = module.Frame(cam, self.img_cur.copy(), img_id=1)

        # Debug information
        if module_type == "C++":
            print(f"\n🔍 {module_type} Frame Debug:")
            print(f"  Ref first KP: {f_ref.kps[0] if len(f_ref.kps) > 0 else 'None'}")
            print(f"  Cur first KP: {f_cur.kps[0] if len(f_cur.kps) > 0 else 'None'}")
            print(f"  Ref first Des: {f_ref.des[0][:5] if len(f_ref.des) > 0 else 'None'}")
            print(f"  Cur first Des: {f_cur.des[0][:5] if len(f_cur.des) > 0 else 'None'}")
        elif module_type == "Python":
            print(f"\n🔍 {module_type} Frame Debug:")
            print(f"  Ref first KP: {f_ref.kps[0] if len(f_ref.kps) > 0 else 'None'}")
            print(f"  Cur first KP: {f_cur.kps[0] if len(f_cur.kps) > 0 else 'None'}")
            print(f"  Ref first Des: {f_ref.des[0][:5] if len(f_ref.des) > 0 else 'None'}")
            print(f"  Cur first Des: {f_cur.des[0][:5] if len(f_cur.des) > 0 else 'None'}")

        result = {
            "kps_ref": f_ref.kps,
            "kps_cur": f_cur.kps,
            "des_ref": f_ref.des,
            "des_cur": f_cur.des,
        }
        return result


# ============================================================================
# Test Units for Sim3Pose
# ============================================================================


class Sim3PoseTest(TestUnit):
    """Test Sim3Pose operations"""

    def __init__(self):
        super().__init__("Sim3Pose")
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.s = 1.0

    def run_test(self, module_type: str, module: Any) -> Dict:
        pose = module.Sim3Pose()
        sim3_T = np.eye(4, dtype=np.float64)
        sim3_T[:3, :3] = self.s * self.R
        sim3_T[:3, 3] = self.t
        pose.from_se3_matrix(sim3_T)
        return pose.matrix()


# ============================================================================
# Performance Test Units
# ============================================================================


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
# Test Suite Setup
# ============================================================================


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
    suite.add_functional_test(FrameExtractionTest())
    suite.add_functional_test(Sim3PoseTest())

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
    # if not suite.setup_environment():
    #     print("❌ Cannot proceed without proper environment setup")
    #     return False

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
