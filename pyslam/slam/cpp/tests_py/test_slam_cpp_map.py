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
import cv2

import pyslam.config as config

# Force use of Python module (just to be sure)
os.environ["PYSLAM_USE_CPP"] = "false"  # set environment variable
import pyslam.slam as python_module

try:
    # Explicitly import the C++ module
    import pyslam.slam.cpp as cpp_module  # NOTE: this first import pyslam.slam and then pyslam.slam.cpp!

    if not cpp_module.CPP_AVAILABLE:
        print("❌ cpp_module imported successfully but C++ core is not available")
        sys.exit(1)
    print("✅ cpp_module imported successfully, C++ core is available")
except ImportError as e:
    print(f"❌ Failed to import C++ module: {e}")
    sys.exit(1)

from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from test_tools import TestSuite, TestUnit, PerformanceTestUnit, TestResult, PerformanceResult


# ============================================================================
# Test Units for Map
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


class MapCreationTest(TestUnit):
    """Test Map creation and basic properties"""

    def __init__(self):
        super().__init__("Map Creation")

    def run_test(self, module_type: str, module: Any) -> Dict:
        map_obj = module.Map()
        return {
            "num_points": map_obj.num_points(),
            "num_frames": map_obj.num_frames(),
            "num_keyframes": map_obj.num_keyframes(),
            "max_point_id": map_obj.max_point_id,
            "max_frame_id": map_obj.max_frame_id,
            "max_keyframe_id": map_obj.max_keyframe_id,
            "is_reloaded": map_obj.is_reloaded(),
        }


class MapPointOperationsTest(TestUnit):
    """Test Map point operations"""

    def __init__(self):
        super().__init__("Map Point Operations")
        self.position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.color = np.array([255, 128, 64], dtype=np.uint8)

    def run_test(self, module_type: str, module: Any) -> Dict:
        map_obj = module.Map()

        # Create a map point
        mp = module.MapPoint(self.position, self.color)

        # Test adding point
        point_id = map_obj.add_point(mp)

        # Test getting points
        points = map_obj.get_points()

        # Test point count
        num_points = map_obj.num_points()

        # Test removing point
        map_obj.remove_point(mp)
        num_points_after_remove = map_obj.num_points()

        return {
            "point_id": point_id,
            "num_points_before_remove": num_points,
            "num_points_after_remove": num_points_after_remove,
            "points_set_size": len(points),
            "max_point_id_after_add": map_obj.max_point_id,
        }


class MapFrameOperationsTest(TestUnit):
    """Test Map frame operations"""

    def __init__(self):
        super().__init__("Map Frame Operations")
        self.cam_settings = {
            "Camera.width": 640,
            "Camera.height": 480,
            "Camera.fx": 500.0,
            "Camera.fy": 500.0,
            "Camera.cx": 320.0,
            "Camera.cy": 240.0,
            "Camera.fps": 30,
            "Camera.sensor_type": "monocular",
        }
        self.dataset_settings = {
            "sensor_type": "monocular",
        }
        self.config = {"cam_settings": self.cam_settings, "dataset_settings": self.dataset_settings}

    def run_test(self, module_type: str, module: Any) -> Dict:
        map_obj = module.Map()

        # Create camera and frame
        cam = module.PinholeCamera(self.config)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = module.Frame(cam, dummy_img)

        # Test adding frame
        frame_id = map_obj.add_frame(frame)

        # Test getting frames
        frames = map_obj.get_frames()
        frame_by_idx = map_obj.get_frame(0)

        # Test frame count
        num_frames = map_obj.num_frames()

        # Test removing frame
        map_obj.remove_frame(frame)
        num_frames_after_remove = map_obj.num_frames()

        return {
            "frame_id": frame_id,
            "num_frames_before_remove": num_frames,
            "num_frames_after_remove": num_frames_after_remove,
            "frames_size": len(frames),
            "frame_by_idx_valid": frame_by_idx is not None,
            "max_frame_id_after_add": map_obj.max_frame_id,
        }


class MapKeyFrameOperationsTest(TestUnit):
    """Test Map keyframe operations"""

    kScriptPath = os.path.realpath(__file__)
    kScriptFolder = os.path.dirname(kScriptPath)
    kRootFolder = kScriptFolder + "/../../../.."
    kDataFolder = kRootFolder + "/test/data"

    def __init__(self):
        super().__init__("Map KeyFrame Operations")

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

    def run_test(self, module_type: str, module: Any) -> Dict:
        map_obj = module.Map()

        # Create camera and keyframe
        cam = module.PinholeCamera(self.config)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = module.Frame(cam, dummy_img)

        keyframe = module.KeyFrame(frame, dummy_img)

        # Test adding keyframe
        keyframe_id = map_obj.add_keyframe(keyframe)

        # Test getting keyframes
        keyframes = map_obj.get_keyframes()
        last_keyframe = map_obj.get_last_keyframe()
        last_keyframes = map_obj.get_last_keyframes(3)

        # Test keyframe count
        num_keyframes = map_obj.num_keyframes()
        num_keyframes_session = map_obj.num_keyframes_session()

        # Test removing keyframe
        map_obj.remove_keyframe(keyframe)
        num_keyframes_after_remove = map_obj.num_keyframes()

        return {
            "keyframe_id": keyframe_id,
            "num_keyframes_before_remove": num_keyframes,
            "num_keyframes_after_remove": num_keyframes_after_remove,
            "num_keyframes_session": num_keyframes_session,
            "keyframes_set_size": len(keyframes),
            "last_keyframe_valid": last_keyframe is not None,
            "last_keyframes_size": len(last_keyframes),
            "max_keyframe_id_after_add": map_obj.max_keyframe_id,
        }


class MapResetTest(TestUnit):
    """Test Map reset operations"""

    def __init__(self):
        super().__init__("Map Reset Operations")
        self.cam_settings = {
            "Camera.width": 640,
            "Camera.height": 480,
            "Camera.fx": 500.0,
            "Camera.fy": 500.0,
            "Camera.cx": 320.0,
            "Camera.cy": 240.0,
            "Camera.fps": 30,
            "Camera.sensor_type": "monocular",
        }
        self.dataset_settings = {
            "sensor_type": "monocular",
        }
        self.config = {"cam_settings": self.cam_settings, "dataset_settings": self.dataset_settings}

    def run_test(self, module_type: str, module: Any) -> Dict:
        map_obj = module.Map()

        # Add some data
        cam = module.PinholeCamera(self.config)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add frame
        frame = module.Frame(cam, dummy_img)
        map_obj.add_frame(frame)

        # Add keyframe
        keyframe = module.KeyFrame(frame, dummy_img)
        map_obj.add_keyframe(keyframe)

        # Add point
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        color = np.array([255, 128, 64], dtype=np.uint8)
        mp = module.MapPoint(position, color)
        map_obj.add_point(mp)

        # Record state before reset
        state_before = {
            "num_frames": map_obj.num_frames(),
            "num_keyframes": map_obj.num_keyframes(),
            "num_points": map_obj.num_points(),
        }

        # Test reset
        map_obj.reset()

        # Record state after reset
        state_after = {
            "num_frames": map_obj.num_frames(),
            "num_keyframes": map_obj.num_keyframes(),
            "num_points": map_obj.num_points(),
        }

        return {
            "before_reset": state_before,
            "after_reset": state_after,
            "reset_successful": all(v == 0 for v in state_after.values()),
        }


# NOTE: this must be completed and fixed
# class MapSessionManagementTest(TestUnit):
#     """Test Map session management"""

#     def __init__(self):
#         super().__init__("Map Session Management")

#     def run_test(self, module_type: str, module: Any) -> Dict:
#         map_obj = module.Map()

#         # Test initial state
#         is_reloaded_initial = map_obj.is_reloaded()
#         reloaded_info_initial = map_obj.get_reloaded_session_info()

#         # Test setting reloaded session info
#         session_info = module.ReloadedSessionMapInfo(
#             num_keyframes=5, num_points=100, max_point_id=50, max_frame_id=30, max_keyframe_id=10
#         )
#         map_obj.set_reloaded_session_info(session_info)

#         # Test state after setting info
#         is_reloaded_after = map_obj.is_reloaded()
#         reloaded_info_after = map_obj.get_reloaded_session_info()

#         # Test reset_session
#         map_obj.reset_session()

#         return {
#             "is_reloaded_initial": is_reloaded_initial,
#             "reloaded_info_initial": reloaded_info_initial is not None,
#             "is_reloaded_after": is_reloaded_after,
#             "reloaded_info_after": reloaded_info_after is not None,
#             "session_info_valid": reloaded_info_after is not None,
#         }


# ============================================================================
# Performance Test Units for Map
# ============================================================================


class MapPointOperationsPerformance(PerformanceTestUnit):
    """Performance test for Map point operations"""

    def __init__(self):
        super().__init__("Map Point Operations Performance")
        self.num_points = 1000

    def setup(self):
        np.random.seed(42)

    def run_benchmark(self, module_type: str, module: Any) -> float:
        map_obj = module.Map()

        start_time = time.perf_counter()

        # Add many points
        for i in range(self.num_points):
            position = np.random.uniform(-10, 10, 3).astype(np.float64)
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            mp = module.MapPoint(position, color)
            map_obj.add_point(mp)

        # Get points multiple times
        for _ in range(10):
            _ = map_obj.get_points()
            _ = map_obj.num_points()

        return time.perf_counter() - start_time


class MapFrameOperationsPerformance(PerformanceTestUnit):
    """Performance test for Map frame operations"""

    def __init__(self):
        super().__init__("Map Frame Operations Performance")
        self.num_frames = 500

    def setup(self):
        np.random.seed(42)
        self.cam_settings = {
            "Camera.width": 640,
            "Camera.height": 480,
            "Camera.fx": 500.0,
            "Camera.fy": 500.0,
            "Camera.cx": 320.0,
            "Camera.cy": 240.0,
            "Camera.fps": 30,
            "Camera.sensor_type": "monocular",
        }
        self.dataset_settings = {
            "sensor_type": "monocular",
        }
        self.config = {"cam_settings": self.cam_settings, "dataset_settings": self.dataset_settings}

    def run_benchmark(self, module_type: str, module: Any) -> float:
        map_obj = module.Map()
        cam = module.PinholeCamera(self.config)

        start_time = time.perf_counter()

        # Add many frames
        for i in range(self.num_frames):
            dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            frame = module.Frame(cam, dummy_img)
            map_obj.add_frame(frame)

        # Get frames multiple times
        for _ in range(10):
            _ = map_obj.get_frames()
            _ = map_obj.num_frames()

        return time.perf_counter() - start_time


class MapKeyFrameOperationsPerformance(PerformanceTestUnit):
    """Performance test for Map keyframe operations"""

    def __init__(self):
        super().__init__("Map KeyFrame Operations Performance")
        self.num_keyframes = 200

    def setup(self):
        np.random.seed(42)
        self.cam_settings = {
            "Camera.width": 640,
            "Camera.height": 480,
            "Camera.fx": 500.0,
            "Camera.fy": 500.0,
            "Camera.cx": 320.0,
            "Camera.cy": 240.0,
            "Camera.fps": 30,
            "Camera.sensor_type": "monocular",
        }
        self.dataset_settings = {
            "sensor_type": "monocular",
        }
        self.config = {"cam_settings": self.cam_settings, "dataset_settings": self.dataset_settings}

    def run_benchmark(self, module_type: str, module: Any) -> float:
        map_obj = module.Map()
        cam = module.PinholeCamera(self.config)

        start_time = time.perf_counter()

        # Add many keyframes
        for i in range(self.num_keyframes):
            dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            keyframe = module.KeyFrame(cam, dummy_img)
            map_obj.add_keyframe(keyframe)

        # Get keyframes multiple times
        for _ in range(10):
            _ = map_obj.get_keyframes()
            _ = map_obj.num_keyframes()
            _ = map_obj.get_last_keyframe()

        return time.perf_counter() - start_time


# ============================================================================
# Test Suite Setup
# ============================================================================


def setup_test_suite() -> TestSuite:
    """Setup and configure the test suite"""
    suite = TestSuite()

    # Add Frame extraction test
    suite.add_functional_test(FrameExtractionTest())

    # Add Map functional tests
    suite.add_functional_test(MapCreationTest())
    suite.add_functional_test(MapPointOperationsTest())
    suite.add_functional_test(MapFrameOperationsTest())
    suite.add_functional_test(MapKeyFrameOperationsTest())
    suite.add_functional_test(MapResetTest())
    # suite.add_functional_test(MapSessionManagementTest()) # to be fixed

    # Add Map performance tests
    suite.add_performance_test(MapPointOperationsPerformance())
    suite.add_performance_test(MapFrameOperationsPerformance())
    suite.add_performance_test(MapKeyFrameOperationsPerformance())

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
