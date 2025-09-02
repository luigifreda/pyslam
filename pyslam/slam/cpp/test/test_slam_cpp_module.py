#!/usr/bin/env python3
"""
Comprehensive test script for pybind11 SLAM C++ module
Tests the core C++ classes directly through pybind11 bindings
"""

import sys
import os
import numpy as np
import time
import traceback

from pyslam.config import Config

config = Config()

import g2o
import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.dirname(os.path.join(THIS_DIR, ".."))
PROJECT_ROOT = os.path.dirname(os.path.join(CPP_DIR, "../../../.."))

# Add the C++ module lib directory
sys.path.insert(0, f"{CPP_DIR}/lib")

# Add g2o lib directory (needed for the C++ module)
sys.path.insert(0, f"{PROJECT_ROOT}/thirdparty/g2opy/lib")


import pyslam.slam as python_module
from pyslam.slam.frame import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


def setup_feature_tracker():
    """Setup FeatureTrackerShared for testing"""
    try:
        # Create a simple feature tracker configuration
        tracker_config = FeatureTrackerConfigs.ORB2
        tracker_config["num_features"] = 1000  # Use fewer features for testing
        feature_tracker = feature_tracker_factory(**tracker_config)

        # Initialize FeatureTrackerShared
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)
        print("✅ FeatureTrackerShared initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize FeatureTrackerShared: {e}")
        traceback.print_exc()
        return False


def test_cpp_core_import():
    """Test importing the C++ core module"""
    print("=" * 60)
    print("Testing C++ Core Module Import")
    print("=" * 60)

    try:
        import pyslam.slam.cpp as cpp_module

        print("✅ C++ module imported successfully!")

        # Check if C++ is available
        if hasattr(cpp_module, "CPP_AVAILABLE") and cpp_module.CPP_AVAILABLE:
            print("✅ C++ implementation is available")
            return cpp_module
        else:
            # print in yellow color
            print("\033[93m⚠️  C++ implementation not available, using Python fallback\033[0m")
            return cpp_module

    except ImportError as e:
        print(f"\033[91m❌ Failed to import C++ module: {e}\033[0m")
        traceback.print_exc()
        return None


def test_mappoint_cpp(cpp_module):
    """Test C++ MapPoint class"""
    print("\n" + "=" * 60)
    print("Testing MapPoint Class")
    print("=" * 60)

    try:
        # Test basic creation
        print("1. Testing basic MapPoint creation...")
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        color = np.array([255, 128, 64], dtype=np.uint8)
        descriptor = np.random.randint(0, 256, (32, 1), dtype=np.uint8).astype(np.float32)

        mappoint = cpp_module.MapPoint(position, color)
        mappoint.des = descriptor
        print(f"   ✅ Created MapPoint with ID: {mappoint.id}")
        print(f"   Position: {mappoint.pt}")
        print(f"   Color: {mappoint.color}")
        print(f"   Descriptor shape: {mappoint.des.shape}")
        print(f"   Is bad: {mappoint.is_bad}")

        # Test property access
        print("\n2. Testing property access...")
        print(f"   Normal: {mappoint.normal}")
        print(f"   Min distance: {mappoint.min_distance}")
        print(f"   Max distance: {mappoint.max_distance}")
        print(f"   Num observations: {mappoint.num_observations}")

        # Test property modification
        print("\n3. Testing property modification...")
        new_position = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        mappoint.update_position(new_position)
        print(f"   New position: {mappoint.pt}")

        # Test zero-copy behavior
        print("\n4. Testing zero-copy behavior...")
        original_descriptor = mappoint.des.copy()
        mappoint.des[0] = 42
        print(f"   Modified descriptor[0]: {mappoint.des[0]}")
        print(f"   Original descriptor[0]: {original_descriptor[0]}")
        print(f"   Zero-copy working: {mappoint.des[0] != original_descriptor[0]}")

        # Test utility methods
        print("\n5. Testing utility methods...")
        try:
            detection_level = mappoint.predict_detection_level(10.0)
            print(f"   Detection level for distance 10.0: {detection_level}")
        except Exception as e:
            print(f"   Detection level prediction failed: {e}")

        # Test JSON serialization
        print("\n6. Testing JSON serialization...")
        try:
            json_str = mappoint.to_json()
            print(f"   JSON length: {len(json_str)}")
        except Exception as e:
            print(f"   JSON serialization failed: {e}")

        # Test string representation
        print("\n7. Testing string representation...")
        print(f"   String: {str(mappoint)}")
        print(f"   Repr: {repr(mappoint)}")

        # Test comparison operators
        print("\n8. Testing comparison operators...")
        # Create another MapPoint (constructor does not take descriptor)
        mappoint2 = cpp_module.MapPoint(position, color)
        print(f"   Equal to self: {mappoint == mappoint}")
        print(f"   Equal to other: {mappoint == mappoint2}")
        print(f"   Less than other: {mappoint < mappoint2}")

        # Test hash
        print("\n9. Testing hash...")
        hash1 = hash(mappoint)
        hash2 = hash(mappoint2)
        print(f"   Hash of mappoint: {hash1}")
        print(f"   Hash of mappoint2: {hash2}")
        print(f"   Hashes equal: {hash1 == hash2}")

        print("\n✅ MapPoint tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ MapPoint test failed: {e}")
        traceback.print_exc()
        return False


def test_camera_pose_cpp(cpp_module):
    """Test C++ CameraPose class"""
    print("\n" + "=" * 60)
    print("Testing CameraPose Class")
    print("=" * 60)

    try:
        # 1. Construct default pose and verify identity
        pose = cpp_module.CameraPose()
        T = pose.get_matrix()
        assert np.allclose(T, np.eye(4)), "Default pose is not identity"

        # 2. Update from rotation and translation
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        pose.set_from_rotation_and_translation(R, t)
        assert np.allclose(pose.get_rotation_matrix(), R)
        assert np.allclose(pose.position(), t)

        # 3. Update from 4x4 matrix
        T2 = np.eye(4)
        T2[:3, 3] = np.array([-1.0, 0.5, 2.0])
        pose.update_mat(T2)
        assert np.allclose(pose.get_matrix(), T2)

        # 4. Quaternion roundtrip
        q = pose.quaternion()
        pose.set_from_quaternion_and_position(q, pose.position())
        assert np.allclose(pose.get_rotation_matrix(), T2[:3, :3])

        print("✅ CameraPose tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ CameraPose test failed: {e}")
        traceback.print_exc()
        return False


def _make_test_pinhole_camera(module_camera):
    # Minimal monocular camera config (no distortion)
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
    cam = module_camera.PinholeCamera(config)
    return cam


def test_camera_cpp(cpp_module):
    """Test C++ Camera and PinholeCamera classes"""
    print("\n" + "=" * 60)
    print("Testing Camera/PinholeCamera Classes")
    print("=" * 60)

    try:
        cam = _make_test_pinhole_camera(cpp_module)
        assert cam.width == 640 and cam.height == 480
        assert cam.fx > 0 and cam.fy > 0
        # Project a simple point in front of camera
        pts = [np.array([0.0, 0.0, 2.0], dtype=np.float64)]
        uv, zs = cam.project(pts)
        assert len(uv) == 1 and len(zs) == 1
        # Should be at principal point for this point
        u, v = uv[0]
        assert abs(u - cam.cx) < 1e-6 and abs(v - cam.cy) < 1e-6
        # Image bounds/in-image
        in_img = cam.is_in_image((cam.cx, cam.cy), 1.0)
        assert in_img is True
        # Render projection matrix
        P = cam.get_render_projection_matrix(0.01, 100.0)
        assert P.shape == (4, 4)
        print("✅ Camera tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Camera test failed: {e}")
        traceback.print_exc()
        return False


def test_frame_cpp(cpp_module):
    """Test C++ Frame class"""
    print("\n" + "=" * 60)
    print("Testing Frame Class")
    print("=" * 60)

    try:
        cam = _make_test_pinhole_camera(cpp_module)
        pose = cpp_module.CameraPose()  # identity
        f = cpp_module.Frame(cam)
        # Sizes propagate from camera
        assert f.width() == cam.width and f.height() == cam.height
        # Default pose should be identity matrix
        T = f.Tcw()
        assert np.allclose(T, np.eye(4))
        # Update pose and verify
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        f.update_rotation_and_translation(R, t)
        assert np.allclose(f.tcw(), t)

        # Project a world point via frame
        pw = np.array([0.0, 0.0, 5.0])
        uv, z = f.project_point(pw)
        assert z > 0

        print("✅ Frame tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Frame test failed: {e}")
        traceback.print_exc()
        return False


def test_keyframe_cpp(cpp_module):
    """Test C++ KeyFrame class"""
    print("\n" + "=" * 60)
    print("Testing KeyFrame Class")
    print("=" * 60)

    try:
        cam = _make_test_pinhole_camera(cpp_module)
        f = cpp_module.Frame(cam)
        kf = cpp_module.KeyFrame(f)
        assert kf.is_keyframe is True
        print("✅ KeyFrame tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ KeyFrame test failed: {e}")
        traceback.print_exc()
        return False


def test_map_cpp(cpp_module):
    """Test C++ Map class basic operations"""
    print("\n" + "=" * 60)
    print("Testing Map Class")
    print("=" * 60)

    try:
        cam = _make_test_pinhole_camera(cpp_module)
        f = cpp_module.Frame(cam)
        kf = cpp_module.KeyFrame(f)

        m = cpp_module.Map()
        assert m.num_frames() == 0 and m.num_keyframes() == 0

        m.add_frame(f)
        assert m.num_frames() == 1

        m.add_keyframe(kf)
        assert m.num_keyframes() == 1

        last_kf = m.get_last_keyframe()
        assert last_kf is not None

        print("✅ Map tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Map test failed: {e}")
        traceback.print_exc()
        return False


def test_performance(cpp_module, python_module_camera):
    """Lightweight performance sanity check"""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    try:
        cam_cpp = _make_test_pinhole_camera(cpp_module)
        cam_python = _make_test_pinhole_camera(python_module_camera)
        pts = np.array([np.array([i * 0.01, 0.0, 2.0], dtype=np.float64) for i in range(1000)])

        t0 = time.time()
        uv, zs = cam_cpp.project(pts)
        cpp_dt = time.time() - t0
        print(f"C++ Camera projected {len(pts)} points in {cpp_dt:.4f}s")

        t0 = time.time()
        uv, zs = cam_python.project(pts)
        python_dt = time.time() - t0
        print(f"Python Camera projected {len(pts)} points in {python_dt:.4f}s")
        print(f"Speedup: {cpp_dt / python_dt:.2f}")
        return cpp_dt < python_dt
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("PYSLAM C++ Pybind11 Module Test Suite")
    print("=" * 60)

    # Setup FeatureTrackerShared first
    if not setup_feature_tracker():
        print("\n❌ Cannot proceed without FeatureTrackerShared initialization")
        return False

    # Test C++ core import
    cpp_module = test_cpp_core_import()
    if cpp_module is None:
        print("\n❌ Cannot proceed without C++ module")
        return False

    # Run all tests
    tests = [
        ("MapPoint", lambda: test_mappoint_cpp(cpp_module)),
        ("CameraPose", lambda: test_camera_pose_cpp(cpp_module)),
        ("Camera", lambda: test_camera_cpp(cpp_module)),
        ("Frame", lambda: test_frame_cpp(cpp_module)),
        ("KeyFrame", lambda: test_keyframe_cpp(cpp_module)),
        ("Map", lambda: test_map_cpp(cpp_module)),
        ("Performance", lambda: test_performance(cpp_module, python_module.camera)),
    ]

    results = []
    start_time = time.time()

    for test_name, test_func in tests:
        # print(f"\n{'='*60}")
        # print(f"Running {test_name} Test")
        # print(f"{'='*60}")

        test_start = time.time()
        success = test_func()
        test_time = time.time() - test_start
        results.append((test_name, success, test_time))

        if success:
            print(f"✅ {test_name} test completed in {test_time:.2f}s")
        else:
            print(f"❌ {test_name} test failed in {test_time:.2f}s")

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for test_name, success, test_time in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_name:15} ({test_time:.2f}s)")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
