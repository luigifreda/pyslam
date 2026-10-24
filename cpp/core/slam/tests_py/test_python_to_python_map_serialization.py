#!/usr/bin/env python3
"""
Test script to verify Python save and reload functionality.
This script creates a map, saves it to JSON, reloads it, and verifies
that all data is correctly preserved through the save/reload cycle.
"""

import sys
import os
import json
import numpy as np

from pyslam.config import Config
from pyslam.config_parameters import Parameters

from pyslam.slam.map import Map
from pyslam.slam.frame import Frame, FrameBase
from pyslam.slam.keyframe import KeyFrame
from pyslam.slam.map_point import MapPoint, MapPointBase
from pyslam.slam.camera import Camera, PinholeCamera
from pyslam.utilities.serialization import convert_inf_nan
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


def create_test_map():
    """Create a test map with frames, keyframes, and map points."""
    print("Creating test map...")

    # Initialize feature tracker (required for Frame creation)
    if FeatureTrackerShared.feature_tracker is None:
        tracker_config = FeatureTrackerConfigs.ORB2.copy()
        tracker_config["num_features"] = 1000
        tracker_config["deterministic"] = True
        feature_tracker = feature_tracker_factory(**tracker_config)
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)
        print("Initialized feature tracker")

    # Create camera config
    camera_config = {
        "width": 640,
        "height": 480,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "fps": 30.0,
        "sensor_type": "mono",
    }

    # Create camera
    camera = PinholeCamera(config=camera_config)

    # Enable image storage for frames (required for serialization test)
    Frame.is_store_imgs = True

    # Create map
    map_obj = Map()

    # Create some keyframes with descriptors
    keyframes = []
    for i in range(5):
        # Create a dummy image with some texture/patterns so feature detection works
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some patterns to make feature detection work
        checker_size = 40
        for y in range(0, 480, checker_size):
            for x in range(0, 640, checker_size):
                if (x // checker_size + y // checker_size) % 2 == 0:
                    dummy_img[y : y + checker_size, x : x + checker_size] = 255
        # Add some random noise for more features
        noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        dummy_img = np.clip(dummy_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Create a frame
        frame = Frame(camera, img=dummy_img, id=i, timestamp=float(i) * 0.1, img_id=i)

        # Ensure we have enough keypoints and descriptors
        num_kps_detected = len(frame.kps) if frame.kps is not None else 0
        num_kps_target = 100

        if num_kps_detected < num_kps_target:
            num_kps_add = num_kps_target - num_kps_detected

            if frame.kps is None or len(frame.kps) == 0:
                frame.kps = np.random.rand(num_kps_add, 2).astype(np.float32) * np.array([640, 480])
                frame.kpsu = frame.kps.copy()
                frame.kpsn = frame.kps.copy()
                frame.octaves = np.random.randint(0, 8, num_kps_add).astype(np.int32)
                frame.sizes = np.random.rand(num_kps_add).astype(np.float32) * 10.0
                frame.angles = np.random.rand(num_kps_add).astype(np.float32) * 360.0
                frame.des = np.random.randint(0, 256, (num_kps_add, 32), dtype=np.uint8)
            else:
                new_kps = np.random.rand(num_kps_add, 2).astype(np.float32) * np.array([640, 480])
                frame.kps = np.vstack([frame.kps, new_kps])
                frame.kpsu = np.vstack([frame.kpsu, new_kps])
                frame.kpsn = np.vstack([frame.kpsn, new_kps])
                frame.octaves = np.hstack(
                    [frame.octaves, np.random.randint(0, 8, num_kps_add).astype(np.int32)]
                )
                frame.sizes = np.hstack(
                    [frame.sizes, np.random.rand(num_kps_add).astype(np.float32) * 10.0]
                )
                frame.angles = np.hstack(
                    [frame.angles, np.random.rand(num_kps_add).astype(np.float32) * 360.0]
                )
                new_des = np.random.randint(0, 256, (num_kps_add, 32), dtype=np.uint8)
                frame.des = np.vstack([frame.des, new_des])

            if frame.des.dtype != np.uint8:
                frame.des = frame.des.astype(np.uint8)

        # Create keyframe
        kf = KeyFrame(frame, kid=i)
        kf.is_keyframe = True
        kf._is_bad = False

        # Add to map
        map_obj.add_frame(frame, override_id=True)
        map_obj.add_keyframe(kf)
        keyframes.append(kf)

    # Create map points and associate them with keyframes
    map_points = []
    for i in range(20):
        pt = np.random.rand(3).astype(np.float64) * 10.0 - 5.0
        color = np.random.randint(0, 256, 3, dtype=np.uint8)

        kf_idx = i % len(keyframes)
        kf = keyframes[kf_idx]

        kp_idx = i % min(100, len(kf.kps)) if kf.kps is not None and len(kf.kps) > 0 else 0

        mp = MapPoint(pt, color, kf, kp_idx, i)
        mp.des = np.random.randint(0, 256, (1, 32), dtype=np.uint8)

        map_obj.add_point(mp)
        map_points.append(mp)

        if kf.points is None:
            num_kps = len(kf.kps) if kf.kps is not None else 100
            kf.points = [None] * num_kps
        else:
            num_kps = len(kf.kps) if kf.kps is not None else len(kf.points)
            if len(kf.points) < num_kps:
                kf.points.extend([None] * (num_kps - len(kf.points)))

        if len(kf.points) <= kp_idx:
            kf.points.extend([None] * (kp_idx + 1 - len(kf.points)))

        kf.points[kp_idx] = mp

        try:
            mp.add_observation(kf, kp_idx)
        except Exception as e:
            print(f"Warning: Could not add observation for map point {mp.id}: {e}")

    # Update max IDs
    map_obj.max_frame_id = 10
    map_obj.max_keyframe_id = 10
    map_obj.max_point_id = 20

    FrameBase._id = 10
    MapPointBase._id = 20

    print(
        f"\nCreated map with {len(map_obj.frames)} frames, {len(map_obj.keyframes)} keyframes, {len(map_obj.points)} points"
    )

    return map_obj


def save_map_to_json(map_obj, output_file):
    """Save map to JSON file."""
    print(f"\nSaving map to {output_file}...")
    map_json = map_obj.to_json()

    # Clean up inf/nan values
    map_json_clean = convert_inf_nan(map_json)

    # Write to file
    with open(output_file, "w") as f:
        json.dump(map_json_clean, f, indent=2)

    file_size = os.path.getsize(output_file)
    print(f"Map saved successfully ({file_size} bytes)")
    return map_json_clean


def load_map_from_json(map_file):
    """Load map from JSON file."""
    print(f"\nLoading map from {map_file}...")
    with open(map_file, "r") as f:
        json_data = json.load(f)

    map_obj = Map()
    map_obj.from_json(json_data)
    print("Map loaded successfully!")
    return map_obj


def verify_map_equality(original_map, reloaded_map):
    """Verify that the reloaded map matches the original map."""
    print("\n" + "=" * 60)
    print("Verifying map equality...")
    print("=" * 60)

    errors = []
    warnings = []

    # Check basic statistics
    print("\n1. Basic Statistics:")
    orig_frames = len(original_map.frames)
    reload_frames = len(reloaded_map.frames)
    orig_kfs = len(original_map.keyframes)
    reload_kfs = len(reloaded_map.keyframes)
    orig_points = len(original_map.points)
    reload_points = len(reloaded_map.points)

    print(f"  Frames: {orig_frames} -> {reload_frames}")
    print(f"  Keyframes: {orig_kfs} -> {reload_kfs}")
    print(f"  Points: {orig_points} -> {reload_points}")

    if orig_frames != reload_frames:
        errors.append(f"Frame count mismatch: {orig_frames} != {reload_frames}")
    if orig_kfs != reload_kfs:
        errors.append(f"Keyframe count mismatch: {orig_kfs} != {reload_kfs}")
    if orig_points != reload_points:
        errors.append(f"Point count mismatch: {orig_points} != {reload_points}")

    # Check max IDs
    print("\n2. Max IDs:")
    print(f"  max_frame_id: {original_map.max_frame_id} -> {reloaded_map.max_frame_id}")
    print(f"  max_keyframe_id: {original_map.max_keyframe_id} -> {reloaded_map.max_keyframe_id}")
    print(f"  max_point_id: {original_map.max_point_id} -> {reloaded_map.max_point_id}")

    if original_map.max_frame_id != reloaded_map.max_frame_id:
        errors.append(
            f"max_frame_id mismatch: {original_map.max_frame_id} != {reloaded_map.max_frame_id}"
        )
    if original_map.max_keyframe_id != reloaded_map.max_keyframe_id:
        errors.append(
            f"max_keyframe_id mismatch: {original_map.max_keyframe_id} != {reloaded_map.max_keyframe_id}"
        )
    if original_map.max_point_id != reloaded_map.max_point_id:
        errors.append(
            f"max_point_id mismatch: {original_map.max_point_id} != {reloaded_map.max_point_id}"
        )

    # Check frames
    print("\n3. Verifying Frames:")
    frame_ids_orig = {f.id for f in original_map.frames if f is not None}
    frame_ids_reload = {f.id for f in reloaded_map.frames if f is not None}

    if frame_ids_orig != frame_ids_reload:
        errors.append(f"Frame IDs mismatch: {frame_ids_orig} != {frame_ids_reload}")

    for frame_id in frame_ids_orig:
        orig_frame = next((f for f in original_map.frames if f.id == frame_id), None)
        reload_frame = next((f for f in reloaded_map.frames if f.id == frame_id), None)

        if orig_frame is None or reload_frame is None:
            errors.append(f"Frame {frame_id} missing in one of the maps")
            continue

        # Check basic frame properties
        if orig_frame.timestamp != reload_frame.timestamp:
            errors.append(
                f"Frame {frame_id} timestamp mismatch: {orig_frame.timestamp} != {reload_frame.timestamp}"
            )

        # Check keypoints
        if orig_frame.kps is not None and reload_frame.kps is not None:
            if not np.allclose(orig_frame.kps, reload_frame.kps, rtol=1e-5, atol=1e-6):
                errors.append(f"Frame {frame_id} keypoints mismatch")
        elif orig_frame.kps is None != reload_frame.kps is None:
            errors.append(f"Frame {frame_id} keypoints None mismatch")

        # Check descriptors
        if orig_frame.des is not None and reload_frame.des is not None:
            orig_des = np.asarray(orig_frame.des, dtype=np.uint8)
            reload_des = np.asarray(reload_frame.des, dtype=np.uint8)
            if orig_des.shape != reload_des.shape:
                errors.append(
                    f"Frame {frame_id} descriptor shape mismatch: {orig_des.shape} != {reload_des.shape}"
                )
            elif not np.array_equal(orig_des, reload_des):
                errors.append(f"Frame {frame_id} descriptors mismatch")
        elif orig_frame.des is None != reload_frame.des is None:
            errors.append(f"Frame {frame_id} descriptors None mismatch")

    # Check keyframes
    print("\n4. Verifying Keyframes:")
    kf_ids_orig = {kf.id for kf in original_map.keyframes if kf is not None}
    kf_ids_reload = {kf.id for kf in reloaded_map.keyframes if kf is not None}

    if kf_ids_orig != kf_ids_reload:
        errors.append(f"Keyframe IDs mismatch: {kf_ids_orig} != {kf_ids_reload}")

    for kf_id in kf_ids_orig:
        orig_kf = next((kf for kf in original_map.keyframes if kf.id == kf_id), None)
        reload_kf = next((kf for kf in reloaded_map.keyframes if kf.id == kf_id), None)

        if orig_kf is None or reload_kf is None:
            errors.append(f"Keyframe {kf_id} missing in one of the maps")
            continue

        # Check pose
        if orig_kf.Tcw() is not None and reload_kf.Tcw() is not None:
            if not np.allclose(orig_kf.Tcw(), reload_kf.Tcw(), rtol=1e-5, atol=1e-6):
                errors.append(f"Keyframe {kf_id} pose mismatch")

        # Check descriptors
        if orig_kf.des is not None and reload_kf.des is not None:
            orig_des = np.asarray(orig_kf.des, dtype=np.uint8)
            reload_des = np.asarray(reload_kf.des, dtype=np.uint8)
            if orig_des.shape != reload_des.shape:
                errors.append(
                    f"Keyframe {kf_id} descriptor shape mismatch: {orig_des.shape} != {reload_des.shape}"
                )
            elif not np.array_equal(orig_des, reload_des):
                errors.append(f"Keyframe {kf_id} descriptors mismatch")
        elif orig_kf.des is None != reload_kf.des is None:
            errors.append(f"Keyframe {kf_id} descriptors None mismatch")

    # Check map points
    print("\n5. Verifying Map Points:")
    point_ids_orig = {mp.id for mp in original_map.points if mp is not None}
    point_ids_reload = {mp.id for mp in reloaded_map.points if mp is not None}

    if point_ids_orig != point_ids_reload:
        errors.append(f"Map point IDs mismatch: {point_ids_orig} != {point_ids_reload}")

    for point_id in point_ids_orig:
        orig_mp = next((mp for mp in original_map.points if mp.id == point_id), None)
        reload_mp = next((mp for mp in reloaded_map.points if mp.id == point_id), None)

        if orig_mp is None or reload_mp is None:
            errors.append(f"Map point {point_id} missing in one of the maps")
            continue

        # Check 3D position (pt is a method, not a property)
        try:
            orig_pt = orig_mp.pt()
            reload_pt = reload_mp.pt()
            # Convert to numpy arrays if needed and ensure same dtype
            orig_pt = np.asarray(orig_pt, dtype=np.float64)
            reload_pt = np.asarray(reload_pt, dtype=np.float64)
            if not np.allclose(orig_pt, reload_pt, rtol=1e-5, atol=1e-6):
                errors.append(f"Map point {point_id} 3D position mismatch")
        except Exception as e:
            errors.append(f"Map point {point_id} 3D position access error: {e}")

        # Check color
        if orig_mp.color is not None and reload_mp.color is not None:
            orig_color = np.asarray(orig_mp.color, dtype=np.uint8)
            reload_color = np.asarray(reload_mp.color, dtype=np.uint8)
            if not np.array_equal(orig_color, reload_color):
                errors.append(f"Map point {point_id} color mismatch")
        elif orig_mp.color is None != reload_mp.color is None:
            errors.append(f"Map point {point_id} color None mismatch")

        # Check descriptors
        if orig_mp.des is not None and reload_mp.des is not None:
            orig_des = np.asarray(orig_mp.des, dtype=np.uint8)
            reload_des = np.asarray(reload_mp.des, dtype=np.uint8)
            if orig_des.shape != reload_des.shape:
                errors.append(
                    f"Map point {point_id} descriptor shape mismatch: {orig_des.shape} != {reload_des.shape}"
                )
            elif not np.array_equal(orig_des, reload_des):
                errors.append(f"Map point {point_id} descriptors mismatch")
        elif orig_mp.des is None != reload_mp.des is None:
            errors.append(f"Map point {point_id} descriptors None mismatch")

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print("=" * 60)
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ No errors found!")

    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return len(errors) == 0


def main():
    """Main test function."""
    print("=" * 60)
    print("Python Save/Reload Test")
    print("=" * 60)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "python_save_reload_test.json")

    # Step 1: Create map
    original_map = create_test_map()

    # Step 2: Save map
    save_map_to_json(original_map, output_file)

    # Step 3: Reload map
    reloaded_map = load_map_from_json(output_file)

    # Step 4: Verify equality
    success = verify_map_equality(original_map, reloaded_map)

    # Final result
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Python save/reload works correctly!")
    else:
        print("❌ TEST FAILED: Python save/reload has issues!")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
