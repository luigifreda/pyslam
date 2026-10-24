#!/usr/bin/env python3
"""
Test script to create a Python-saved map that can be loaded by C++.
This script creates a map with frames, keyframes, and map points,
saves it to JSON, and then verifies C++ can load it.
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
from pyslam.slam.map_point import MapPoint
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
    # This ensures images are stored in Frame objects and can be serialized
    Frame.is_store_imgs = True

    # Create map
    map_obj = Map()

    # Create some keyframes with descriptors
    keyframes = []
    for i in range(5):
        # Create a dummy image with some texture/patterns so feature detection works
        # This ensures descriptors are actually generated from the image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some patterns to make feature detection work
        # Create a checkerboard pattern
        checker_size = 40
        for y in range(0, 480, checker_size):
            for x in range(0, 640, checker_size):
                if (x // checker_size + y // checker_size) % 2 == 0:
                    dummy_img[y : y + checker_size, x : x + checker_size] = 255
        # Add some random noise for more features
        noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        dummy_img = np.clip(dummy_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Create a frame (this will call detectAndCompute internally)
        frame = Frame(camera, img=dummy_img, id=i, timestamp=float(i) * 0.1, img_id=i)

        # If detectAndCompute didn't find enough features, supplement with manual ones
        # This ensures we always have descriptors for the test
        num_kps_detected = len(frame.kps) if frame.kps is not None else 0
        num_kps_target = 100

        if num_kps_detected < num_kps_target:
            # Supplement with additional manually created keypoints and descriptors
            num_kps_add = num_kps_target - num_kps_detected

            # Get existing data or create new arrays
            if frame.kps is None or len(frame.kps) == 0:
                frame.kps = np.zeros((num_kps_add, 2), dtype=np.float32)
                frame.kpsu = np.zeros((num_kps_add, 2), dtype=np.float32)
                frame.kpsn = np.zeros((num_kps_add, 2), dtype=np.float32)
                frame.octaves = np.zeros(num_kps_add, dtype=np.int32)
                frame.sizes = np.zeros(num_kps_add, dtype=np.float32)
                frame.angles = np.zeros(num_kps_add, dtype=np.float32)
                frame.des = np.zeros((num_kps_add, 32), dtype=np.uint8)
            else:
                # Append to existing arrays
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
                # Create ORB descriptors (32 bytes each) for the additional keypoints
                new_des = np.random.randint(0, 256, (num_kps_add, 32), dtype=np.uint8)
                frame.des = np.vstack([frame.des, new_des])

            # Ensure descriptor type is correct (ORB uses uint8)
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
    # This is critical for relocalization - keyframes need map points at keypoint indices
    map_points = []
    for i in range(20):
        # Random 3D position
        pt = np.random.rand(3).astype(np.float64) * 10.0 - 5.0
        color = np.random.randint(0, 256, 3, dtype=np.uint8)

        # Select a keyframe to associate this point with
        kf_idx = i % len(keyframes)
        kf = keyframes[kf_idx]

        # Select a keypoint index in this keyframe (ensure it's valid)
        kp_idx = i % min(100, len(kf.kps)) if kf.kps is not None and len(kf.kps) > 0 else 0

        # Create map point associated with the keyframe
        mp = MapPoint(pt, color, kf, kp_idx, i)

        # Add descriptor
        mp.des = np.random.randint(0, 256, (1, 32), dtype=np.uint8)

        # Add to map
        map_obj.add_point(mp)
        map_points.append(mp)

        # CRITICAL: Associate the map point with the keyframe at the specific keypoint index
        # This is what makes relocalization work - keyframes.points[kp_idx] should point to the map point
        if kf.points is None:
            # Initialize points array if needed - must match keypoints size
            num_kps = len(kf.kps) if kf.kps is not None else 100
            kf.points = [None] * num_kps
        else:
            # Ensure points array matches keypoints size
            num_kps = len(kf.kps) if kf.kps is not None else len(kf.points)
            if len(kf.points) < num_kps:
                # Extend points array to match keypoints size
                kf.points.extend([None] * (num_kps - len(kf.points)))

        # Ensure points array is large enough for this specific index
        if len(kf.points) <= kp_idx:
            kf.points.extend([None] * (kp_idx + 1 - len(kf.points)))

        # Associate map point with keyframe at keypoint index
        kf.points[kp_idx] = mp

        # Debug: Verify association
        if kf.points[kp_idx] != mp:
            print(
                f"ERROR: Failed to associate map point {mp.id} with keyframe {kf.id} at keypoint index {kp_idx}"
            )

        # Also add observation to map point (bidirectional relationship)
        # This ensures the map point knows it's observed by this keyframe
        try:
            mp.add_observation(kf, kp_idx)
        except Exception as e:
            # If add_observation fails, at least we have the points array set
            print(f"Warning: Could not add observation for map point {mp.id}: {e}")

    # Update max IDs
    map_obj.max_frame_id = 10
    map_obj.max_keyframe_id = 10
    map_obj.max_point_id = 20

    # Set FrameBase._id and MapPointBase._id
    FrameBase._id = 10
    from pyslam.slam.map_point import MapPointBase

    MapPointBase._id = 20

    # Verify associations before saving
    print("\nVerifying map point associations...")
    for kf in keyframes:
        if kf.points is None:
            print(f"  WARNING: Keyframe {kf.id} has no points array!")
        else:
            num_associated = sum(1 for p in kf.points if p is not None)
            print(
                f"  Keyframe {kf.id}: {num_associated} map points associated out of {len(kf.points)} keypoints"
            )

    print(
        f"\nCreated map with {len(map_obj.frames)} frames, {len(map_obj.keyframes)} keyframes, {len(map_obj.points)} points"
    )

    return map_obj


def save_map_to_json(map_obj, output_file):
    """Save map to JSON file."""
    print(f"Saving map to {output_file}...")

    # Get JSON representation
    map_json = map_obj.to_json()

    # Convert inf/nan to None for JSON compatibility
    map_json_clean = convert_inf_nan(map_json)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(map_json_clean, f, indent=2)

    print(f"Map saved successfully. File size: {os.path.getsize(output_file)} bytes")
    return map_json_clean


def main():
    """Main test function."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "python_saved_map.json")

    # Create and save map
    map_obj = create_test_map()
    map_json = save_map_to_json(map_obj, output_file)

    # Print summary
    print("\nMap summary:")
    print(f"  Frames: {len(map_json.get('frames', []))}")
    print(f"  Keyframes: {len(map_json.get('keyframes', []))}")
    print(f"  Points: {len(map_json.get('points', []))}")
    print(f"  FrameBase._id: {map_json.get('FrameBase._id', 'N/A')}")
    print(f"  MapPointBase._id: {map_json.get('MapPointBase._id', 'N/A')}")
    print(f"  max_frame_id: {map_json.get('max_frame_id', 'N/A')}")
    print(f"  max_keyframe_id: {map_json.get('max_keyframe_id', 'N/A')}")
    print(f"  max_point_id: {map_json.get('max_point_id', 'N/A')}")

    # Verify a sample keyframe has descriptors
    if len(map_json.get("keyframes", [])) > 0:
        sample_kf = map_json["keyframes"][0]
        has_des = "des" in sample_kf and sample_kf["des"] is not None
        print(f"\nSample keyframe (id={sample_kf.get('id', 'N/A')}):")
        print(f"  Has descriptors: {has_des}")
        if has_des:
            print(f"  Descriptor field type: {type(sample_kf['des']).__name__}")

    print(f"\nTest map saved to: {output_file}")
    print("You can now run the C++ test to load this map.")


if __name__ == "__main__":
    main()
