#!/usr/bin/env python3
"""
Test script to load a C++-saved map and verify it can be correctly loaded by Python.
This script loads a map saved from C++, verifies all data is correctly deserialized,
and checks that relocalization prerequisites are met.
"""

import sys
import os
import json
import numpy as np

# Add parent directory to path to import pyslam
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from pyslam.config import Config
from pyslam.config_parameters import Parameters

from pyslam.slam.map import Map
from pyslam.slam.frame import Frame, FrameBase
from pyslam.slam.keyframe import KeyFrame
from pyslam.slam.map_point import MapPoint, MapPointBase
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


def load_and_verify_map(map_file):
    """Load a C++-saved map and verify it's correctly deserialized."""
    print(f"Loading C++-saved map from {map_file}...")

    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Map file not found: {map_file}")

    # Initialize feature tracker (required for descriptor_distance)
    tracker_config = FeatureTrackerConfigs.ORB2.copy()
    tracker_config["num_features"] = 1000
    tracker_config["deterministic"] = True
    feature_tracker = feature_tracker_factory(**tracker_config)
    FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

    # Load map from JSON
    with open(map_file, "r") as f:
        map_json = json.load(f)

    print(f"JSON file size: {os.path.getsize(map_file)} bytes")

    # Create map and load from JSON
    map_obj = Map()
    map_obj.from_json(map_json)

    print("Map loaded successfully!")

    # Verify basic properties
    num_frames = len(map_obj.frames)
    num_keyframes = len(map_obj.keyframes)
    num_points = len(map_obj.points)

    print("\nLoaded map statistics:")
    print(f"  Frames: {num_frames}")
    print(f"  Keyframes: {num_keyframes}")
    print(f"  Points: {num_points}")
    print(f"  max_frame_id: {map_obj.max_frame_id}")
    print(f"  max_keyframe_id: {map_obj.max_keyframe_id}")
    print(f"  max_point_id: {map_obj.max_point_id}")

    # Verify we have data
    assert num_keyframes > 0, "Map should have at least one keyframe"
    assert num_points > 0, "Map should have at least one map point"

    # Verify keyframes_map is populated
    assert (
        len(map_obj.keyframes_map) == num_keyframes
    ), "keyframes_map should have same size as keyframes"
    print(f"  keyframes_map size: {len(map_obj.keyframes_map)}")

    # Verify FrameBase._id and MapPointBase._id are restored
    frame_base_id = FrameBase._id
    mappoint_base_id = MapPointBase._id

    print(f"  FrameBase._id: {frame_base_id}")
    print(f"  MapPointBase._id: {mappoint_base_id}")

    assert frame_base_id == map_obj.max_frame_id, "FrameBase._id should match max_frame_id"
    assert mappoint_base_id == map_obj.max_point_id, "MapPointBase._id should match max_point_id"

    # Verify keyframes have descriptors
    keyframes_with_des = 0
    keyframes_with_valid_des = 0
    keyframes_with_matching_des_kps = 0
    total_descriptors = 0

    for kf in map_obj.keyframes:
        if kf and kf.des is not None and kf.des.size > 0:
            keyframes_with_des += 1

            # Check descriptor validity
            has_valid_des = True
            if kf.des.dtype != np.uint8:
                print(
                    f"WARNING: Keyframe {kf.id} has descriptors with wrong type (expected uint8, got {kf.des.dtype})"
                )
                has_valid_des = False

            # Check descriptor dimensions (should be N x 32 for ORB)
            if len(kf.des.shape) != 2 or kf.des.shape[1] != 32:
                print(
                    f"WARNING: Keyframe {kf.id} has descriptors with wrong shape (expected (N, 32), got {kf.des.shape})"
                )
                has_valid_des = False

            # Check if descriptors match keypoint count
            num_kps = len(kf.kps) if kf.kps is not None else 0
            num_des = kf.des.shape[0] if len(kf.des.shape) > 0 else 0
            if num_kps > 0 and num_des != num_kps:
                print(
                    f"WARNING: Keyframe {kf.id} has mismatched descriptor/keypoint count (kps: {num_kps}, des: {num_des})"
                )
                has_valid_des = False
            elif num_kps == num_des and num_kps > 0:
                keyframes_with_matching_des_kps += 1

            # Check if descriptors are all zeros (invalid)
            if has_valid_des and num_des > 0:
                if np.all(kf.des == 0):
                    print(f"WARNING: Keyframe {kf.id} has all-zero descriptors (likely invalid)")
                    has_valid_des = False

            if has_valid_des:
                keyframes_with_valid_des += 1
                total_descriptors += num_des

    print(f"\n  Keyframes with descriptors: {keyframes_with_des} / {num_keyframes}")
    print(f"  Keyframes with valid descriptors: {keyframes_with_valid_des} / {num_keyframes}")
    print(
        f"  Keyframes with matching descriptor/keypoint counts: {keyframes_with_matching_des_kps} / {num_keyframes}"
    )
    print(f"  Total descriptors across all keyframes: {total_descriptors}")

    if keyframes_with_des == 0:
        print("WARNING: No keyframes have descriptors loaded!")
        print("This will cause relocalization to fail.")
    elif keyframes_with_valid_des == 0:
        print("WARNING: No keyframes have valid descriptors!")
        print("This will cause relocalization to fail.")

    # Verify keyframes have keypoints
    keyframes_with_kps = 0
    for kf in map_obj.keyframes:
        if kf and kf.kps is not None and len(kf.kps) > 0:
            keyframes_with_kps += 1

    print(f"  Keyframes with keypoints: {keyframes_with_kps} / {num_keyframes}")

    # Verify map points have descriptors
    points_with_des = 0
    points_with_valid_des = 0
    total_point_descriptors = 0

    for mp in map_obj.points:
        if mp and mp.des is not None and mp.des.size > 0:
            points_with_des += 1

            # Check descriptor validity
            has_valid_des = True
            if mp.des.dtype != np.uint8:
                print(
                    f"WARNING: MapPoint {mp.id} has descriptor with wrong type (expected uint8, got {mp.des.dtype})"
                )
                has_valid_des = False

            # Check descriptor dimensions (should be 1 x 32 or (32,) for ORB)
            # Handle both 1D (32,) and 2D (1, 32) shapes
            if mp.des.ndim == 1:
                if mp.des.shape[0] != 32:
                    print(
                        f"WARNING: MapPoint {mp.id} has descriptor with wrong shape (expected (32,) or (1, 32), got {mp.des.shape})"
                    )
                    has_valid_des = False
            elif mp.des.ndim == 2:
                if mp.des.shape[1] != 32:
                    print(
                        f"WARNING: MapPoint {mp.id} has descriptor with wrong shape (expected (32,) or (1, 32), got {mp.des.shape})"
                    )
                    has_valid_des = False
            else:
                print(
                    f"WARNING: MapPoint {mp.id} has descriptor with wrong number of dimensions (expected 1 or 2, got {mp.des.ndim})"
                )
                has_valid_des = False

            # Check if descriptor is all zeros (invalid)
            if has_valid_des and mp.des.size > 0:
                if np.all(mp.des == 0):
                    print(f"WARNING: MapPoint {mp.id} has all-zero descriptor (likely invalid)")
                    has_valid_des = False

            if has_valid_des:
                points_with_valid_des += 1
                # Count descriptors: 1 for 1D array, shape[0] for 2D array
                if mp.des.ndim == 1:
                    total_point_descriptors += 1
                else:
                    total_point_descriptors += mp.des.shape[0]

    print(f"  Map points with descriptors: {points_with_des} / {num_points}")
    print(f"  Map points with valid descriptors: {points_with_valid_des} / {num_points}")
    print(f"  Total descriptors across all map points: {total_point_descriptors}")

    # Verify images are loaded (critical for relocalization via compute_frame_matches)
    keyframes_with_img = 0
    for kf in map_obj.keyframes:
        if kf and kf.img is not None and len(kf.img.shape) >= 2:
            keyframes_with_img += 1

    print(f"  Keyframes with images: {keyframes_with_img} / {num_keyframes}")

    if keyframes_with_img == 0:
        print("WARNING: No keyframes have images loaded!")
        print("This will cause compute_frame_matches to fail (returns 0 compared pairs).")
        print("Relocalization requires images for feature matching.")

    # Verify keyframes are not marked as bad
    bad_keyframes = sum(1 for kf in map_obj.keyframes if kf and kf.is_bad())
    print(f"  Bad keyframes: {bad_keyframes} / {num_keyframes}")

    if bad_keyframes == num_keyframes:
        print("WARNING: All keyframes are marked as bad!")
        print(
            "This will cause compute_frame_matches to skip all frames (returns 0 compared pairs)."
        )

    # ========================================================================
    # Verify Map Point Associations with Keyframes
    # ========================================================================
    print("\nMap Point Association Check:")

    keyframes_with_points = 0
    total_keyframe_points = 0
    keyframes_with_valid_points = 0
    total_valid_keyframe_points = 0

    for kf in map_obj.keyframes:
        if not kf:
            continue

        kf_points = kf.points if kf.points is not None else []
        if len(kf_points) > 0:
            keyframes_with_points += 1
            total_keyframe_points += len(kf_points)

            # Count valid (non-null, non-bad) map points
            valid_points = sum(1 for p in kf_points if p is not None and not p.is_bad())

            if valid_points > 0:
                keyframes_with_valid_points += 1
                total_valid_keyframe_points += valid_points

    print(f"  Keyframes with points array: {keyframes_with_points} / {num_keyframes}")
    print(f"  Total map point associations: {total_keyframe_points}")
    print(f"  Keyframes with valid map points: {keyframes_with_valid_points} / {num_keyframes}")
    print(f"  Total valid map point associations: {total_valid_keyframe_points}")

    if keyframes_with_valid_points == 0:
        print("\nWARNING: No keyframes have valid map point associations!")
        print("This will cause prepare_input_data_for_pnpsolver to return 0 correspondences.")
        print("Relocalization will fail even if feature matching finds many matches.")
    elif keyframes_with_valid_points < num_keyframes:
        print(
            f"\nWARNING: Only {keyframes_with_valid_points} / {num_keyframes} keyframes have valid map point associations!"
        )

    # Test that we can actually retrieve map points from keyframes at keypoint indices
    print("\nTesting map point retrieval (simulating prepare_input_data_for_pnpsolver):")
    test_keyframes_with_retrievable_points = 0
    total_retrievable_points = 0

    for kf in map_obj.keyframes:
        if not kf or kf.kps is None or len(kf.kps) == 0:
            continue

        kf_points = kf.points if kf.points is not None else []
        if len(kf_points) == 0:
            continue

        # Test retrieving map points at keypoint indices (like prepare_input_data_for_pnpsolver does)
        retrievable_points = 0
        num_kps = len(kf.kps)
        num_points_arr = len(kf_points)
        max_idx = min(num_kps, num_points_arr)

        for i in range(max_idx):
            mp = (
                kf.get_point_match(i)
                if hasattr(kf, "get_point_match")
                else (kf_points[i] if i < len(kf_points) else None)
            )
            if mp is not None and not mp.is_bad():
                retrievable_points += 1

        if retrievable_points > 0:
            test_keyframes_with_retrievable_points += 1
            total_retrievable_points += retrievable_points

    print(
        f"  Keyframes with retrievable map points: {test_keyframes_with_retrievable_points} / {num_keyframes}"
    )
    print(f"  Total retrievable map points: {total_retrievable_points}")

    # ========================================================================
    # Verify Descriptor Shapes and Indexing (Critical for numba compatibility)
    # ========================================================================
    print("\nDescriptor Shape and Indexing Check (numba compatibility):")

    # Check Frame/KeyFrame descriptors
    keyframes_with_valid_des_shape = 0
    keyframes_with_indexable_des = 0
    keyframes_with_scalar_des_issue = 0

    for kf in map_obj.keyframes:
        if not kf or kf.des is None:
            continue

        # Check descriptor shape (should be 2D: N x 32 for ORB)
        if kf.des.ndim == 2 and kf.des.shape[1] == 32 and kf.des.shape[0] > 0:
            keyframes_with_valid_des_shape += 1

            # Test indexing: cur_des[kd_idx] should return 1D array, not scalar
            try:
                if kf.des.shape[0] > 0:
                    # Test indexing first descriptor
                    indexed_des = kf.des[0]
                    if isinstance(indexed_des, np.ndarray):
                        if indexed_des.ndim == 1 and indexed_des.shape[0] == 32:
                            keyframes_with_indexable_des += 1
                        elif indexed_des.ndim == 0:
                            print(
                                f"ERROR: Keyframe {kf.id} descriptor indexing returns 0D array (scalar)!"
                            )
                            print(
                                f"  kf.des.shape: {kf.des.shape}, kf.des[0].shape: {indexed_des.shape}"
                            )
                            keyframes_with_scalar_des_issue += 1
                        else:
                            print(
                                f"WARNING: Keyframe {kf.id} descriptor indexing returns unexpected shape"
                            )
                            print(
                                f"  kf.des.shape: {kf.des.shape}, kf.des[0].shape: {indexed_des.shape}"
                            )
                    else:
                        print(
                            f"ERROR: Keyframe {kf.id} descriptor indexing returns non-array: {type(indexed_des)}"
                        )
            except Exception as e:
                print(f"ERROR: Keyframe {kf.id} descriptor indexing failed: {e}")
        elif kf.des.ndim == 0:
            print(f"ERROR: Keyframe {kf.id} has 0D descriptor (scalar)!")
            print(f"  kf.des.shape: {kf.des.shape}, kf.des.dtype: {kf.des.dtype}")
            keyframes_with_scalar_des_issue += 1
        elif kf.des.ndim == 1:
            print(f"WARNING: Keyframe {kf.id} has 1D descriptor (expected 2D: N x 32)")
            print(f"  kf.des.shape: {kf.des.shape}")

    print(
        f"  Keyframes with valid 2D descriptor shape: {keyframes_with_valid_des_shape} / {num_keyframes}"
    )
    print(
        f"  Keyframes with indexable descriptors (1D result): {keyframes_with_indexable_des} / {keyframes_with_valid_des_shape}"
    )

    if keyframes_with_scalar_des_issue > 0:
        print(
            f"\nERROR: {keyframes_with_scalar_des_issue} keyframes have scalar descriptor issues!"
        )
        print("This will cause numba errors in descriptor_distance()")
        raise RuntimeError("Descriptor shape validation failed - scalar descriptors detected")

    # Check MapPoint descriptors
    points_with_valid_des_shape = 0
    points_with_scalar_des_issue = 0

    for mp in map_obj.points:
        if not mp or mp.des is None:
            continue

        # Check descriptor shape (should be 1D (32,) or 2D (1, 32) for ORB)
        if mp.des.ndim == 1:
            if mp.des.shape[0] == 32:
                points_with_valid_des_shape += 1
            else:
                print(f"WARNING: MapPoint {mp.id} has 1D descriptor with wrong size")
                print(f"  mp.des.shape: {mp.des.shape}, expected (32,)")
        elif mp.des.ndim == 2:
            if mp.des.shape[1] == 32 and mp.des.shape[0] > 0:
                points_with_valid_des_shape += 1
            else:
                print(f"WARNING: MapPoint {mp.id} has 2D descriptor with wrong shape")
                print(f"  mp.des.shape: {mp.des.shape}, expected (1, 32) or (N, 32)")
        elif mp.des.ndim == 0:
            print(f"ERROR: MapPoint {mp.id} has 0D descriptor (scalar)!")
            print(f"  mp.des.shape: {mp.des.shape}, mp.des.dtype: {mp.des.dtype}")
            points_with_scalar_des_issue += 1
        else:
            print(f"WARNING: MapPoint {mp.id} has descriptor with unexpected dimensions")
            print(f"  mp.des.ndim: {mp.des.ndim}, mp.des.shape: {mp.des.shape}")

    print(f"  Map points with valid descriptor shape: {points_with_valid_des_shape} / {num_points}")

    if points_with_scalar_des_issue > 0:
        print(f"\nERROR: {points_with_scalar_des_issue} map points have scalar descriptor issues!")
        print("This will cause numba errors in descriptor_distance()")
        raise RuntimeError("Descriptor shape validation failed - scalar descriptors detected")

    # Test actual descriptor distance computation (simulating min_des_distance)
    print("\nTesting descriptor distance computation (simulating min_des_distance):")

    test_pairs = 0
    successful_pairs = 0
    failed_pairs = 0

    for kf in map_obj.keyframes:
        if not kf or kf.des is None or kf.des.ndim != 2 or kf.des.shape[0] == 0:
            continue

        for mp in map_obj.points:
            if not mp or mp.des is None or mp.is_bad():
                continue

            # Test descriptor distance computation
            try:
                # Get a descriptor from keyframe (simulating cur_des[kd_idx])
                kf_des = kf.des[0]  # Should be 1D array of shape (32,)

                # Ensure mp.des is at least 1D
                mp_des = mp.des.reshape(-1) if mp.des.ndim == 0 else mp.des
                if mp_des.ndim == 1:
                    mp_des = mp_des[:32]  # Take first 32 elements if longer
                elif mp_des.ndim == 2:
                    mp_des = mp_des[0, :]  # Take first row if 2D
                else:
                    continue

                # Ensure both are 1D arrays of length 32
                if kf_des.ndim != 1 or kf_des.shape[0] != 32:
                    print(
                        f"WARNING: Keyframe {kf.id} descriptor[0] has wrong shape: {kf_des.shape}"
                    )
                    continue
                if mp_des.ndim != 1 or mp_des.shape[0] != 32:
                    print(
                        f"WARNING: MapPoint {mp.id} descriptor has wrong shape after reshape: {mp_des.shape}"
                    )
                    continue

                # Test descriptor distance (this will fail with numba if descriptors are scalars)
                test_pairs += 1
                distance = FeatureTrackerShared.descriptor_distance(kf_des, mp_des)
                if isinstance(distance, (int, float)) and not np.isnan(distance):
                    successful_pairs += 1
                else:
                    failed_pairs += 1
                    print(f"WARNING: Descriptor distance returned invalid value: {distance}")

                # Only test a few pairs per keyframe to avoid too many tests
                if test_pairs >= 10:
                    break
            except Exception as e:
                failed_pairs += 1
                print(
                    f"ERROR: Descriptor distance computation failed for KF {kf.id} and MP {mp.id}: {e}"
                )
                import traceback

                traceback.print_exc()

        if test_pairs >= 10:
            break

    print(f"  Tested {test_pairs} descriptor pairs")
    print(f"  Successful: {successful_pairs}, Failed: {failed_pairs}")

    if failed_pairs > 0:
        print(f"\nERROR: {failed_pairs} descriptor distance computations failed!")
        print("This indicates numba compatibility issues with descriptor shapes")
        raise RuntimeError("Descriptor distance computation failed - numba compatibility issue")

    if test_keyframes_with_retrievable_points == 0:
        print("\nERROR: Cannot retrieve map points from keyframes at keypoint indices!")
        print("This means prepare_input_data_for_pnpsolver will always return 0 correspondences.")
        print("Relocalization will fail even with perfect feature matching.")
        raise RuntimeError(
            "Map point associations not properly restored - relocalization will fail"
        )
    else:
        print("  ✓ Map points can be retrieved from keyframes at keypoint indices")
        print("  ✓ prepare_input_data_for_pnpsolver should work correctly")

    # Verify map points have observations linked to keyframes
    print("\nMap Point Observations Check:")
    points_with_observations = 0
    total_observations = 0

    for mp in map_obj.points:
        if not mp or mp.is_bad():
            continue

        num_obs = mp.num_observations()
        if num_obs > 0:
            points_with_observations += 1
            total_observations += num_obs

    print(f"  Map points with observations: {points_with_observations} / {num_points}")
    print(f"  Total observations: {total_observations}")

    if points_with_observations == 0:
        print("WARNING: No map points have observations linked to keyframes!")

    # ========================================================================
    # Advanced Checks: Map Point 3D Positions and Validity
    # ========================================================================
    print("\nMap Point 3D Position Validity Check:")

    points_with_valid_3d = 0
    points_with_invalid_3d = 0
    points_with_nan_3d = 0
    points_with_inf_3d = 0

    for mp in map_obj.points:
        if not mp or mp.is_bad():
            continue

        try:
            pt_3d = mp.pt()
            if pt_3d is None:
                points_with_invalid_3d += 1
                continue

            # Check for NaN
            if np.any(np.isnan(pt_3d)):
                points_with_nan_3d += 1
                print(f"ERROR: MapPoint {mp.id} has NaN in 3D position: {pt_3d}")
                continue

            # Check for Inf
            if np.any(np.isinf(pt_3d)):
                points_with_inf_3d += 1
                print(f"ERROR: MapPoint {mp.id} has Inf in 3D position: {pt_3d}")
                continue

            # Check for reasonable values (not too far from origin)
            dist_from_origin = np.linalg.norm(pt_3d)
            if dist_from_origin > 1e6:  # 1 million units is unreasonable
                points_with_invalid_3d += 1
                print(
                    f"WARNING: MapPoint {mp.id} has very large 3D position: {pt_3d} (distance: {dist_from_origin:.2f})"
                )
                continue

            # Check for zero or near-zero positions (might indicate uninitialized points)
            if dist_from_origin < 1e-6:
                points_with_invalid_3d += 1
                print(f"WARNING: MapPoint {mp.id} has zero or near-zero 3D position: {pt_3d}")
                continue

            points_with_valid_3d += 1
        except Exception as e:
            points_with_invalid_3d += 1
            print(f"ERROR: MapPoint {mp.id} failed to get 3D position: {e}")

    print(f"  Map points with valid 3D positions: {points_with_valid_3d} / {num_points}")
    if points_with_nan_3d > 0:
        print(f"  ERROR: {points_with_nan_3d} map points have NaN in 3D positions!")
        raise RuntimeError("Map points with NaN 3D positions detected")
    if points_with_inf_3d > 0:
        print(f"  ERROR: {points_with_inf_3d} map points have Inf in 3D positions!")
        raise RuntimeError("Map points with Inf 3D positions detected")
    if points_with_valid_3d == 0:
        print("  ERROR: No map points have valid 3D positions!")
        raise RuntimeError("No valid 3D map point positions - tracking will fail")

    # ========================================================================
    # Advanced Checks: Keyframe Pose Validity
    # ========================================================================
    print("\nKeyframe Pose Validity Check:")

    keyframes_with_valid_pose = 0
    keyframes_with_invalid_pose = 0

    for kf in map_obj.keyframes:
        if not kf or kf.is_bad():
            continue

        try:
            pose = kf.pose()
            if pose is None:
                keyframes_with_invalid_pose += 1
                print(f"ERROR: Keyframe {kf.id} has None pose")
                continue

            # Check pose matrix shape (should be 4x4)
            if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
                keyframes_with_invalid_pose += 1
                print(
                    f"ERROR: Keyframe {kf.id} has invalid pose shape: {pose.shape if isinstance(pose, np.ndarray) else type(pose)}"
                )
                continue

            # Check for NaN or Inf
            if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
                keyframes_with_invalid_pose += 1
                print(f"ERROR: Keyframe {kf.id} has NaN/Inf in pose matrix")
                continue

            # Check rotation matrix validity (should be orthogonal)
            R = pose[:3, :3]
            should_be_identity = R @ R.T
            identity = np.eye(3)
            orthogonality_error = np.max(np.abs(should_be_identity - identity))
            if orthogonality_error > 1e-3:
                keyframes_with_invalid_pose += 1
                if keyframes_with_invalid_pose <= 3:  # Only print first few
                    print(
                        f"WARNING: Keyframe {kf.id} rotation matrix orthogonality error: {orthogonality_error:.6f}"
                    )
                continue

            # Check determinant (should be 1 for proper rotation)
            det = np.linalg.det(R)
            if abs(det - 1.0) > 1e-3:
                keyframes_with_invalid_pose += 1
                if keyframes_with_invalid_pose <= 3:  # Only print first few
                    print(
                        f"WARNING: Keyframe {kf.id} rotation matrix determinant: {det:.6f} (expected ~1.0)"
                    )
                continue

            keyframes_with_valid_pose += 1
        except Exception as e:
            keyframes_with_invalid_pose += 1
            print(f"ERROR: Keyframe {kf.id} failed pose validation: {e}")

    print(f"  Keyframes with valid poses: {keyframes_with_valid_pose} / {num_keyframes}")
    if keyframes_with_valid_pose == 0:
        print("  ERROR: No keyframes have valid poses!")
        raise RuntimeError("No valid keyframe poses - tracking will fail")

    # ========================================================================
    # Advanced Checks: Simulate prepare_input_data_for_pnpsolver
    # ========================================================================
    print("\nSimulating prepare_input_data_for_pnpsolver:")

    from pyslam.slam.frame import prepare_input_data_for_pnpsolver

    test_keyframes_with_valid_pnp_data = 0
    total_valid_correspondences = 0

    # Create a dummy frame for testing
    if keyframes_with_valid_pose > 0 and points_with_valid_3d > 0:
        # Find a keyframe with valid data
        test_kf = None
        for kf in map_obj.keyframes:
            if kf and not kf.is_bad() and kf.des is not None and kf.des.shape[0] > 0:
                test_kf = kf
                break

        if test_kf:
            # Create a minimal dummy frame with matching descriptors
            try:
                # Get camera from keyframe
                camera = test_kf.camera

                # Create dummy frame with same camera and some keypoints
                dummy_img = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
                dummy_frame = Frame(camera, img=dummy_img, id=99999, timestamp=0.0, img_id=99999)

                # Copy some keypoints and descriptors from keyframe to dummy frame
                num_test_kps = min(10, test_kf.des.shape[0])
                dummy_frame.kps = test_kf.kps[:num_test_kps].copy()
                dummy_frame.des = test_kf.des[:num_test_kps].copy()
                dummy_frame.octaves = (
                    test_kf.octaves[:num_test_kps].copy()
                    if test_kf.octaves is not None
                    else np.zeros(num_test_kps, dtype=np.int32)
                )

                # Create matching indices (simulating feature matching)
                idxs_frame = np.arange(num_test_kps)
                idxs_kf = np.arange(num_test_kps)

                # Test prepare_input_data_for_pnpsolver
                try:
                    points_3d, points_2d, sigmas2, idxs1, idxs2 = prepare_input_data_for_pnpsolver(
                        dummy_frame, test_kf, idxs_frame, idxs_kf
                    )

                    num_correspondences = len(points_2d)
                    if num_correspondences > 0:
                        test_keyframes_with_valid_pnp_data += 1
                        total_valid_correspondences += num_correspondences

                        # Validate the output
                        if len(points_3d) != num_correspondences:
                            print(
                                f"WARNING: prepare_input_data_for_pnpsolver returned mismatched sizes"
                            )
                        if len(points_2d) != num_correspondences:
                            print(
                                f"WARNING: prepare_input_data_for_pnpsolver returned mismatched sizes"
                            )
                        if np.any(np.isnan(points_3d)) or np.any(np.isinf(points_3d)):
                            print(
                                f"ERROR: prepare_input_data_for_pnpsolver returned NaN/Inf in 3D points"
                            )
                            raise RuntimeError(
                                "prepare_input_data_for_pnpsolver returned invalid 3D points"
                            )
                        if np.any(np.isnan(points_2d)) or np.any(np.isinf(points_2d)):
                            print(
                                f"ERROR: prepare_input_data_for_pnpsolver returned NaN/Inf in 2D points"
                            )
                            raise RuntimeError(
                                "prepare_input_data_for_pnpsolver returned invalid 2D points"
                            )

                        print(f"  ✓ prepare_input_data_for_pnpsolver works correctly")
                        print(f"  ✓ Generated {num_correspondences} valid correspondences")
                    else:
                        print(
                            f"  WARNING: prepare_input_data_for_pnpsolver returned 0 correspondences"
                        )
                        print(f"  This will cause relocalization to fail")
                except Exception as e:
                    print(f"  ERROR: prepare_input_data_for_pnpsolver failed: {e}")
                    import traceback

                    traceback.print_exc()
                    raise RuntimeError(f"prepare_input_data_for_pnpsolver failed: {e}")
            except Exception as e:
                print(
                    f"  WARNING: Could not create test frame for prepare_input_data_for_pnpsolver: {e}"
                )

    # ========================================================================
    # Advanced Checks: Descriptor Matching Quality
    # ========================================================================
    print("\nDescriptor Matching Quality Check:")

    # Test that descriptors from map points and keyframes can actually match
    test_matches = 0
    successful_matches = 0
    failed_matches = 0

    for kf in map_obj.keyframes[:2]:  # Test first 2 keyframes only
        if not kf or kf.des is None or kf.des.shape[0] == 0:
            continue

        # Get map points associated with this keyframe
        kf_points = (
            kf.get_points()
            if hasattr(kf, "get_points")
            else (kf.points if kf.points is not None else [])
        )

        for mp in kf_points[:5]:  # Test first 5 map points
            if not mp or mp.des is None or mp.is_bad():
                continue

            try:
                # Get descriptors
                kf_des = kf.des[0]  # First descriptor from keyframe
                mp_des = mp.des.reshape(-1) if mp.des.ndim == 0 else mp.des
                if mp_des.ndim == 2:
                    mp_des = mp_des[0, :]
                elif mp_des.ndim == 1:
                    mp_des = mp_des[:32]
                else:
                    continue

                if kf_des.shape[0] != 32 or mp_des.shape[0] != 32:
                    continue

                # Compute descriptor distance
                test_matches += 1
                distance = FeatureTrackerShared.descriptor_distance(kf_des, mp_des)

                # Check if distance is reasonable (0-256 for Hamming distance)
                if 0 <= distance <= 256:
                    successful_matches += 1
                else:
                    failed_matches += 1
                    print(f"WARNING: Unusual descriptor distance: {distance} (expected 0-256)")
            except Exception as e:
                failed_matches += 1
                print(f"WARNING: Descriptor matching failed: {e}")

    print(f"  Tested {test_matches} descriptor matches")
    print(f"  Successful matches: {successful_matches}, Failed: {failed_matches}")

    if test_matches > 0 and successful_matches == 0:
        print("  ERROR: All descriptor matches failed!")
        raise RuntimeError("Descriptor matching quality check failed")

    # ========================================================================
    # Advanced Checks: Map Point Projection (Critical for tracking)
    # ========================================================================
    print("\nMap Point Projection Check (using Frame.project_map_points):")

    projection_test_keyframes = 0
    projection_test_successful = 0
    projection_test_failed = 0
    projection_test_errors = []

    for kf in map_obj.keyframes[:2]:  # Test first 2 keyframes
        if not kf or kf.is_bad() or kf.camera is None:
            continue

        # Get map points associated with this keyframe
        kf_points = (
            kf.get_points()
            if hasattr(kf, "get_points")
            else (kf.points if kf.points is not None else [])
        )
        valid_mps = [
            mp for mp in kf_points[:10] if mp is not None and not mp.is_bad()
        ]  # Test first 10

        if len(valid_mps) == 0:
            continue

        try:
            # Test projection using the actual Frame.project_map_points method
            projection_test_keyframes += 1

            # Use the actual projection method (this is what search_frame_by_projection uses)
            try:
                projs, depths = kf.project_map_points(valid_mps, do_stereo_project=False)

                # Validate output
                if projs is None or depths is None:
                    projection_test_failed += 1
                    projection_test_errors.append(
                        f"Keyframe {kf.id}: project_map_points returned None"
                    )
                    continue

                # Check shapes
                if projs.shape[0] != len(valid_mps) or projs.shape[1] != 2:
                    projection_test_failed += 1
                    projection_test_errors.append(
                        f"Keyframe {kf.id}: project_map_points returned wrong shape: {projs.shape}, expected ({len(valid_mps)}, 2)"
                    )
                    continue

                if depths.shape[0] != len(valid_mps):
                    projection_test_failed += 1
                    projection_test_errors.append(
                        f"Keyframe {kf.id}: depths shape mismatch: {depths.shape}, expected ({len(valid_mps)},)"
                    )
                    continue

                # Check for NaN/Inf in projections
                if np.any(np.isnan(projs)) or np.any(np.isinf(projs)):
                    projection_test_failed += 1
                    projection_test_errors.append(f"Keyframe {kf.id}: NaN/Inf in projections")
                    continue

                if np.any(np.isnan(depths)) or np.any(np.isinf(depths)):
                    projection_test_failed += 1
                    projection_test_errors.append(f"Keyframe {kf.id}: NaN/Inf in depths")
                    continue

                # Check visibility using Frame.are_visible (this is what search_frame_by_projection uses)
                # are_visible expects a list of MapPoint objects, not projections
                try:
                    is_visible_flags, uvs_check, zs_check, dists_check = kf.are_visible(
                        valid_mps, do_stereo_project=False
                    )
                    num_visible = (
                        np.sum(is_visible_flags) if isinstance(is_visible_flags, np.ndarray) else 0
                    )

                    # Note: Not all map points need to be visible from the keyframe's current pose
                    # They might have been observed from a different pose. We just need to verify
                    # that the projection method works correctly.
                    if num_visible > 0:
                        projection_test_successful += 1
                        print(
                            f"  ✓ Keyframe {kf.id}: {num_visible}/{len(valid_mps)} map points are visible (projection works)"
                        )
                    else:
                        # This is not necessarily an error - points might not be visible from current pose
                        # But we should at least verify the projection method works
                        print(
                            f"  ℹ Keyframe {kf.id}: 0/{len(valid_mps)} map points visible from current pose (may be normal if pose changed)"
                        )
                        # Still count as successful if projection method works (no errors)
                        projection_test_successful += 1

                except Exception as e:
                    projection_test_failed += 1
                    projection_test_errors.append(f"Keyframe {kf.id}: are_visible failed: {e}")
                    import traceback

                    traceback.print_exc()

            except Exception as e:
                projection_test_failed += 1
                projection_test_errors.append(f"Keyframe {kf.id}: project_map_points failed: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            projection_test_failed += 1
            projection_test_errors.append(f"Keyframe {kf.id}: projection setup failed: {e}")

    print(f"  Tested {projection_test_keyframes} keyframes for projection")
    print(f"  Successful: {projection_test_successful}, Failed: {projection_test_failed}")

    if projection_test_errors:
        print("  Errors encountered:")
        for error in projection_test_errors[:5]:  # Show first 5 errors
            print(f"    - {error}")

    # Only fail if projection method itself fails (not if points aren't visible)
    if projection_test_keyframes > 0 and projection_test_successful == 0:
        print("  ERROR: All projection tests failed!")
        print("  This indicates Frame.project_map_points is not working correctly")
        raise RuntimeError(
            "Map point projection check failed - Frame.project_map_points not working"
        )

    # ========================================================================
    # Advanced Checks: Numerical Precision and Consistency
    # ========================================================================
    print("\nNumerical Precision and Consistency Check:")

    # Check for pose consistency across keyframes
    pose_consistency_issues = 0
    max_pose_difference = 0.0

    # Sample a few keyframes and check if their relative poses make sense
    sample_keyframes = [kf for kf in map_obj.keyframes if kf and not kf.is_bad()][
        : min(5, num_keyframes)
    ]

    if len(sample_keyframes) >= 2:
        for i in range(len(sample_keyframes) - 1):
            kf1 = sample_keyframes[i]
            kf2 = sample_keyframes[i + 1]

            try:
                pose1 = kf1.pose()
                pose2 = kf2.pose()

                # Compute relative pose
                pose1_inv = np.linalg.inv(pose1)
                relative_pose = pose2 @ pose1_inv

                # Check if relative pose is reasonable (not too large translation)
                translation = relative_pose[:3, 3]
                translation_norm = np.linalg.norm(translation)

                if translation_norm > 1000:  # Unreasonably large translation
                    pose_consistency_issues += 1
                    if pose_consistency_issues <= 3:
                        print(
                            f"WARNING: Large relative translation between KF {kf1.id} and {kf2.id}: {translation_norm:.2f}"
                        )

                # Check relative rotation
                R_rel = relative_pose[:3, :3]
                R_rel_ortho_error = np.max(np.abs(R_rel @ R_rel.T - np.eye(3)))
                if R_rel_ortho_error > 1e-3:
                    pose_consistency_issues += 1
                    if pose_consistency_issues <= 3:
                        print(
                            f"WARNING: Non-orthogonal relative rotation between KF {kf1.id} and {kf2.id}: {R_rel_ortho_error:.6f}"
                        )

                max_pose_difference = max(max_pose_difference, translation_norm)
            except Exception as e:
                pose_consistency_issues += 1
                if pose_consistency_issues <= 3:
                    print(
                        f"WARNING: Failed to check pose consistency between KF {kf1.id} and {kf2.id}: {e}"
                    )

    print(f"  Checked {len(sample_keyframes)} keyframes for pose consistency")
    if pose_consistency_issues > 0:
        print(f"  WARNING: {pose_consistency_issues} pose consistency issues found")
        print("  This may indicate numerical precision problems that could cause tracking failures")
    else:
        print("  ✓ Pose consistency OK")

    # Check map point position distribution
    print("\nMap Point Position Distribution Check:")
    point_positions = []
    for mp in map_obj.points:
        if mp and not mp.is_bad():
            try:
                pt = mp.pt()
                if pt is not None and not np.any(np.isnan(pt)) and not np.any(np.isinf(pt)):
                    point_positions.append(pt)
            except:
                pass

    if len(point_positions) > 0:
        point_positions = np.array(point_positions)
        distances = np.linalg.norm(point_positions, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        print(f"  Map point distances from origin:")
        print(f"    Mean: {mean_dist:.2f}, Std: {std_dist:.2f}")
        print(f"    Min: {min_dist:.2f}, Max: {max_dist:.2f}")

        # Check for suspicious distributions (all points too close or too far)
        if mean_dist < 0.1:
            print(
                "  WARNING: Map points are all very close to origin - may indicate initialization issues"
            )
        if max_dist > 1e5:
            print(
                "  WARNING: Some map points are extremely far from origin - may indicate numerical issues"
            )

        # Check spread (points should have reasonable spread)
        if std_dist < 0.01 and len(point_positions) > 1:
            print(
                "  WARNING: Map points have very little spread - may indicate all points are at same location"
            )

    # Summary of relocalization prerequisites
    print("\nRelocalization prerequisites check:")
    has_valid_descriptors = keyframes_with_valid_des > 0
    has_matching_des_kps = keyframes_with_matching_des_kps > 0
    has_keypoints = keyframes_with_kps > 0
    has_images = keyframes_with_img > 0
    has_good_keyframes = bad_keyframes < num_keyframes
    has_map_point_associations = test_keyframes_with_retrievable_points > 0

    print(
        f"  ✓ Valid descriptors: {'OK' if has_valid_descriptors else 'MISSING'} ({keyframes_with_valid_des}/{num_keyframes} keyframes)"
    )
    print(
        f"  ✓ Matching descriptor/keypoint counts: {'OK' if has_matching_des_kps else 'MISSING'} ({keyframes_with_matching_des_kps}/{num_keyframes} keyframes)"
    )
    print(
        f"  ✓ Keypoints: {'OK' if has_keypoints else 'MISSING'} ({keyframes_with_kps}/{num_keyframes} keyframes)"
    )
    print(
        f"  ✓ Images: {'OK' if has_images else 'MISSING'} ({keyframes_with_img}/{num_keyframes} keyframes)"
    )
    print(
        f"  ✓ Good keyframes: {'OK' if has_good_keyframes else 'MISSING'} ({num_keyframes - bad_keyframes}/{num_keyframes} keyframes)"
    )
    print(
        f"  ✓ Map point associations: {'OK' if has_map_point_associations else 'MISSING'} ({test_keyframes_with_retrievable_points}/{num_keyframes} keyframes)"
    )

    if not (
        has_valid_descriptors
        and has_matching_des_kps
        and has_keypoints
        and has_images
        and has_good_keyframes
        and has_map_point_associations
    ):
        print("\nWARNING: Some relocalization prerequisites are missing!")
        print("Relocalization may fail with 'compute_frame_matches: #compared pairs: 0'")
        if not has_valid_descriptors:
            print("  - Missing or invalid descriptors in keyframes")
        if not has_matching_des_kps:
            print("  - Descriptor/keypoint count mismatch in keyframes")
        if not has_map_point_associations:
            print("  - Missing map point associations in keyframes")
    else:
        print("  ✓ All relocalization prerequisites met!")
        print(f"  ✓ Total descriptors available: {total_descriptors}")

    print("\nC++-to-Python map loading tests passed!")
    return map_obj


def main():
    """Main test function."""
    # Get path to C++-saved map
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, "test_data")
    map_file = os.path.join(test_data_dir, "cpp_saved_map.json")

    # Try to find the map file
    if not os.path.exists(map_file):
        # Try alternative paths
        possible_paths = [
            map_file,
            os.path.join(script_dir, "../tests_cpp/../tests_py/test_data/cpp_saved_map.json"),
            "pyslam/slam/cpp/tests_py/test_data/cpp_saved_map.json",
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                map_file = path
                found = True
                break

        if not found:
            print("ERROR: C++-saved map file not found.")
            print("Please run the C++ test first to generate the test map.")
            print("Expected location: pyslam/slam/cpp/tests_py/test_data/cpp_saved_map.json")
            return 1

    try:
        map_obj = load_and_verify_map(map_file)
        print("\nAll C++-to-Python serialization tests passed!")
        return 0
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
