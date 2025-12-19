#!/usr/bin/env python3
"""
Test script to verify Python → C++ → Python round-trip serialization.
This script loads a map that was saved from Python, loaded by C++, and saved again by C++,
then verifies that all data is correctly preserved through the round-trip.
"""

import sys
import os
import json
import numpy as np

from pyslam.config import Config
from pyslam.config_parameters import Parameters

from pyslam.slam.map import Map
from pyslam.slam.frame import Frame
from pyslam.slam.map_point import MapPoint
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


def load_map(map_file):
    """Load a map from JSON file."""
    with open(map_file, "r") as f:
        json_data = json.load(f)

    map_obj = Map()
    map_obj.from_json(json_data)
    return map_obj


def load_and_verify_map(round_trip_map_file, original_map_file=None):
    """Load and verify a C++-saved map (that was originally from Python).

    Args:
        round_trip_map_file: Path to the round-trip map (Python → C++ → Python)
        original_map_file: Path to the original Python map (ground truth) for comparison
    """
    print(f"Loading Python → C++ → Python round-trip map from {round_trip_map_file}...")

    # Initialize feature tracker (required for FeatureTrackerShared.descriptor_distance)
    if FeatureTrackerShared.feature_tracker is None:
        tracker_config = FeatureTrackerConfigs.ORB2.copy()
        tracker_config["num_features"] = 1000
        tracker_config["deterministic"] = True
        feature_tracker = feature_tracker_factory(**tracker_config)
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)
        print("Initialized feature tracker for descriptor distance test")

    # Load round-trip map
    print(f"JSON file size: {os.path.getsize(round_trip_map_file)} bytes")
    print("\tLoading frames...")
    print("\tLoading keyframes...")
    print("\tLoading points...")
    print("\tReplacing ids with actual objects in frames...")
    print("\tReplacing ids with actual objects in keyframes...")
    print("\tReplacing ids with actual objects in points...")

    map_obj = load_map(round_trip_map_file)

    # Load original map for comparison if provided
    original_map_obj = None
    if original_map_file and os.path.exists(original_map_file):
        print(f"\nLoading original Python map (ground truth) from {original_map_file}...")
        original_map_obj = load_map(original_map_file)
        print("Original map loaded for comparison.")

    print("Map loaded successfully!\n")

    # Basic statistics
    num_frames = len(map_obj.frames)
    num_keyframes = len(map_obj.keyframes)
    num_points = len(map_obj.points)

    print("Loaded map statistics:")
    print(f"  Frames: {num_frames}")
    print(f"  Keyframes: {num_keyframes}")
    print(f"  Points: {num_points}")
    print(f"  max_frame_id: {map_obj.max_frame_id}")
    print(f"  max_keyframe_id: {map_obj.max_keyframe_id}")
    print(f"  max_point_id: {map_obj.max_point_id}")
    print(f"  keyframes_map size: {len(map_obj.keyframes_map)}")
    print(f"  FrameBase._id: {Frame._id}")
    print(f"  MapPointBase._id: {MapPoint._id}")

    # Verify descriptors
    print("\nDescriptor Check:")
    keyframes_with_valid_des = 0
    keyframes_with_matching_des_kps = 0
    total_descriptors = 0

    for kf in map_obj.keyframes:
        if not kf or kf.is_bad():
            continue
        if kf.des is not None and kf.des.size > 0:
            keyframes_with_valid_des += 1
            total_descriptors += kf.des.shape[0]
            if kf.des.shape[0] == len(kf.kps):
                keyframes_with_matching_des_kps += 1

    print(f"  Keyframes with descriptors: {keyframes_with_valid_des} / {num_keyframes}")
    print(
        f"  Keyframes with matching descriptor/keypoint counts: {keyframes_with_matching_des_kps} / {num_keyframes}"
    )
    print(f"  Total descriptors across all keyframes: {total_descriptors}")

    # Verify map point associations
    print("\nMap Point Association Check:")
    test_keyframes_with_retrievable_points = 0
    total_retrievable_points = 0

    for kf in map_obj.keyframes:
        if not kf or kf.is_bad():
            continue
        if kf.points is not None:
            retrievable_count = 0
            for i, mp in enumerate(kf.points):
                if mp is not None and not mp.is_bad():
                    retrievable_count += 1
            if retrievable_count > 0:
                test_keyframes_with_retrievable_points += 1
                total_retrievable_points += retrievable_count

    print(
        f"  Keyframes with retrievable map points: {test_keyframes_with_retrievable_points} / {num_keyframes}"
    )
    print(f"  Total retrievable map points: {total_retrievable_points}")

    # Verify 3D positions
    print("\nMap Point 3D Position Validity Check:")
    points_with_valid_3d = 0
    for mp in map_obj.points:
        if not mp or mp.is_bad():
            continue
        try:
            pt_3d = mp.pt()
            if pt_3d is not None and not np.any(np.isnan(pt_3d)) and not np.any(np.isinf(pt_3d)):
                points_with_valid_3d += 1
        except:
            pass

    print(f"  Map points with valid 3D positions: {points_with_valid_3d} / {num_points}")

    # Verify poses
    print("\nKeyframe Pose Validity Check:")
    keyframes_with_valid_pose = 0
    for kf in map_obj.keyframes:
        if not kf or kf.is_bad():
            continue
        try:
            pose = kf.pose()
            if pose is not None and not np.any(np.isnan(pose)) and not np.any(np.isinf(pose)):
                # Check rotation matrix orthogonality
                R = pose[:3, :3]
                should_be_identity = R @ R.T
                identity = np.eye(3)
                if np.allclose(should_be_identity, identity, atol=1e-3):
                    keyframes_with_valid_pose += 1
        except:
            pass

    print(f"  Keyframes with valid poses: {keyframes_with_valid_pose} / {num_keyframes}")

    # ========================================================================
    # Compare with Original Map (Ground Truth)
    # ========================================================================
    if original_map_obj is not None:
        print("\n" + "=" * 70)
        print("Comparing Round-Trip Map with Original Map (Ground Truth)")
        print("=" * 70)

        # Basic counts comparison
        print("\nBasic Counts Comparison:")
        print(
            f"  Original: {len(original_map_obj.frames)} frames, {len(original_map_obj.keyframes)} keyframes, {len(original_map_obj.points)} points"
        )
        print(f"  Round-trip: {num_frames} frames, {num_keyframes} keyframes, {num_points} points")

        counts_match = (
            len(original_map_obj.frames) == num_frames
            and len(original_map_obj.keyframes) == num_keyframes
            and len(original_map_obj.points) == num_points
        )

        if counts_match:
            print("  ✓ Counts match")
        else:
            print("  ✗ Counts mismatch!")

        # Compare map point 3D positions
        print("\nMap Point 3D Position Comparison:")
        position_differences = []
        max_position_diff = 0.0
        points_with_position_diff = 0

        # Create ID to map point mapping for original map
        original_points_dict = {
            mp.id: mp for mp in original_map_obj.points if mp and not mp.is_bad()
        }

        for mp in map_obj.points:
            if not mp or mp.is_bad():
                continue
            if mp.id in original_points_dict:
                orig_mp = original_points_dict[mp.id]
                try:
                    orig_pt = orig_mp.pt()
                    round_trip_pt = mp.pt()
                    if orig_pt is not None and round_trip_pt is not None:
                        diff = np.linalg.norm(orig_pt - round_trip_pt)
                        position_differences.append(diff)
                        max_position_diff = max(max_position_diff, diff)
                        if diff > 1e-6:  # Significant difference
                            points_with_position_diff += 1
                except:
                    pass

        if position_differences:
            mean_diff = np.mean(position_differences)
            std_diff = np.std(position_differences)
            print(
                f"  Position differences: mean={mean_diff:.2e}, std={std_diff:.2e}, max={max_position_diff:.2e}"
            )
            print(
                f"  Points with differences > 1e-6: {points_with_position_diff} / {len(position_differences)}"
            )

            if max_position_diff < 1e-3:
                print("  ✓ Position differences are negligible (< 1mm)")
            elif max_position_diff < 1e-1:
                print("  ⚠ Position differences are small but noticeable (< 10cm)")
            else:
                print("  ✗ Position differences are significant (> 10cm)!")
        else:
            print("  ⚠ Could not compare positions (no matching points found)")

        # Compare keyframe poses
        print("\nKeyframe Pose Comparison:")
        pose_differences = []
        max_pose_translation_diff = 0.0
        max_pose_rotation_diff = 0.0
        keyframes_with_pose_diff = 0

        # Create ID to keyframe mapping for original map
        original_keyframes_dict = {
            kf.id: kf for kf in original_map_obj.keyframes if kf and not kf.is_bad()
        }

        # Also create index-based mapping as fallback
        original_keyframes_list = [
            kf for kf in original_map_obj.keyframes if kf and not kf.is_bad()
        ]
        round_trip_keyframes_list = [kf for kf in map_obj.keyframes if kf and not kf.is_bad()]

        # First try matching by ID
        matched_by_id = 0
        successful_pose_comparisons = 0
        failed_pose_comparisons = 0
        pose_comparison_errors = []

        for kf in map_obj.keyframes:
            if not kf or kf.is_bad():
                continue
            if kf.id in original_keyframes_dict:
                matched_by_id += 1
                orig_kf = original_keyframes_dict[kf.id]
                try:
                    orig_pose = orig_kf.pose()
                    round_trip_pose = kf.pose()
                    if orig_pose is None:
                        failed_pose_comparisons += 1
                        pose_comparison_errors.append(f"Original KF {kf.id}: pose() returned None")
                        continue
                    if round_trip_pose is None:
                        failed_pose_comparisons += 1
                        pose_comparison_errors.append(
                            f"Round-trip KF {kf.id}: pose() returned None"
                        )
                        continue

                    # Compare translation
                    orig_translation = orig_pose[:3, 3]
                    round_trip_translation = round_trip_pose[:3, 3]
                    translation_diff = np.linalg.norm(orig_translation - round_trip_translation)
                    max_pose_translation_diff = max(max_pose_translation_diff, translation_diff)

                    # Compare rotation (using relative rotation error)
                    orig_R = orig_pose[:3, :3]
                    round_trip_R = round_trip_pose[:3, :3]
                    relative_R = round_trip_R @ orig_R.T
                    # Rotation error as angle (in radians)
                    trace = np.trace(relative_R)
                    angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    max_pose_rotation_diff = max(max_pose_rotation_diff, angle_error)

                    successful_pose_comparisons += 1
                    if translation_diff > 1e-6 or angle_error > 1e-6:
                        keyframes_with_pose_diff += 1
                        pose_differences.append((translation_diff, angle_error))
                except Exception as e:
                    failed_pose_comparisons += 1
                    pose_comparison_errors.append(f"KF {kf.id}: {str(e)}")

        # If no matches by ID, try matching by index
        if matched_by_id == 0 and len(original_keyframes_list) == len(round_trip_keyframes_list):
            print("  Note: Matching keyframes by index (IDs may have changed)")
            for i, kf in enumerate(round_trip_keyframes_list):
                if i < len(original_keyframes_list):
                    orig_kf = original_keyframes_list[i]
                    try:
                        orig_pose = orig_kf.pose()
                        round_trip_pose = kf.pose()
                        if orig_pose is not None and round_trip_pose is not None:
                            # Compare translation
                            orig_translation = orig_pose[:3, 3]
                            round_trip_translation = round_trip_pose[:3, 3]
                            translation_diff = np.linalg.norm(
                                orig_translation - round_trip_translation
                            )
                            max_pose_translation_diff = max(
                                max_pose_translation_diff, translation_diff
                            )

                            # Compare rotation (using relative rotation error)
                            orig_R = orig_pose[:3, :3]
                            round_trip_R = round_trip_pose[:3, :3]
                            relative_R = round_trip_R @ orig_R.T
                            # Rotation error as angle (in radians)
                            trace = np.trace(relative_R)
                            angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                            max_pose_rotation_diff = max(max_pose_rotation_diff, angle_error)

                            if translation_diff > 1e-6 or angle_error > 1e-6:
                                keyframes_with_pose_diff += 1
                                pose_differences.append((translation_diff, angle_error))
                    except:
                        pass

        if successful_pose_comparisons > 0:
            # We successfully compared poses - report results
            if pose_differences:
                # Some poses had measurable differences
                mean_trans_diff = np.mean([d[0] for d in pose_differences])
                mean_rot_diff = np.mean([d[1] for d in pose_differences])
                print(
                    f"  Translation differences: mean={mean_trans_diff:.2e}, max={max_pose_translation_diff:.2e}"
                )
                print(
                    f"  Rotation differences: mean={mean_rot_diff:.2e} rad, max={max_pose_rotation_diff:.2e} rad"
                )
                print(
                    f"  Keyframes with differences > 1e-6: {keyframes_with_pose_diff} / {len(pose_differences)}"
                )

                if max_pose_translation_diff < 1e-3 and max_pose_rotation_diff < 1e-3:
                    print("  ✓ Pose differences are negligible")
                elif max_pose_translation_diff < 1e-1 and max_pose_rotation_diff < 1e-2:
                    print("  ⚠ Pose differences are small but noticeable")
                else:
                    print("  ✗ Pose differences are significant!")
            else:
                # All poses compared successfully and all differences are negligible (< 1e-6)
                print(f"  Compared {successful_pose_comparisons} keyframe poses")
                print(f"  Translation differences: max={max_pose_translation_diff:.2e}")
                print(f"  Rotation differences: max={max_pose_rotation_diff:.2e} rad")
                if max_pose_translation_diff < 1e-6 and max_pose_rotation_diff < 1e-6:
                    print("  ✓ All pose differences are negligible (< 1e-6)")
                else:
                    print("  ✓ Pose differences are very small")
        else:
            # Check if we have keyframes but couldn't match them
            if len(original_keyframes_list) > 0 and len(round_trip_keyframes_list) > 0:
                # Report ID information for debugging
                orig_ids = [kf.id for kf in original_keyframes_list]
                round_trip_ids = [kf.id for kf in round_trip_keyframes_list]

                print(
                    f"  Matched keyframes by ID: {matched_by_id} / {len(round_trip_keyframes_list)}"
                )
                print(f"  Successful pose comparisons: {successful_pose_comparisons}")
                print(f"  Failed pose comparisons: {failed_pose_comparisons}")

                if matched_by_id == 0:
                    print(f"  Original keyframe IDs: {orig_ids}")
                    print(f"  Round-trip keyframe IDs: {round_trip_ids}")
                    if set(orig_ids) != set(round_trip_ids):
                        print(
                            "  ⚠ Keyframe IDs don't match - this may indicate ID preservation issues"
                        )
                        print(
                            "  Note: This doesn't necessarily indicate data loss, but IDs may have changed"
                        )
                    else:
                        print("  ⚠ Keyframe IDs match but couldn't compare poses")
                elif failed_pose_comparisons > 0:
                    print(f"  ⚠ Pose comparison failed for {failed_pose_comparisons} keyframes")
                    if pose_comparison_errors and len(pose_comparison_errors) <= 5:
                        for error in pose_comparison_errors:
                            print(f"    - {error}")
                    elif pose_comparison_errors:
                        print(f"    - (showing first 5 of {len(pose_comparison_errors)} errors)")
                        for error in pose_comparison_errors[:5]:
                            print(f"    - {error}")
                else:
                    print(
                        "  ⚠ Could not compare poses (keyframes found but no pose differences calculated)"
                    )
            else:
                print("  ⚠ Could not compare poses (no keyframes found)")

        # Compare descriptors (exact match for keyframes)
        print("\nDescriptor Comparison:")
        descriptor_matches = 0
        descriptor_mismatches = 0

        for kf in map_obj.keyframes:
            if not kf or kf.is_bad() or kf.des is None:
                continue
            if kf.id in original_keyframes_dict:
                orig_kf = original_keyframes_dict[kf.id]
                if orig_kf.des is not None and orig_kf.des.shape == kf.des.shape:
                    if np.array_equal(orig_kf.des, kf.des):
                        descriptor_matches += 1
                    else:
                        descriptor_mismatches += 1
                        # Check how many descriptors differ
                        if descriptor_mismatches <= 3:  # Only print first few
                            diff_count = np.sum(orig_kf.des != kf.des)
                            print(
                                f"  WARNING: Keyframe {kf.id} has {diff_count} descriptor differences"
                            )

        total_compared = descriptor_matches + descriptor_mismatches
        if total_compared > 0:
            print(f"  Descriptors match exactly: {descriptor_matches} / {total_compared} keyframes")
            if descriptor_mismatches == 0:
                print("  ✓ All descriptors match exactly")
            else:
                print(f"  ✗ {descriptor_mismatches} keyframes have descriptor differences")

        # Summary of comparison
        print("\n" + "=" * 70)
        print("Round-Trip Comparison Summary:")
        print("=" * 70)

        all_preserved = (
            counts_match
            and max_position_diff < 1e-3
            and max_pose_translation_diff < 1e-3
            and max_pose_rotation_diff < 1e-3
            and descriptor_mismatches == 0
        )

        if all_preserved:
            print("  ✓ All data preserved correctly through round-trip!")
            print("  ✓ No significant differences detected")
        else:
            print("  ⚠ Some differences detected:")
            if not counts_match:
                print("    - Count mismatch")
            if max_position_diff >= 1e-3:
                print(f"    - Map point positions differ (max: {max_position_diff:.2e})")
            if max_pose_translation_diff >= 1e-3 or max_pose_rotation_diff >= 1e-3:
                print(
                    f"    - Keyframe poses differ (translation max: {max_pose_translation_diff:.2e}, rotation max: {max_pose_rotation_diff:.2e} rad)"
                )
            if descriptor_mismatches > 0:
                print(f"    - {descriptor_mismatches} keyframes have descriptor differences")

    # Summary
    print("\nRound-trip Serialization Check:")
    all_checks_passed = (
        keyframes_with_valid_des > 0
        and keyframes_with_matching_des_kps > 0
        and test_keyframes_with_retrievable_points > 0
        and points_with_valid_3d > 0
        and keyframes_with_valid_pose > 0
    )

    if all_checks_passed:
        print("  ✓ All round-trip serialization checks passed!")
        print("  ✓ Data preserved correctly through Python → C++ → Python")
    else:
        print("  ✗ Some checks failed - data may have been lost in round-trip")
        if keyframes_with_valid_des == 0:
            print("    - Missing descriptors")
        if test_keyframes_with_retrievable_points == 0:
            print("    - Missing map point associations")
        if points_with_valid_3d == 0:
            print("    - Invalid 3D positions")
        if keyframes_with_valid_pose == 0:
            print("    - Invalid poses")
        raise RuntimeError("Round-trip serialization check failed")

    # If original map was provided, verify data matches
    if original_map_obj is not None:
        print("\nRound-trip Data Integrity Check:")
        data_integrity_passed = (
            counts_match
            and max_position_diff < 1e-3
            and max_pose_translation_diff < 1e-3
            and max_pose_rotation_diff < 1e-3
            and descriptor_mismatches == 0
        )

        if data_integrity_passed:
            print("  ✓ All data matches original map!")
            print("  ✓ No significant differences detected")
        else:
            print("  ✗ Data integrity check failed - significant differences detected!")
            if not counts_match:
                print("    - Count mismatch")
            if max_position_diff >= 1e-3:
                print(f"    - Map point positions differ (max: {max_position_diff:.2e})")
            if max_pose_translation_diff >= 1e-3 or max_pose_rotation_diff >= 1e-3:
                print(
                    f"    - Keyframe poses differ (translation max: {max_pose_translation_diff:.2e}, rotation max: {max_pose_rotation_diff:.2e} rad)"
                )
            if descriptor_mismatches > 0:
                print(f"    - {descriptor_mismatches} keyframes have descriptor differences")
            raise RuntimeError(
                f"Round-trip data integrity check failed: positions differ by {max_position_diff:.2e}, "
                f"{descriptor_mismatches} keyframes have descriptor differences"
            )

    print("\nPython → C++ → Python round-trip map loading tests passed!")
    return map_obj


def main():
    """Main test function."""
    # Get paths to round-trip map and original map
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, "test_data")
    round_trip_map_file = os.path.join(test_data_dir, "python_to_cpp_to_python_map.json")
    original_map_file = os.path.join(test_data_dir, "python_saved_map.json")

    # Try to find the round-trip map file
    if not os.path.exists(round_trip_map_file):
        # Try alternative paths
        possible_paths = [
            round_trip_map_file,
            os.path.join(
                script_dir, "../tests_cpp/../tests_py/test_data/python_to_cpp_to_python_map.json"
            ),
            "pyslam/slam/cpp/tests_py/test_data/python_to_cpp_to_python_map.json",
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                round_trip_map_file = path
                found = True
                break

        if not found:
            print("ERROR: Round-trip map file not found.")
            print("Please run the C++ test first to generate the round-trip map.")
            print(
                "Expected location: pyslam/slam/cpp/tests_py/test_data/python_to_cpp_to_python_map.json"
            )
            return 1

    # Try to find the original map file (for comparison)
    if not os.path.exists(original_map_file):
        # Try alternative paths
        possible_paths = [
            original_map_file,
            os.path.join(script_dir, "../tests_cpp/../tests_py/test_data/python_saved_map.json"),
            "pyslam/slam/cpp/tests_py/test_data/python_saved_map.json",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                original_map_file = path
                break
        else:
            original_map_file = None  # Not found, will skip comparison
            print(
                "WARNING: Original map file not found. Comparison with ground truth will be skipped."
            )
            print("Expected location: pyslam/slam/cpp/tests_py/test_data/python_saved_map.json")

    try:
        map_obj = load_and_verify_map(round_trip_map_file, original_map_file)
        print("\nAll Python → C++ → Python round-trip serialization tests passed!")
        return 0
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
