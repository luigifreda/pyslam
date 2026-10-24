#!/usr/bin/env python3
"""
Test script to compare map loading between C++ core and Python core.
This test loads a map from the system state (results/slam_state/map.json)
using both C++ and Python cores, then compares the keyframes to ensure
they are loaded identically.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

from pyslam.config import Config
from pyslam.config_parameters import Parameters
from pyslam.slam.map import Map
from pyslam.utilities.serialization import convert_inf_nan


def load_map_from_system_state(state_path, use_cpp_core):
    """
    Load a map from system state path using specified core.

    Args:
        state_path: Path to the system state directory (e.g., results/slam_state)
        use_cpp_core: If True, use C++ core; if False, use Python core

    Returns:
        Map object loaded from the system state
    """
    # Save original USE_CPP_CORE setting
    original_use_cpp_core = Parameters.USE_CPP_CORE

    try:
        # Set the core type
        Parameters.USE_CPP_CORE = use_cpp_core

        # Construct path to map.json
        map_file_path = os.path.join(state_path, "map.json")

        if not os.path.exists(map_file_path):
            raise FileNotFoundError(f"Map file not found: {map_file_path}")

        # Read the system state JSON file
        with open(map_file_path, "rb") as f:
            loaded_json = json.loads(f.read())

        # Extract map JSON
        map_json = loaded_json.get("map")
        if map_json is None:
            raise ValueError("No 'map' key found in system state JSON")

        # Create map object
        map_obj = Map()

        # Load map based on core type
        if Parameters.USE_CPP_CORE:
            # C++ Map::from_json expects a JSON string
            map_json_clean = convert_inf_nan(map_json)
            map_json_str = (
                json.dumps(map_json_clean) if isinstance(map_json_clean, dict) else map_json_clean
            )
            map_obj.from_json(map_json_str)
        else:
            # Python Map expects a dict
            map_obj.from_json(map_json)

        return map_obj

    finally:
        # Restore original USE_CPP_CORE setting
        Parameters.USE_CPP_CORE = original_use_cpp_core


def compare_keyframe_poses(kf1, kf2, tolerance=1e-6):
    """
    Compare poses of two keyframes.

    Returns:
        tuple: (translation_diff, rotation_diff_angle) in radians
    """
    try:
        pose1 = kf1.pose()
        pose2 = kf2.pose()

        if pose1 is None or pose2 is None:
            return None, None

        # Compare translation
        translation1 = pose1[:3, 3]
        translation2 = pose2[:3, 3]
        translation_diff = np.linalg.norm(translation1 - translation2)

        # Compare rotation (using relative rotation error)
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        relative_R = R2 @ R1.T
        trace = np.trace(relative_R)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        return translation_diff, angle_error
    except Exception as e:
        print(f"  Error comparing poses for KF {kf1.id}: {e}")
        return None, None


def compare_keyframes(cpp_map, python_map, tolerance=1e-6):
    """
    Compare keyframes between C++ and Python loaded maps.

    Args:
        cpp_map: Map loaded with C++ core
        python_map: Map loaded with Python core
        tolerance: Numerical tolerance for comparisons

    Returns:
        dict: Comparison results
    """
    results = {
        "num_keyframes_match": False,
        "keyframe_ids_match": False,
        "poses_match": True,
        "descriptors_match": True,
        "keypoints_match": True,
        "max_translation_diff": 0.0,
        "max_rotation_diff": 0.0,
        "num_pose_comparisons": 0,
        "num_descriptor_comparisons": 0,
        "num_keypoint_comparisons": 0,
        "errors": [],
    }

    # Get valid keyframes (not bad)
    cpp_keyframes = [kf for kf in cpp_map.keyframes if kf and not kf.is_bad()]
    python_keyframes = [kf for kf in python_map.keyframes if kf and not kf.is_bad()]

    # Compare number of keyframes
    num_cpp = len(cpp_keyframes)
    num_python = len(python_keyframes)
    results["num_keyframes_match"] = num_cpp == num_python

    if not results["num_keyframes_match"]:
        results["errors"].append(f"Keyframe count mismatch: C++={num_cpp}, Python={num_python}")
        return results

    # Create keyframe dictionaries by ID
    cpp_kf_dict = {kf.id: kf for kf in cpp_keyframes}
    python_kf_dict = {kf.id: kf for kf in python_keyframes}

    # Check if IDs match
    cpp_ids = set(cpp_kf_dict.keys())
    python_ids = set(python_kf_dict.keys())
    results["keyframe_ids_match"] = cpp_ids == python_ids

    if not results["keyframe_ids_match"]:
        missing_in_python = cpp_ids - python_ids
        missing_in_cpp = python_ids - cpp_ids
        if missing_in_python:
            results["errors"].append(f"Keyframes missing in Python: {sorted(missing_in_python)}")
        if missing_in_cpp:
            results["errors"].append(f"Keyframes missing in C++: {sorted(missing_in_cpp)}")
        return results

    # Compare each keyframe
    for kf_id in sorted(cpp_ids):
        cpp_kf = cpp_kf_dict[kf_id]
        python_kf = python_kf_dict[kf_id]

        # Compare poses
        trans_diff, rot_diff = compare_keyframe_poses(cpp_kf, python_kf, tolerance)
        if trans_diff is not None and rot_diff is not None:
            results["num_pose_comparisons"] += 1
            results["max_translation_diff"] = max(results["max_translation_diff"], trans_diff)
            results["max_rotation_diff"] = max(results["max_rotation_diff"], rot_diff)

            if trans_diff > tolerance or rot_diff > tolerance:
                results["poses_match"] = False
                if len(results["errors"]) < 10:  # Limit error messages
                    results["errors"].append(
                        f"KF {kf_id}: pose diff (trans={trans_diff:.2e}, rot={rot_diff:.2e} rad)"
                    )

        # Compare descriptors
        if cpp_kf.des is not None and python_kf.des is not None:
            if cpp_kf.des.shape == python_kf.des.shape:
                results["num_descriptor_comparisons"] += 1
                if not np.array_equal(cpp_kf.des, python_kf.des):
                    results["descriptors_match"] = False
                    if len(results["errors"]) < 10:
                        diff_count = np.sum(cpp_kf.des != python_kf.des)
                        results["errors"].append(f"KF {kf_id}: {diff_count} descriptor differences")
            else:
                results["descriptors_match"] = False
                results["errors"].append(
                    f"KF {kf_id}: descriptor shape mismatch "
                    f"(C++={cpp_kf.des.shape}, Python={python_kf.des.shape})"
                )
        elif cpp_kf.des is None and python_kf.des is None:
            # Both None, that's fine
            pass
        else:
            results["descriptors_match"] = False
            results["errors"].append(
                f"KF {kf_id}: descriptor None mismatch "
                f"(C++={cpp_kf.des is None}, Python={python_kf.des is None})"
            )

        # Compare keypoints
        if cpp_kf.kps is not None and python_kf.kps is not None:
            if cpp_kf.kps.shape == python_kf.kps.shape:
                results["num_keypoint_comparisons"] += 1
                if not np.allclose(cpp_kf.kps, python_kf.kps, atol=tolerance):
                    results["keypoints_match"] = False
                    if len(results["errors"]) < 10:
                        max_diff = np.max(np.abs(cpp_kf.kps - python_kf.kps))
                        results["errors"].append(f"KF {kf_id}: keypoint diff max={max_diff:.2e}")
            else:
                results["keypoints_match"] = False
                results["errors"].append(
                    f"KF {kf_id}: keypoint shape mismatch "
                    f"(C++={cpp_kf.kps.shape}, Python={python_kf.kps.shape})"
                )
        elif cpp_kf.kps is None and python_kf.kps is None:
            # Both None, that's fine
            pass
        else:
            results["keypoints_match"] = False
            results["errors"].append(
                f"KF {kf_id}: keypoint None mismatch "
                f"(C++={cpp_kf.kps is None}, Python={python_kf.kps is None})"
            )

    return results


def print_comparison_results(results):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 70)
    print("Keyframe Comparison Results")
    print("=" * 70)

    print(f"\nNumber of keyframes match: {'✅' if results['num_keyframes_match'] else '❌'}")
    print(f"Keyframe IDs match: {'✅' if results['keyframe_ids_match'] else '❌'}")
    print(f"Poses match: {'✅' if results['poses_match'] else '❌'}")
    print(f"Descriptors match: {'✅' if results['descriptors_match'] else '❌'}")
    print(f"Keypoints match: {'✅' if results['keypoints_match'] else '❌'}")

    print(f"\nPose comparison statistics:")
    print(f"  Comparisons performed: {results['num_pose_comparisons']}")
    print(f"  Max translation difference: {results['max_translation_diff']:.2e}")
    print(f"  Max rotation difference: {results['max_rotation_diff']:.2e} rad")

    print(f"\nDescriptor comparisons: {results['num_descriptor_comparisons']}")
    print(f"Keypoint comparisons: {results['num_keypoint_comparisons']}")

    if results["errors"]:
        print(f"\nErrors and warnings ({len(results['errors'])}):")
        for error in results["errors"][:20]:  # Show first 20 errors
            print(f"  - {error}")
        if len(results["errors"]) > 20:
            print(f"  ... and {len(results['errors']) - 20} more errors")

    # Overall result
    all_match = (
        results["num_keyframes_match"]
        and results["keyframe_ids_match"]
        and results["poses_match"]
        and results["descriptors_match"]
        and results["keypoints_match"]
    )

    print("\n" + "=" * 70)
    if all_match:
        print("✅ All keyframe comparisons passed!")
    else:
        print("❌ Some keyframe comparisons failed!")
    print("=" * 70)

    return all_match


def main():
    """Main test function."""
    # Get the system state path from config or use default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "../../../..")
    repo_root = os.path.abspath(repo_root)

    # Try to load config to get the system state path
    state_path = None
    try:
        config = Config(repo_root)
        state_path = config.get("SYSTEM_STATE", {}).get("folder_path", "results/slam_state")
        # Make path absolute if relative
        if not os.path.isabs(state_path):
            state_path = os.path.join(repo_root, state_path)
    except Exception as e:
        print(f"Warning: Could not load config, using default path: {e}")
        state_path = os.path.join(repo_root, "results/slam_state")

    print(f"Loading map from system state: {state_path}")

    if not os.path.exists(state_path):
        print(f"ERROR: System state path does not exist: {state_path}")
        print("Please ensure you have a saved map at this location.")
        return 1

    map_file_path = os.path.join(state_path, "map.json")
    if not os.path.exists(map_file_path):
        print(f"ERROR: Map file not found: {map_file_path}")
        return 1

    try:
        # Load map with C++ core
        print("\n" + "=" * 70)
        print("Loading map with C++ core...")
        print("=" * 70)
        cpp_map = load_map_from_system_state(state_path, use_cpp_core=True)
        print(
            f"✅ C++ core: Loaded {len(cpp_map.keyframes)} keyframes, "
            f"{len(cpp_map.points)} points"
        )

        # Load map with Python core
        print("\n" + "=" * 70)
        print("Loading map with Python core...")
        print("=" * 70)
        python_map = load_map_from_system_state(state_path, use_cpp_core=False)
        print(
            f"✅ Python core: Loaded {len(python_map.keyframes)} keyframes, "
            f"{len(python_map.points)} points"
        )

        # Compare keyframes
        print("\n" + "=" * 70)
        print("Comparing keyframes...")
        print("=" * 70)
        comparison_results = compare_keyframes(cpp_map, python_map)
        all_match = print_comparison_results(comparison_results)

        if all_match:
            print("\n✅ All tests passed! C++ and Python cores load keyframes identically.")
            return 0
        else:
            print("\n❌ Tests failed! Differences found between C++ and Python core loading.")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
