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
Test to verify that feature computation and matching results are identical
between Python and C++ implementations when using deterministic Orbslam2Feature2D.
"""

import sys
import os
import numpy as np
import cv2

# Add parent directory to path
kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.abspath(os.path.join(kScriptFolder, "../../../.."))
kDataFolder = os.path.join(kRootFolder, "test/data")

sys.path.insert(0, kRootFolder)

from pyslam.config import Config
from pyslam.config_parameters import Parameters

USE_CPP = True
Parameters.USE_CPP_CORE = USE_CPP

from pyslam.slam.cpp import cpp_module, python_module, CPP_AVAILABLE

if not CPP_AVAILABLE:
    print("❌ C++ core is not available, skipping test")
    sys.exit(0)

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.slam.frame import match_frames


def extract_features_and_match_complete(img_ref, img_cur, config, use_cpp):
    """Extract features and match frames using specified implementation.

    This function runs completely under either Python or C++ core module.
    """
    # Set the core module type
    Parameters.USE_CPP_CORE = use_cpp

    # Import the appropriate modules based on use_cpp
    if use_cpp:
        from pyslam.slam.cpp import cpp_module

        Frame = cpp_module.Frame
        PinholeCamera = cpp_module.PinholeCamera
    else:
        from pyslam.slam.cpp import python_module

        Frame = python_module.Frame
        PinholeCamera = python_module.PinholeCamera

    # Create camera with the appropriate implementation
    cam = PinholeCamera(config)

    # Create frames with the same images
    f_ref = Frame(cam, img_ref.copy(), img_id=0)
    f_cur = Frame(cam, img_cur.copy(), img_id=1)

    # Match frames
    matching_result = match_frames(f_ref, f_cur)

    return {
        "f_ref": f_ref,
        "f_cur": f_cur,
        "idxs_ref": matching_result.idxs1,
        "idxs_cur": matching_result.idxs2,
        "num_matches": len(matching_result.idxs1),
        "kps_ref": f_ref.kpsu,
        "kps_cur": f_cur.kpsu,
        "des_ref": f_ref.des,
        "des_cur": f_cur.des,
        "octaves_ref": f_ref.octaves,
        "octaves_cur": f_cur.octaves,
    }


def compare_keypoints(kps1, kps2, tolerance=1e-6):
    """Compare two sets of keypoints."""
    if len(kps1) != len(kps2):
        return False, f"Different number of keypoints: {len(kps1)} vs {len(kps2)}"

    # Compare coordinates
    coords1 = kps1[:, :2] if kps1.ndim == 2 else kps1
    coords2 = kps2[:, :2] if kps2.ndim == 2 else kps2

    if not np.allclose(coords1, coords2, atol=tolerance):
        max_diff = np.max(np.abs(coords1 - coords2))
        return False, f"Keypoint coordinates differ (max diff: {max_diff})"

    return True, "Keypoints match"


def compare_descriptors(des1, des2):
    """Compare two descriptor matrices."""
    if des1.shape != des2.shape:
        return False, f"Different descriptor shapes: {des1.shape} vs {des2.shape}"

    if not np.array_equal(des1, des2):
        # Count differences
        diff_mask = des1 != des2
        num_diffs = np.sum(diff_mask)
        return False, f"Descriptors differ: {num_diffs} bytes differ out of {des1.size}"

    return True, "Descriptors match"


def compare_matches(matches1, matches2):
    """Compare two sets of matches."""
    if len(matches1["idxs_ref"]) != len(matches2["idxs_ref"]):
        return (
            False,
            f"Different number of matches: {len(matches1['idxs_ref'])} vs {len(matches2['idxs_ref'])}",
        )

    # Sort matches by reference index for comparison
    idxs1_ref = np.array(matches1["idxs_ref"])
    idxs1_cur = np.array(matches1["idxs_cur"])
    idxs2_ref = np.array(matches2["idxs_ref"])
    idxs2_cur = np.array(matches2["idxs_cur"])

    # Sort by reference index
    sort_idx1 = np.argsort(idxs1_ref)
    sort_idx2 = np.argsort(idxs2_ref)

    idxs1_ref_sorted = idxs1_ref[sort_idx1]
    idxs1_cur_sorted = idxs1_cur[sort_idx1]
    idxs2_ref_sorted = idxs2_ref[sort_idx2]
    idxs2_cur_sorted = idxs2_cur[sort_idx2]

    if not np.array_equal(idxs1_ref_sorted, idxs2_ref_sorted):
        return False, "Match reference indices differ"

    if not np.array_equal(idxs1_cur_sorted, idxs2_cur_sorted):
        return False, "Match current indices differ"

    return True, "Matches are identical"


def compare_descriptor_distances(matches_py, matches_cpp, tolerance=1e-6):
    """Compare descriptor distances computed for matched pairs between Python and C++ implementations.

    This verifies that descriptor distances are computed identically in both implementations.
    """
    if len(matches_py["idxs_ref"]) != len(matches_cpp["idxs_ref"]):
        return (
            False,
            f"Different number of matches: {len(matches_py['idxs_ref'])} vs {len(matches_cpp['idxs_ref'])}",
        )

    if len(matches_py["idxs_ref"]) == 0:
        return True, "No matches to compare"

    # Get matched descriptor pairs
    idxs_ref = matches_py["idxs_ref"]
    idxs_cur = matches_py["idxs_cur"]

    des_ref = matches_py["des_ref"]
    des_cur = matches_py["des_cur"]

    # Compute distances using Python implementation
    # Python's hamming_distance now counts bits (fixed to match C++ and OpenCV)
    distances_py = []
    distances_cv2 = []  # Also compute using OpenCV's norm for comparison
    for i in range(len(idxs_ref)):
        des1 = des_ref[idxs_ref[i]]
        des2 = des_cur[idxs_cur[i]]
        # Ensure descriptors are 1D arrays
        if des1.ndim > 1:
            des1 = des1.ravel()
        if des2.ndim > 1:
            des2 = des2.ravel()
        dist = FeatureTrackerShared.descriptor_distance(des1, des2)
        distances_py.append(dist)

        # Also compute using OpenCV's norm (which should count bits correctly)
        try:
            cv2_dist = cv2.norm(des1, des2, cv2.NORM_HAMMING)
            distances_cv2.append(cv2_dist)
        except:
            distances_cv2.append(None)

    distances_py = np.array(distances_py)
    distances_cv2 = np.array([d for d in distances_cv2 if d is not None])

    # Compute distances using C++ implementation via MapPoint.min_des_distance
    # We'll create temporary C++ MapPoints to use their min_des_distance method
    # which internally calls pyslam::descriptor_distance()
    distances_cpp = []
    try:
        MapPoint = cpp_module.MapPoint

        for i in range(len(idxs_ref)):
            des1 = des_ref[idxs_ref[i]]
            des2 = des_cur[idxs_cur[i]]

            # Ensure descriptors are contiguous numpy arrays
            if des1.ndim > 1:
                des1 = des1.ravel()
            if des2.ndim > 1:
                des2 = des2.ravel()

            # Ensure descriptors are uint8 and contiguous
            des1 = np.ascontiguousarray(des1, dtype=np.uint8)
            des2 = np.ascontiguousarray(des2, dtype=np.uint8)

            # Create a temporary MapPoint with des1 as its descriptor
            # MapPoint constructor requires position and color, but we can use dummy values
            # since we only need the descriptor distance computation
            dummy_position = np.array([0.0, 0.0, 0.0])
            dummy_color = np.array([0, 0, 0], dtype=np.uint8)
            temp_mp = MapPoint(dummy_position, dummy_color)

            # Set the descriptor (des is a public member in C++ MapPoint)
            # pybind11 should handle numpy array to cv::Mat conversion automatically
            temp_mp.des = des1

            # Compute distance using C++ MapPoint's min_des_distance
            # pybind11 should handle numpy array to cv::Mat conversion automatically
            dist_cpp = temp_mp.min_des_distance(des2)
            distances_cpp.append(dist_cpp)

        distances_cpp = np.array(distances_cpp)

        # Compare distances
        if distances_py.shape != distances_cpp.shape:
            return (
                False,
                f"Distance arrays have different shapes: {distances_py.shape} vs {distances_cpp.shape}",
            )

        # Check if Python distances match C++ distances
        if not np.allclose(distances_py, distances_cpp, atol=tolerance, rtol=0):
            max_diff = np.max(np.abs(distances_py - distances_cpp))
            max_diff_idx = np.argmax(np.abs(distances_py - distances_cpp))

            # Check if OpenCV distances match C++ (they should, since both count bits)
            cv2_cpp_match = False
            if len(distances_cv2) == len(distances_cpp):
                cv2_cpp_match = np.allclose(distances_cv2, distances_cpp, atol=tolerance, rtol=0)

            return False, (
                f"Descriptor distances differ (max diff: {max_diff} at index {max_diff_idx}, "
                f"py={distances_py[max_diff_idx]}, cpp={distances_cpp[max_diff_idx]}, "
                f"cv2={distances_cv2[max_diff_idx] if max_diff_idx < len(distances_cv2) else 'N/A'}). "
                f"NOTE: Python's hamming_distance counts bytes, not bits (incorrect). "
                f"C++ and OpenCV count bits (correct). "
                f"cv2==cpp: {cv2_cpp_match}"
            )

        return True, f"All {len(distances_py)} descriptor distances match"

    except Exception as e:
        return False, f"Error computing C++ descriptor distances: {e}"


def test_feature_extraction_and_matching():
    """Test that Python and C++ produce identical feature extraction and matching results."""

    print("=" * 80)
    print("Testing Feature Extraction and Matching Equivalence (Python vs C++)")
    print("=" * 80)

    # Setup configuration
    config = Config()
    config.config[config.dataset_type]["settings"] = "settings/KITTI04-12.yaml"
    config.sensor_type = "mono"
    config.get_general_system_settings()

    # Load test images
    img_ref_path = os.path.join(kDataFolder, "kitti06-12.png")
    img_cur_path = os.path.join(kDataFolder, "kitti06-13.png")

    if not os.path.exists(img_ref_path):
        print(f"❌ Test image not found: {img_ref_path}")
        print("   Please ensure test data is available")
        return False

    if not os.path.exists(img_cur_path):
        print(f"❌ Test image not found: {img_cur_path}")
        print("   Please ensure test data is available")
        return False

    img_ref = cv2.imread(img_ref_path, cv2.IMREAD_COLOR)
    img_cur = cv2.imread(img_cur_path, cv2.IMREAD_COLOR)

    if img_ref is None or img_cur is None:
        print("❌ Failed to load test images")
        return False

    print(f"Loaded images: {img_ref.shape} and {img_cur.shape}")

    # Setup deterministic feature tracker
    tracker_config = FeatureTrackerConfigs.ORB2.copy()
    tracker_config["num_features"] = 1000
    tracker_config["deterministic"] = True  # Use deterministic Orbslam2Feature2D

    print(f"\nFeature tracker config: {tracker_config}")

    print("\n" + "=" * 80)
    print("Testing Python Implementation (Complete Run)")
    print("=" * 80)

    # Initialize feature tracker for Python (this sets up FeatureTrackerShared)
    Parameters.USE_CPP_CORE = False
    feature_tracker_py = feature_tracker_factory(**tracker_config)
    FeatureTrackerShared.set_feature_tracker(feature_tracker_py, force=True)

    # Test with Python implementation (complete run)
    matches_py = extract_features_and_match_complete(img_ref, img_cur, config, use_cpp=False)
    print(f"Python - f_ref: {len(matches_py['kps_ref'])} keypoints")
    print(f"Python - f_cur: {len(matches_py['kps_cur'])} keypoints")
    print(f"Python - Matches: {matches_py['num_matches']}")

    print("\n" + "=" * 80)
    print("Testing C++ Implementation (Complete Run)")
    print("=" * 80)

    # Initialize feature tracker for C++ (this sets up FeatureTrackerShared)
    Parameters.USE_CPP_CORE = True
    feature_tracker_cpp = feature_tracker_factory(**tracker_config)
    FeatureTrackerShared.set_feature_tracker(feature_tracker_cpp, force=True)

    # Test with C++ implementation (complete run)
    matches_cpp = extract_features_and_match_complete(img_ref, img_cur, config, use_cpp=True)
    print(f"C++ - f_ref: {len(matches_cpp['kps_ref'])} keypoints")
    print(f"C++ - f_cur: {len(matches_cpp['kps_cur'])} keypoints")
    print(f"C++ - Matches: {matches_cpp['num_matches']}")

    print("\n" + "=" * 80)
    print("Comparing Results")
    print("=" * 80)

    all_passed = True

    # Compare keypoints
    print("\n1. Comparing reference frame keypoints...")
    passed, msg = compare_keypoints(matches_py["kps_ref"], matches_cpp["kps_ref"])
    if passed:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        all_passed = False

    print("\n2. Comparing current frame keypoints...")
    passed, msg = compare_keypoints(matches_py["kps_cur"], matches_cpp["kps_cur"])
    if passed:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        all_passed = False

    # Compare descriptors
    print("\n3. Comparing reference frame descriptors...")
    passed, msg = compare_descriptors(matches_py["des_ref"], matches_cpp["des_ref"])
    if passed:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        all_passed = False

    print("\n4. Comparing current frame descriptors...")
    passed, msg = compare_descriptors(matches_py["des_cur"], matches_cpp["des_cur"])
    if passed:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        all_passed = False

    # Compare octaves
    print("\n5. Comparing reference frame octaves...")
    if np.array_equal(matches_py["octaves_ref"], matches_cpp["octaves_ref"]):
        print(f"   ✅ Octaves match")
    else:
        print(f"   ❌ Octaves differ")
        all_passed = False

    print("\n6. Comparing current frame octaves...")
    if np.array_equal(matches_py["octaves_cur"], matches_cpp["octaves_cur"]):
        print(f"   ✅ Octaves match")
    else:
        print(f"   ❌ Octaves differ")
        all_passed = False

    # Compare matches
    print("\n7. Comparing matches...")
    passed, msg = compare_matches(matches_py, matches_cpp)
    if passed:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        all_passed = False

    # Compare descriptor distances for matched pairs
    print("\n8. Comparing descriptor distances for matched pairs...")
    print("   (Verifying that Python and C++ compute distances identically)")
    passed, msg = compare_descriptor_distances(matches_py, matches_cpp)
    if passed:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Python and C++ implementations are equivalent")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED - Python and C++ implementations differ")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = test_feature_extraction_and_matching()
    sys.exit(0 if success else 1)
