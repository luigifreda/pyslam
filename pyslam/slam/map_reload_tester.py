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

import numpy as np
import traceback
import os
import json

from pyslam.utilities.logging import Printer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .slam import Slam


class MapReloadTester:
    """
    Test class to verify that a saved map can be correctly reloaded and matches the original.
    This class reloads the map from the saved path and compares it with the current map.
    """

    # Pose comparison thresholds
    kPoseTransErrorThreshold = 1e-6  # translation threshold for errors
    kPoseAngleErrorThreshold = 1e-6  #  degree rotation threshold for errors
    kPoseTransWarningThreshold = 1e-5  # translation threshold for warnings
    kPoseAngleWarningThreshold = 1e-5  #  degree rotation threshold for warnings

    # 3D position comparison thresholds
    kPositionErrorThreshold = 1e-6  # position difference threshold for errors
    kPositionWarningThreshold = 1e-5  # position difference threshold for warnings

    # Keypoint coordinate comparison tolerances (pixels)
    # Used for kps, kpsu, kps_r, kps_ur (pixel coordinates)
    # JSON serialization can introduce small floating point differences
    kKeypointPixelAbsoluteTolerance = 1e-6  # pixel absolute tolerance for keypoint coordinates

    # Keypoint size comparison tolerance
    # Used for sizes (floating point, may have serialization differences)
    kSizeAbsoluteTolerance = 1e-6  # Small tolerance for sizes to account for JSON serialization

    # Angle comparison tolerance (radians)
    # Used for keypoint angles
    kAngleAbsoluteTolerance = 1e-6  # rad absolute tolerance for angles

    # Depth comparison tolerance (meters)
    # Used for depths
    kDepthAbsoluteTolerance = 1e-6  # 1mm absolute tolerance for depths (in meters)
    kDepthRelativeTolerance = 1e-6  # Relative tolerance for depths

    # Normalized coordinate comparison tolerances
    # Used for kpsn (normalized keypoints)
    kNormalizedCoordinateAbsoluteTolerance = 1e-6  # Absolute tolerance for normalized coordinates

    # Descriptor comparison tolerance (for floating point descriptors, if any)
    kDescriptorAbsoluteTolerance = 1e-6  # Small tolerance for floating point descriptors
    kDescriptorRelativeTolerance = 1e-8  # Very small relative tolerance

    # Output limits (number of items to show per category)
    kMaxPoseErrorsShown = 10
    kMaxPositionErrorsShown = 5
    kMaxWarningsShown = 20
    kMaxOtherErrorsShown = 10

    # Relocalization quality test parameters
    kRelocalizationMaxKeyframesToTest = 10  # Maximum number of keyframes to test for relocalization
    kRelocalizationSamplingInterval = 5  # Sample every Nth keyframe (e.g., 5 = every 5th KF)
    kRelocalizationPoseReasonablenessTransThreshold = (
        0.1  # meters - threshold for pose reasonableness check (translation)
    )
    kRelocalizationPoseReasonablenessAngleThreshold = (
        10.0  # degrees - threshold for pose reasonableness check (rotation)
    )
    kRelocalizationPoseErrorComparisonTransThreshold = 0.01  # meters - threshold for comparing pose errors between original and reloaded (translation)
    kRelocalizationPoseErrorComparisonAngleThreshold = 1.0  # degrees - threshold for comparing pose errors between original and reloaded (rotation)

    def __init__(self):
        self.errors = []
        self.warnings = []

    def test_map_reload(self, slam_instance, saved_path):
        """
        Test that the map can be reloaded and matches the original.

        Args:
            slam_instance: The SLAM instance with the current map
            saved_path: Path where the map was saved

        Returns:
            bool: True if the maps match, False otherwise
        """
        Printer.blue(f"\n{'='*70}")
        Printer.blue("MapReloadTester: Testing map reload and comparison...")
        Printer.blue(f"{'='*70}")

        try:
            # Import here to avoid circular import
            from pyslam.slam.slam import Slam, SlamMode

            # Create a new SLAM instance for reloading
            # We need to use MAP_BROWSER mode to avoid initializing loop closing and volumetric integrator
            reloaded_slam = Slam(
                camera=slam_instance.camera,
                feature_tracker_config=slam_instance.feature_tracker_config,
                loop_detector_config=slam_instance.loop_detector_config,
                semantic_mapping_config=slam_instance.semantic_mapping_config,
                sensor_type=slam_instance.sensor_type,
                environment_type=slam_instance.environment_type,
                slam_mode=SlamMode.MAP_BROWSER,
                config=None,
                headless=True,
            )

            # Load the system state
            reloaded_slam.load_system_state(saved_path)

            # Initialize loop closing for both SLAM instances if needed for comparison
            # (MAP_BROWSER mode doesn't initialize it by default, but we need it for loop DB checks)

            # Initialize for original SLAM if needed
            if (
                slam_instance.loop_closing is None
                and slam_instance.loop_detector_config is not None
            ):
                try:
                    slam_instance.init_loop_closing(
                        slam_instance.loop_detector_config, headless=True
                    )
                except Exception as e:
                    Printer.orange(
                        f"MapReloadTester: Could not initialize loop closing for original SLAM: {e}"
                    )

            # Initialize for reloaded SLAM if needed
            if (
                reloaded_slam.loop_closing is None
                and reloaded_slam.loop_detector_config is not None
            ):
                try:
                    reloaded_slam.init_loop_closing(
                        reloaded_slam.loop_detector_config, headless=True
                    )
                    # Load the loop closing state after initialization
                    reloaded_slam.loop_closing.load(saved_path)
                except Exception as e:
                    Printer.orange(
                        f"MapReloadTester: Could not initialize loop closing for reloaded SLAM: {e}"
                    )

            # Compare the maps
            # Note: If the original map is being optimized/modified after save,
            # differences in poses and 3D positions may be expected
            success = self._compare_maps(
                slam_instance.map, reloaded_slam.map, slam_instance, reloaded_slam, saved_path
            )

            # Clean up
            reloaded_slam.quit()

            # Print summary with categorization
            Printer.blue(f"\n{'='*70}")
            Printer.blue("MapReloadTester: Verification Summary")
            Printer.blue(f"{'='*70}")

            if self.errors:
                Printer.red(f"\n❌ Found {len(self.errors)} errors:")

                # Categorize errors for better reporting
                pose_errors = [e for e in self.errors if "pose mismatch" in e]
                position_errors = [e for e in self.errors if "3D position mismatch" in e]
                point_array_errors = [e for e in self.errors if "point at index" in e]
                observation_errors = [
                    e for e in self.errors if "observations" in e or "num_observations" in e
                ]
                other_errors = [
                    e
                    for e in self.errors
                    if e
                    not in pose_errors + position_errors + point_array_errors + observation_errors
                ]

                if pose_errors:
                    Printer.red(f"\n  Pose mismatches ({len(pose_errors)}):")
                    for error in pose_errors[: self.kMaxPoseErrorsShown]:
                        Printer.red(f"    - {error}")
                    if len(pose_errors) > self.kMaxPoseErrorsShown:
                        Printer.red(
                            f"    ... and {len(pose_errors) - self.kMaxPoseErrorsShown} more pose mismatches"
                        )

                if position_errors:
                    Printer.red(f"\n  3D Position mismatches ({len(position_errors)}):")
                    # Show first few with details, then summary
                    for error in position_errors[: self.kMaxPositionErrorsShown]:
                        Printer.red(f"    - {error}")
                    if len(position_errors) > self.kMaxPositionErrorsShown:
                        Printer.red(
                            f"    ... and {len(position_errors) - self.kMaxPositionErrorsShown} more position mismatches"
                        )

                if point_array_errors:
                    Printer.red(f"\n  Point array mismatches ({len(point_array_errors)}):")
                    for error in point_array_errors:
                        Printer.red(f"    - {error}")

                if observation_errors:
                    Printer.red(f"\n  Observation mismatches ({len(observation_errors)}):")
                    for error in observation_errors:
                        Printer.red(f"    - {error}")

                if other_errors:
                    Printer.red(f"\n  Other errors ({len(other_errors)}):")
                    for error in other_errors[: self.kMaxOtherErrorsShown]:
                        Printer.red(f"    - {error}")
                    if len(other_errors) > self.kMaxOtherErrorsShown:
                        Printer.red(
                            f"    ... and {len(other_errors) - self.kMaxOtherErrorsShown} more errors"
                        )
            else:
                Printer.green("\n✅ No errors found! Maps match correctly.")

            if self.warnings:
                Printer.orange(f"\n⚠️  Found {len(self.warnings)} warnings:")
                for warning in self.warnings[: self.kMaxWarningsShown]:
                    Printer.orange(f"  - {warning}")
                if len(self.warnings) > self.kMaxWarningsShown:
                    Printer.orange(
                        f"  ... and {len(self.warnings) - self.kMaxWarningsShown} more warnings"
                    )

            Printer.blue(f"{'='*70}\n")

            return success

        except Exception as e:
            Printer.red(f"MapReloadTester: Exception during test: {e}")
            Printer.red(f"Traceback: {traceback.format_exc()}")
            return False

    def _compare_maps(
        self, original_map, reloaded_map, original_slam=None, reloaded_slam=None, saved_path=None
    ):
        """Compare two maps and report differences.

        Args:
            original_map: The original map to compare
            reloaded_map: The reloaded map to compare
            original_slam: Optional SLAM instance for original map (for loop DB checks)
            reloaded_slam: Optional SLAM instance for reloaded map (for loop DB checks)
        """
        self.errors = []
        self.warnings = []

        # Check basic statistics
        Printer.blue("\n1. Basic Statistics:")
        orig_frames = len(original_map.frames)
        reload_frames = len(reloaded_map.frames)
        orig_kfs = len(original_map.keyframes)
        reload_kfs = len(reloaded_map.keyframes)
        orig_points = len(original_map.points)
        reload_points = len(reloaded_map.points)

        Printer.blue(f"  Frames: {orig_frames} -> {reload_frames}")
        Printer.blue(f"  Keyframes: {orig_kfs} -> {reload_kfs}")
        Printer.blue(f"  Points: {orig_points} -> {reload_points}")

        # Frames are stored in a limited deque; count mismatches are often benign.
        if orig_frames != reload_frames:
            self.warnings.append(
                f"Frame count mismatch (deque truncation likely): {orig_frames} != {reload_frames}"
            )
        if orig_kfs != reload_kfs:
            self.errors.append(f"Keyframe count mismatch: {orig_kfs} != {reload_kfs}")
        if orig_points != reload_points:
            self.errors.append(f"Point count mismatch: {orig_points} != {reload_points}")
            # Additional diagnostic: Check if point count mismatch is due to bad points
            Printer.orange(
                f"\n⚠️  Point count mismatch: {orig_points} != {reload_points} (difference: {orig_points - reload_points})"
            )
            Printer.orange(
                "   This suggests some map points were filtered out during save/load (likely marked as bad)"
            )

        # Check max IDs
        Printer.blue("\n2. Max IDs:")
        Printer.blue(f"  max_frame_id: {original_map.max_frame_id} -> {reloaded_map.max_frame_id}")
        Printer.blue(
            f"  max_keyframe_id: {original_map.max_keyframe_id} -> {reloaded_map.max_keyframe_id}"
        )
        Printer.blue(f"  max_point_id: {original_map.max_point_id} -> {reloaded_map.max_point_id}")

        if original_map.max_frame_id != reloaded_map.max_frame_id:
            self.errors.append(
                f"max_frame_id mismatch: {original_map.max_frame_id} != {reloaded_map.max_frame_id}"
            )
        if original_map.max_keyframe_id != reloaded_map.max_keyframe_id:
            self.errors.append(
                f"max_keyframe_id mismatch: {original_map.max_keyframe_id} != {reloaded_map.max_keyframe_id}"
            )
        if original_map.max_point_id != reloaded_map.max_point_id:
            self.errors.append(
                f"max_point_id mismatch: {original_map.max_point_id} != {reloaded_map.max_point_id}"
            )

        # Check keyframes
        Printer.blue("\n3. Verifying Keyframes:")
        kf_ids_orig = {kf.id for kf in original_map.keyframes if kf is not None and not kf.is_bad()}
        kf_ids_reload = {
            kf.id for kf in reloaded_map.keyframes if kf is not None and not kf.is_bad()
        }

        Printer.blue(f"  Checking {len(kf_ids_orig)} keyframes...")

        if kf_ids_orig != kf_ids_reload:
            self.errors.append(f"Keyframe IDs mismatch: {kf_ids_orig} != {kf_ids_reload}")

        for kf_id in kf_ids_orig:
            orig_kf = next((kf for kf in original_map.keyframes if kf.id == kf_id), None)
            reload_kf = next((kf for kf in reloaded_map.keyframes if kf.id == kf_id), None)

            if orig_kf is None or reload_kf is None:
                self.errors.append(f"Keyframe {kf_id} missing in one of the maps")
                continue

            # Check pose
            # Use more lenient tolerance for poses as they may be optimized after save
            try:
                orig_Tcw = orig_kf.Tcw()
                reload_Tcw = reload_kf.Tcw()
                if orig_Tcw is not None and reload_Tcw is not None:
                    # More lenient tolerance: 1mm translation, 0.1 degree rotation
                    trans_diff = np.linalg.norm(orig_Tcw[:3, 3] - reload_Tcw[:3, 3])
                    rot_diff = orig_Tcw[:3, :3] @ reload_Tcw[:3, :3].T
                    angle_diff = (
                        np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)) * 180 / np.pi
                    )

                    if (
                        trans_diff > self.kPoseTransErrorThreshold
                        or angle_diff > self.kPoseAngleErrorThreshold
                    ):
                        pose_diff = np.abs(orig_Tcw - reload_Tcw)
                        max_diff = np.max(pose_diff)
                        self.errors.append(
                            f"Keyframe {kf_id} pose mismatch: max_diff={max_diff:.12e}, trans_diff={trans_diff:.12e}, angle_diff={angle_diff:.10e}deg"
                        )
                    elif (
                        trans_diff > self.kPoseTransWarningThreshold
                        or angle_diff > self.kPoseAngleWarningThreshold
                    ):
                        self.warnings.append(
                            f"Keyframe {kf_id} pose small difference: trans_diff={trans_diff:.12e}, angle_diff={angle_diff:.10e}deg"
                        )
            except Exception as e:
                self.warnings.append(f"Keyframe {kf_id} pose check failed: {e}")

            # Check timestamp - use tolerance for floating point serialization differences
            if orig_kf.timestamp is not None and reload_kf.timestamp is not None:
                # Timestamps are typically floats, allow small serialization differences
                if abs(orig_kf.timestamp - reload_kf.timestamp) > 1e-9:
                    self.errors.append(
                        f"Keyframe {kf_id} timestamp mismatch: {orig_kf.timestamp} != {reload_kf.timestamp}"
                    )
            elif orig_kf.timestamp is None != reload_kf.timestamp is None:
                self.errors.append(
                    f"Keyframe {kf_id} timestamp None mismatch: {orig_kf.timestamp} != {reload_kf.timestamp}"
                )

            # Check camera
            try:
                if orig_kf.camera is not None and reload_kf.camera is not None:
                    if orig_kf.camera.width != reload_kf.camera.width:
                        self.errors.append(
                            f"Keyframe {kf_id} camera width mismatch: {orig_kf.camera.width} != {reload_kf.camera.width}"
                        )
                    if orig_kf.camera.height != reload_kf.camera.height:
                        self.errors.append(
                            f"Keyframe {kf_id} camera height mismatch: {orig_kf.camera.height} != {reload_kf.camera.height}"
                        )
            except Exception as e:
                self.warnings.append(f"Keyframe {kf_id} camera check failed: {e}")

            # Check keypoints (kps)
            self._check_keypoints(orig_kf, reload_kf, kf_id)

            # Check descriptors (des and des_r)
            self._check_descriptors(orig_kf, reload_kf, kf_id)

            # Check keypoint properties (octaves, sizes, angles)
            self._check_keypoint_properties(orig_kf, reload_kf, kf_id)

            # Check depths and stereo information
            self._check_depths_stereo(orig_kf, reload_kf, kf_id)

            # Check keyframe points array (map point associations)
            self._check_keyframe_points(orig_kf, reload_kf, kf_id)

            # Check covisibility graph and spanning tree
            self._check_keyframe_graph(orig_kf, reload_kf, kf_id, original_map, reloaded_map)

            # Check per-KF coverage (map points and observations)
            self._check_keyframe_coverage(orig_kf, reload_kf, kf_id)

        Printer.blue(
            f"  Verified {len(kf_ids_orig)} keyframes (pose, timestamp, camera, keypoints, descriptors, graph)"
        )

        # Check local map selection sanity for a sample of keyframes
        self._check_local_map_selection(original_map, reloaded_map)

        # Check loop DB mapping if SLAM instances are available
        if original_slam is not None and reloaded_slam is not None:
            self._check_loop_db_mapping(original_slam, reloaded_slam, saved_path)

        # Check relocalization quality if SLAM instances are available
        if original_slam is not None and reloaded_slam is not None:
            self._check_relocalization_quality(
                original_slam, reloaded_slam, original_map, reloaded_map
            )

        # Check map points
        Printer.blue("\n4. Verifying Map Points:")
        point_ids_orig = {mp.id for mp in original_map.points if mp is not None and not mp.is_bad()}
        point_ids_reload = {
            mp.id for mp in reloaded_map.points if mp is not None and not mp.is_bad()
        }

        if point_ids_orig != point_ids_reload:
            self.errors.append(f"Map point IDs mismatch: {point_ids_orig} != {point_ids_reload}")

        for point_id in point_ids_orig:
            orig_mp = next((mp for mp in original_map.points if mp.id == point_id), None)
            reload_mp = next((mp for mp in reloaded_map.points if mp.id == point_id), None)

            if orig_mp is None or reload_mp is None:
                self.errors.append(f"Map point {point_id} missing in one of the maps")
                continue

            # Check 3D position
            # Use more lenient tolerance for positions as they may be optimized after save
            try:
                orig_pt = np.asarray(orig_mp.pt(), dtype=np.float64)
                reload_pt = np.asarray(reload_mp.pt(), dtype=np.float64)
                pos_diff = np.linalg.norm(orig_pt - reload_pt)

                if pos_diff > self.kPositionErrorThreshold:
                    self.errors.append(
                        f"Map point {point_id} 3D position mismatch: diff={pos_diff:.12e}, "
                        f"orig={orig_pt}, reload={reload_pt}"
                    )
                elif pos_diff > self.kPositionWarningThreshold:
                    self.warnings.append(
                        f"Map point {point_id} 3D position small difference: diff={pos_diff:.12e}"
                    )
            except Exception as e:
                self.warnings.append(f"Map point {point_id} 3D position check failed: {e}")

            # Check color - integers should match exactly, but allow for type conversion issues
            if orig_mp.color is not None and reload_mp.color is not None:
                orig_color = np.asarray(orig_mp.color, dtype=np.uint8)
                reload_color = np.asarray(reload_mp.color, dtype=np.uint8)
                if not np.array_equal(orig_color, reload_color):
                    # Color mismatches are non-critical (visualization only)
                    self.warnings.append(f"Map point {point_id} color mismatch (non-critical)")
            elif orig_mp.color is None != reload_mp.color is None:
                # None mismatch is also non-critical
                self.warnings.append(f"Map point {point_id} color None mismatch (non-critical)")

            # Check observations (critical for keyframe generation)
            self._check_map_point_observations(
                orig_mp, reload_mp, point_id, original_map, reloaded_map
            )

        return len(self.errors) == 0

    def _check_keypoints(self, orig_kf, reload_kf, kf_id):
        """Check keypoint coordinates (kps, kpsu, kpsn, kps_r, kps_ur)."""
        # Check kps (left keypoints) - pixels
        if orig_kf.kps is not None and reload_kf.kps is not None:
            if not np.allclose(
                orig_kf.kps, reload_kf.kps, rtol=0, atol=self.kKeypointPixelAbsoluteTolerance
            ):
                self.errors.append(f"Keyframe {kf_id} kps mismatch")
        elif orig_kf.kps is None != reload_kf.kps is None:
            self.errors.append(f"Keyframe {kf_id} kps None mismatch")

        # Check kpsu (undistorted keypoints) - pixels
        if orig_kf.kpsu is not None and reload_kf.kpsu is not None:
            if not np.allclose(
                orig_kf.kpsu, reload_kf.kpsu, rtol=0, atol=self.kKeypointPixelAbsoluteTolerance
            ):
                self.errors.append(f"Keyframe {kf_id} kpsu mismatch")
        elif orig_kf.kpsu is None != reload_kf.kpsu is None:
            self.errors.append(f"Keyframe {kf_id} kpsu None mismatch")

        # Check kpsn (normalized keypoints) - normalized coordinates
        if orig_kf.kpsn is not None and reload_kf.kpsn is not None:
            if not np.allclose(
                orig_kf.kpsn,
                reload_kf.kpsn,
                rtol=0,
                atol=self.kNormalizedCoordinateAbsoluteTolerance,
            ):
                self.errors.append(f"Keyframe {kf_id} kpsn mismatch")
        elif orig_kf.kpsn is None != reload_kf.kpsn is None:
            self.errors.append(f"Keyframe {kf_id} kpsn None mismatch")

        # Check kps_r (right keypoints for stereo) - pixels
        if orig_kf.kps_r is not None and reload_kf.kps_r is not None:
            if not np.allclose(
                orig_kf.kps_r, reload_kf.kps_r, rtol=0, atol=self.kKeypointPixelAbsoluteTolerance
            ):
                self.errors.append(f"Keyframe {kf_id} kps_r mismatch")
        elif orig_kf.kps_r is None != reload_kf.kps_r is None:
            # This is OK - stereo may not be available
            pass

        # Check kps_ur (right u-coordinates for stereo) - pixels
        if orig_kf.kps_ur is not None and reload_kf.kps_ur is not None:
            if not np.allclose(
                orig_kf.kps_ur, reload_kf.kps_ur, rtol=0, atol=self.kKeypointPixelAbsoluteTolerance
            ):
                self.errors.append(f"Keyframe {kf_id} kps_ur mismatch")
        elif orig_kf.kps_ur is None != reload_kf.kps_ur is None:
            # This is OK - stereo may not be available
            pass

    def _check_descriptors(self, orig_kf, reload_kf, kf_id):
        """Check descriptors (des and des_r)."""

        def _compare_des(orig_des_raw, reload_des_raw, label):
            if orig_des_raw is None and reload_des_raw is None:
                return
            if (orig_des_raw is None) != (reload_des_raw is None):
                self.errors.append(f"Keyframe {kf_id} {label} None mismatch")
                return

            orig_des = np.asarray(orig_des_raw)
            reload_des = np.asarray(reload_des_raw)

            if orig_des.shape != reload_des.shape:
                self.errors.append(
                    f"Keyframe {kf_id} {label} shape mismatch: {orig_des.shape} != {reload_des.shape}"
                )
                return

            # For integer types (e.g., uint8 descriptors), require exact match
            # For floating point types, use tolerance to account for serialization differences
            if np.issubdtype(orig_des.dtype, np.integer) and np.issubdtype(
                reload_des.dtype, np.integer
            ):
                # Integer descriptors should match exactly (they're used for binary matching)
                if not np.array_equal(orig_des, reload_des):
                    self.errors.append(f"Keyframe {kf_id} {label} mismatch")
            else:
                # Floating point descriptors may have small serialization differences
                if not np.allclose(
                    orig_des,
                    reload_des,
                    rtol=self.kDescriptorRelativeTolerance,
                    atol=self.kDescriptorAbsoluteTolerance,
                ):
                    self.errors.append(f"Keyframe {kf_id} {label} mismatch")

        # Check left descriptors
        _compare_des(orig_kf.des, reload_kf.des, "descriptors")

        # Check right descriptors (for stereo)
        _compare_des(orig_kf.des_r, reload_kf.des_r, "descriptors_r")

    def _check_keypoint_properties(self, orig_kf, reload_kf, kf_id):
        """Check keypoint properties (octaves, sizes, angles)."""
        # Check octaves
        if orig_kf.octaves is not None and reload_kf.octaves is not None:
            if not np.array_equal(orig_kf.octaves, reload_kf.octaves):
                self.errors.append(f"Keyframe {kf_id} octaves mismatch")
        elif orig_kf.octaves is None != reload_kf.octaves is None:
            self.errors.append(f"Keyframe {kf_id} octaves None mismatch")

        # Check octaves_r (for stereo)
        if orig_kf.octaves_r is not None and reload_kf.octaves_r is not None:
            if not np.array_equal(orig_kf.octaves_r, reload_kf.octaves_r):
                self.errors.append(f"Keyframe {kf_id} octaves_r mismatch")
        elif orig_kf.octaves_r is None != reload_kf.octaves_r is None:
            # This is OK - stereo may not be available
            pass

        # Check sizes - use tolerance for floating point serialization differences
        if orig_kf.sizes is not None and reload_kf.sizes is not None:
            if not np.allclose(
                orig_kf.sizes, reload_kf.sizes, rtol=0, atol=self.kSizeAbsoluteTolerance
            ):
                self.errors.append(f"Keyframe {kf_id} sizes mismatch")
        elif orig_kf.sizes is None != reload_kf.sizes is None:
            self.errors.append(f"Keyframe {kf_id} sizes None mismatch")

        # Check angles - radians
        if orig_kf.angles is not None and reload_kf.angles is not None:
            if not np.allclose(
                orig_kf.angles, reload_kf.angles, rtol=0, atol=self.kAngleAbsoluteTolerance
            ):
                self.errors.append(f"Keyframe {kf_id} angles mismatch")
        elif orig_kf.angles is None != reload_kf.angles is None:
            self.errors.append(f"Keyframe {kf_id} angles None mismatch")

    def _check_depths_stereo(self, orig_kf, reload_kf, kf_id):
        """Check depths and stereo information."""
        # Check depths - meters
        if orig_kf.depths is not None and reload_kf.depths is not None:
            if not np.allclose(
                orig_kf.depths,
                reload_kf.depths,
                rtol=self.kDepthRelativeTolerance,
                atol=self.kDepthAbsoluteTolerance,
            ):
                self.errors.append(f"Keyframe {kf_id} depths mismatch")
        elif orig_kf.depths is None != reload_kf.depths is None:
            # This is OK - depths may not be available for monocular
            pass

        # Check median_depth - meters
        # Note: median_depth differences are expected and non-critical because:
        # 1. Python uses np.median() which uses a sophisticated algorithm
        # 2. C++ uses manual sorting and for even-sized arrays, averages the two middle elements
        # 3. These can produce slightly different results due to floating-point precision
        # 4. The differences are typically < 1e-6 meters (sub-millimeter), which is negligible
        if orig_kf.median_depth is not None and reload_kf.median_depth is not None:
            if not np.allclose(
                orig_kf.median_depth,
                reload_kf.median_depth,
                rtol=self.kDepthRelativeTolerance,
                atol=self.kDepthAbsoluteTolerance,
            ):
                self.warnings.append(
                    f"Keyframe {kf_id} median_depth mismatch (non-critical): "
                    f"orig={orig_kf.median_depth:.10f}, reload={reload_kf.median_depth:.10f}, "
                    f"diff={abs(orig_kf.median_depth - reload_kf.median_depth):.10e}"
                )
        elif orig_kf.median_depth is None != reload_kf.median_depth is None:
            # This is OK
            pass

    def _check_keyframe_points(self, orig_kf, reload_kf, kf_id):
        """Check keyframe points array (map point associations at each keypoint index)."""
        if orig_kf.points is None and reload_kf.points is None:
            return

        if orig_kf.points is None or reload_kf.points is None:
            self.errors.append(f"Keyframe {kf_id} points array None mismatch")
            return

        if len(orig_kf.points) != len(reload_kf.points):
            self.errors.append(
                f"Keyframe {kf_id} points array length mismatch: {len(orig_kf.points)} != {len(reload_kf.points)}"
            )
            return

        # Check that map points are associated at the same indices
        mismatches = []
        for idx in range(len(orig_kf.points)):
            orig_mp = orig_kf.points[idx]
            reload_mp = reload_kf.points[idx]

            if orig_mp is None and reload_mp is None:
                continue

            if orig_mp is None or reload_mp is None:
                mismatches.append((idx, orig_mp, reload_mp))
                continue

            if orig_mp.id != reload_mp.id:
                self.errors.append(
                    f"Keyframe {kf_id} point at index {idx} ID mismatch: {orig_mp.id} != {reload_mp.id}"
                )

        # Only report mismatches if they're not due to missing map points (bad points filtered during save)
        # Filter out mismatches where the original map point doesn't exist in the reloaded map
        # (these are expected - bad points are filtered out)
        filtered_mismatches = []
        for idx, orig_mp, reload_mp in mismatches:
            # If original has a map point but reloaded has None, check if the map point exists in reloaded map
            if orig_mp is not None and reload_mp is None:
                # Check if this map point exists in the reloaded map (might be filtered as bad)
                # We'll still report it, but with a note that it might be expected
                filtered_mismatches.append((idx, orig_mp, reload_mp, "orig_mp_missing_in_reload"))
            # If original has None but reloaded has a map point, this is more problematic
            elif orig_mp is None and reload_mp is not None:
                filtered_mismatches.append((idx, orig_mp, reload_mp, "reload_mp_extra"))
            else:
                filtered_mismatches.append((idx, orig_mp, reload_mp, "both_present"))

        # Report point array mismatches with more detail and diagnostic info
        if filtered_mismatches:
            for idx, orig_mp, reload_mp, mismatch_type in filtered_mismatches:
                orig_id = orig_mp.id if orig_mp is not None else None
                reload_id = reload_mp.id if reload_mp is not None else None

                # Check if the reloaded map point has this keyframe in its observations
                diagnostic_info = ""
                if reload_mp is not None:
                    try:
                        reload_obs = reload_mp.observations()
                        has_this_kf = any(kf.id == kf_id for kf, _ in reload_obs)
                        obs_indices = [obs_idx for kf, obs_idx in reload_obs if kf.id == kf_id]
                        diagnostic_info = f" (reloaded MP {reload_id} has KF {kf_id} in obs: {has_this_kf}, at indices: {obs_indices})"
                    except:
                        pass

                # Check if original map point has this keyframe in its observations
                if orig_mp is not None:
                    try:
                        orig_obs = orig_mp.observations()
                        has_this_kf = any(kf.id == kf_id for kf, _ in orig_obs)
                        obs_indices = [obs_idx for kf, obs_idx in orig_obs if kf.id == kf_id]
                        diagnostic_info += f" (original MP {orig_id} has KF {kf_id} in obs: {has_this_kf}, at indices: {obs_indices})"
                    except:
                        pass

                # Add note about mismatch type
                if mismatch_type == "orig_mp_missing_in_reload":
                    diagnostic_info += " [NOTE: Original MP may be filtered as bad during save]"
                elif mismatch_type == "reload_mp_extra":
                    diagnostic_info += " [WARNING: Reloaded map has MP where original had None - possible observation-based reconstruction]"

                self.errors.append(
                    f"Keyframe {kf_id} point at index {idx} mismatch: orig={orig_id}, reload={reload_id}{diagnostic_info}"
                )

    def _check_keyframe_graph(self, orig_kf, reload_kf, kf_id, original_map, reloaded_map):
        """Check covisibility graph and spanning tree relationships."""
        # Check parent
        orig_parent_id = orig_kf.parent.id if orig_kf.parent is not None else None
        reload_parent_id = reload_kf.parent.id if reload_kf.parent is not None else None
        if orig_parent_id != reload_parent_id:
            self.errors.append(
                f"Keyframe {kf_id} parent mismatch: {orig_parent_id} != {reload_parent_id}"
            )

        # Check children
        orig_children_ids = {c.id for c in orig_kf.get_children()}
        reload_children_ids = {c.id for c in reload_kf.get_children()}
        if orig_children_ids != reload_children_ids:
            self.errors.append(
                f"Keyframe {kf_id} children mismatch: {orig_children_ids} != {reload_children_ids}"
            )

        # Check loop edges
        orig_loop_ids = {kf.id for kf in orig_kf.get_loop_edges()}
        reload_loop_ids = {kf.id for kf in reload_kf.get_loop_edges()}
        if orig_loop_ids != reload_loop_ids:
            self.errors.append(
                f"Keyframe {kf_id} loop edges mismatch: {orig_loop_ids} != {reload_loop_ids}"
            )

        # Check covisibility connections (sample a few to avoid too much output)
        try:
            orig_covisible = orig_kf.get_covisible_keyframes()
            reload_covisible = reload_kf.get_covisible_keyframes()
            orig_cov_ids = {kf.id for kf in orig_covisible}
            reload_cov_ids = {kf.id for kf in reload_covisible}
            if orig_cov_ids != reload_cov_ids:
                self.errors.append(
                    f"Keyframe {kf_id} covisible keyframes mismatch: {orig_cov_ids} != {reload_cov_ids}"
                )
        except Exception as e:
            self.warnings.append(f"Keyframe {kf_id} covisibility check failed: {e}")

        # Check covisibility weights consistency
        try:
            # Use the proper API method that works for both Python and C++ keyframes
            orig_connected_weights = orig_kf.get_connected_keyframes_weights()
            reload_connected_weights = reload_kf.get_connected_keyframes_weights()

            if orig_connected_weights != reload_connected_weights:
                missing_weights = set(orig_connected_weights.keys()) - set(
                    reload_connected_weights.keys()
                )
                extra_weights = set(reload_connected_weights.keys()) - set(
                    orig_connected_weights.keys()
                )
                weight_mismatches = {
                    kf_id: (orig_connected_weights[kf_id], reload_connected_weights[kf_id])
                    for kf_id in set(orig_connected_weights.keys())
                    & set(reload_connected_weights.keys())
                    if orig_connected_weights[kf_id] != reload_connected_weights[kf_id]
                }

                if missing_weights:
                    self.errors.append(
                        f"Keyframe {kf_id} missing connected weights: {missing_weights}"
                    )
                if extra_weights:
                    self.errors.append(f"Keyframe {kf_id} extra connected weights: {extra_weights}")
                if weight_mismatches:
                    # Show first few mismatches
                    mismatches_str = ", ".join(
                        [
                            f"KF{id}: {orig}->{reload}"
                            for id, (orig, reload) in list(weight_mismatches.items())[:5]
                        ]
                    )
                    if len(weight_mismatches) > 5:
                        mismatches_str += f" ... and {len(weight_mismatches) - 5} more"
                    self.errors.append(
                        f"Keyframe {kf_id} connected weights mismatch: {mismatches_str}"
                    )
        except Exception as e:
            self.warnings.append(f"Keyframe {kf_id} covisibility weights check failed: {e}")

    def _check_map_point_observations(
        self, orig_mp, reload_mp, point_id, original_map, reloaded_map
    ):
        """Check map point observations (which keyframes observe this point and at which indices)."""
        try:
            orig_obs = orig_mp.observations()
            reload_obs = reload_mp.observations()

            # Create sets of (keyframe_id, index) pairs for comparison
            orig_obs_set = {(kf.id, idx) for kf, idx in orig_obs if kf is not None}
            reload_obs_set = {(kf.id, idx) for kf, idx in reload_obs if kf is not None}

            if orig_obs_set != reload_obs_set:
                missing_in_reload = orig_obs_set - reload_obs_set
                extra_in_reload = reload_obs_set - orig_obs_set
                if missing_in_reload:
                    self.errors.append(
                        f"Map point {point_id} missing observations in reloaded map: {missing_in_reload}"
                    )
                if extra_in_reload:
                    self.errors.append(
                        f"Map point {point_id} extra observations in reloaded map: {extra_in_reload}"
                    )

            # Check num_observations
            orig_num_obs = orig_mp.num_observations()
            reload_num_obs = reload_mp.num_observations()
            if orig_num_obs != reload_num_obs:
                self.errors.append(
                    f"Map point {point_id} num_observations mismatch: {orig_num_obs} != {reload_num_obs}"
                )

        except Exception as e:
            self.warnings.append(f"Map point {point_id} observations check failed: {e}")

    def _check_keyframe_coverage(self, orig_kf, reload_kf, kf_id):
        """Check per-KF coverage: number of associated map points and observations."""
        try:
            # Count associated map points (non-None entries in points array)
            orig_num_points = (
                sum(1 for p in orig_kf.points if p is not None) if orig_kf.points is not None else 0
            )
            reload_num_points = (
                sum(1 for p in reload_kf.points if p is not None)
                if reload_kf.points is not None
                else 0
            )

            if orig_num_points != reload_num_points:
                self.errors.append(
                    f"Keyframe {kf_id} associated map points count mismatch: {orig_num_points} != {reload_num_points}"
                )

            # Count total observations (sum of observations from all map points in this KF)
            orig_num_obs = 0
            reload_num_obs = 0

            if orig_kf.points is not None:
                for p in orig_kf.points:
                    if p is not None and not p.is_bad():
                        # Count how many times this KF appears in the map point's observations
                        try:
                            obs = p.observations()
                            if any(kf.id == kf_id for kf, _ in obs):
                                orig_num_obs += 1
                        except:
                            pass

            if reload_kf.points is not None:
                for p in reload_kf.points:
                    if p is not None and not p.is_bad():
                        try:
                            obs = p.observations()
                            if any(kf.id == kf_id for kf, _ in obs):
                                reload_num_obs += 1
                        except:
                            pass

            if orig_num_obs != reload_num_obs:
                self.warnings.append(
                    f"Keyframe {kf_id} total observations count mismatch: {orig_num_obs} != {reload_num_obs} "
                    f"(may indicate weak tracking seeds if large drop)"
                )

        except Exception as e:
            self.warnings.append(f"Keyframe {kf_id} coverage check failed: {e}")

    def _check_local_map_selection(self, original_map, reloaded_map):
        """Check local map selection sanity by recomputing covisible sets for sample keyframes."""
        Printer.blue("\n5. Verifying Local Map Selection:")
        try:
            # Sample a few keyframes to check (first, middle, last)
            orig_kfs_list = [
                kf for kf in original_map.keyframes if kf is not None and not kf.is_bad()
            ]
            reload_kfs_list = [
                kf for kf in reloaded_map.keyframes if kf is not None and not kf.is_bad()
            ]

            if not orig_kfs_list or not reload_kfs_list:
                Printer.blue("  No keyframes available for checking")
                return

            # Sample keyframes: first, middle, last
            sample_indices = [0]
            if len(orig_kfs_list) > 1:
                sample_indices.append(len(orig_kfs_list) // 2)
            if len(orig_kfs_list) > 2:
                sample_indices.append(len(orig_kfs_list) - 1)

            Printer.blue(
                f"  Sampling {len(sample_indices)} keyframes (indices: {sample_indices}) for covisibility check..."
            )

            num_checked = 0
            for idx in sample_indices:
                if idx >= len(orig_kfs_list) or idx >= len(reload_kfs_list):
                    continue

                orig_kf = orig_kfs_list[idx]
                reload_kf = next((kf for kf in reload_kfs_list if kf.id == orig_kf.id), None)

                if reload_kf is None:
                    continue

                try:
                    # Get local covisible sets
                    orig_covisible = orig_kf.get_covisible_keyframes()
                    reload_covisible = reload_kf.get_covisible_keyframes()

                    orig_cov_ids = {kf.id for kf in orig_covisible}
                    reload_cov_ids = {kf.id for kf in reload_covisible}

                    orig_size = len(orig_cov_ids)
                    reload_size = len(reload_cov_ids)

                    if orig_size != reload_size:
                        self.warnings.append(
                            f"Keyframe {orig_kf.id} local covisible set size mismatch: {orig_size} != {reload_size} "
                            f"(degraded local map may cause unstable tracking)"
                        )

                    if orig_cov_ids != reload_cov_ids:
                        missing = orig_cov_ids - reload_cov_ids
                        extra = reload_cov_ids - orig_cov_ids
                        if missing or extra:
                            self.warnings.append(
                                f"Keyframe {orig_kf.id} local covisible set mismatch: missing={missing}, extra={extra}"
                            )

                except Exception as e:
                    self.warnings.append(
                        f"Keyframe {orig_kf.id} local map selection check failed: {e}"
                    )
                else:
                    num_checked += 1

            Printer.blue(f"  Checked {num_checked} keyframes for local map selection consistency")

        except Exception as e:
            self.warnings.append(f"Local map selection check failed: {e}")

    def _check_loop_db_mapping(self, original_slam: "Slam", reloaded_slam: "Slam", saved_path=None):
        """Check loop DB mapping: map_entry_id_to_frame_id coverage and entry count.

        Since loop_detector is in a separate process and not directly accessible,
        we load the saved database maps from JSON files instead.
        """
        Printer.blue("\n6. Verifying Loop DB Mapping:")
        try:
            # The loop detector data is saved to JSON files, not directly accessible from the process
            # We load it from the saved state files
            if saved_path is None:
                self.warnings.append("Loop DB mapping check skipped: saved_path not provided")
                return

            # Load loop detector data from saved state
            db_maps_path = os.path.join(saved_path, "loop_closing_db_maps.json")

            if not os.path.exists(db_maps_path):
                self.warnings.append(
                    f"Loop DB mapping check skipped: database maps file not found at {db_maps_path}"
                )
                return

            # Load the database maps (same file for both original and reloaded since they're loaded from the same path)
            try:
                with open(db_maps_path, "rb") as f:
                    output = json.load(f)

                # Convert keys back to integers (they are transformed to strings by json when saving)
                def convert_dict(d):
                    return {int(k): v for k, v in d.items()} if d else {}

                # Both original and reloaded should have the same data since they're loaded from the same file
                orig_entry_id = output.get("entry_id", 0)
                reload_entry_id = output.get("entry_id", 0)
                orig_map = convert_dict(output.get("map_entry_id_to_frame_id", {}))
                reload_map = convert_dict(output.get("map_entry_id_to_frame_id", {}))

            except Exception as e:
                self.warnings.append(f"Failed to load loop DB maps from {db_maps_path}: {e}")
                return

            # Since both are loaded from the same file, they should be identical
            # But we still check for consistency with the actual maps

            Printer.blue(
                f"  Loaded loop DB data: entry_id={orig_entry_id}, map_size={len(orig_map)}"
            )

            # Check entry_id
            if orig_entry_id != reload_entry_id:
                self.errors.append(
                    f"Loop DB entry_id mismatch: {orig_entry_id} != {reload_entry_id}"
                )

            # Check map_entry_id_to_frame_id
            if len(orig_map) != len(reload_map):
                self.errors.append(
                    f"Loop DB map_entry_id_to_frame_id size mismatch: {len(orig_map)} != {len(reload_map)}"
                )

            # Check coverage - all entry IDs should map to valid keyframe IDs
            orig_kf_ids = {
                kf.id for kf in original_slam.map.keyframes if kf is not None and not kf.is_bad()
            }
            reload_kf_ids = {
                kf.id for kf in reloaded_slam.map.keyframes if kf is not None and not kf.is_bad()
            }

            orig_mapped_frame_ids = set(orig_map.values())
            reload_mapped_frame_ids = set(reload_map.values())

            # Check if mapped frame IDs are valid keyframe IDs
            orig_invalid = orig_mapped_frame_ids - orig_kf_ids
            reload_invalid = reload_mapped_frame_ids - reload_kf_ids

            if orig_invalid:
                self.errors.append(
                    f"Loop DB original map has invalid frame IDs (not in keyframes): {orig_invalid}"
                )
            if reload_invalid:
                self.errors.append(
                    f"Loop DB reloaded map has invalid frame IDs (not in keyframes): {reload_invalid}"
                )

            # Check alignment: entry count should align with KF count
            orig_entry_count = len(orig_map)
            reload_entry_count = len(reload_map)
            orig_kf_count = len(orig_kf_ids)
            reload_kf_count = len(reload_kf_ids)

            # Entry count should be close to KF count (may be slightly different if some KFs weren't added)
            if abs(orig_entry_count - orig_kf_count) > orig_kf_count * 0.1:  # Allow 10% difference
                self.warnings.append(
                    f"Loop DB original entry count ({orig_entry_count}) significantly differs from KF count ({orig_kf_count})"
                )
            if abs(reload_entry_count - reload_kf_count) > reload_kf_count * 0.1:
                self.warnings.append(
                    f"Loop DB reloaded entry count ({reload_entry_count}) significantly differs from KF count ({reload_kf_count})"
                )

            # Per-KF check: verify each KF ID has a DB entry (or at least check coverage)
            orig_kfs_without_db_entry = orig_kf_ids - orig_mapped_frame_ids
            reload_kfs_without_db_entry = reload_kf_ids - reload_mapped_frame_ids

            # Check if missing keyframes are early ones (likely initialization-related)
            orig_missing_sorted = sorted(list(orig_kfs_without_db_entry))
            reload_missing_sorted = sorted(list(reload_kfs_without_db_entry))

            # Early keyframes (first 3) missing from DB is often expected due to initialization
            orig_early_missing = [kf_id for kf_id in orig_missing_sorted if kf_id < 3]
            reload_early_missing = [kf_id for kf_id in reload_missing_sorted if kf_id < 3]
            orig_late_missing = [kf_id for kf_id in orig_missing_sorted if kf_id >= 3]
            reload_late_missing = [kf_id for kf_id in reload_missing_sorted if kf_id >= 3]

            if orig_kfs_without_db_entry:
                # Show first few missing entries
                missing_str = ", ".join([f"KF{kf_id}" for kf_id in orig_missing_sorted[:10]])
                if len(orig_kfs_without_db_entry) > 10:
                    missing_str += f" ... and {len(orig_kfs_without_db_entry) - 10} more"

                # Add context about whether this is expected
                if orig_late_missing:
                    # Late keyframes missing is more concerning
                    self.warnings.append(
                        f"Loop DB original: {len(orig_kfs_without_db_entry)} keyframes without DB entries: {missing_str}"
                        f" (including {len(orig_late_missing)} late keyframes, which may indicate a problem)"
                    )
                elif orig_early_missing:
                    # Only early keyframes missing is often expected
                    self.warnings.append(
                        f"Loop DB original: {len(orig_kfs_without_db_entry)} early keyframes without DB entries: {missing_str}"
                        f" (early KFs may be created before loop detector initialization)"
                    )
                else:
                    self.warnings.append(
                        f"Loop DB original: {len(orig_kfs_without_db_entry)} keyframes without DB entries: {missing_str}"
                    )

            if reload_kfs_without_db_entry:
                # Show first few missing entries
                missing_str = ", ".join([f"KF{kf_id}" for kf_id in reload_missing_sorted[:10]])
                if len(reload_kfs_without_db_entry) > 10:
                    missing_str += f" ... and {len(reload_kfs_without_db_entry) - 10} more"

                # Add context about whether this is expected
                if reload_late_missing:
                    # Late keyframes missing is more concerning
                    self.warnings.append(
                        f"Loop DB reloaded: {len(reload_kfs_without_db_entry)} keyframes without DB entries: {missing_str}"
                        f" (including {len(reload_late_missing)} late keyframes, which may indicate a problem)"
                    )
                elif reload_early_missing:
                    # Only early keyframes missing is often expected
                    self.warnings.append(
                        f"Loop DB reloaded: {len(reload_kfs_without_db_entry)} early keyframes without DB entries: {missing_str}"
                        f" (early KFs may be created before loop detector initialization)"
                    )
                else:
                    self.warnings.append(
                        f"Loop DB reloaded: {len(reload_kfs_without_db_entry)} keyframes without DB entries: {missing_str}"
                    )

            # Summary: DB coverage statistics
            orig_coverage = (
                (1.0 - len(orig_kfs_without_db_entry) / orig_kf_count * 100)
                if orig_kf_count > 0
                else 0.0
            )
            reload_coverage = (
                (1.0 - len(reload_kfs_without_db_entry) / reload_kf_count * 100)
                if reload_kf_count > 0
                else 0.0
            )
            Printer.blue(
                f"  Loop DB coverage: original={orig_coverage:.1f}% ({orig_entry_count}/{orig_kf_count}), "
                f"reloaded={reload_coverage:.1f}% ({reload_entry_count}/{reload_kf_count})"
            )

            # Check if the mapping itself matches (for common entry IDs)
            common_entry_ids = set(orig_map.keys()) & set(reload_map.keys())
            mismatched_mappings = {
                eid: (orig_map[eid], reload_map[eid])
                for eid in common_entry_ids
                if orig_map[eid] != reload_map[eid]
            }

            if mismatched_mappings:
                # Show first few mismatches
                mismatches_str = ", ".join(
                    [
                        f"entry{eid}: KF{orig}->KF{reload}"
                        for eid, (orig, reload) in list(mismatched_mappings.items())[:5]
                    ]
                )
                if len(mismatched_mappings) > 5:
                    mismatches_str += f" ... and {len(mismatched_mappings) - 5} more"
                self.errors.append(f"Loop DB map_entry_id_to_frame_id mismatches: {mismatches_str}")

            # Print summary
            Printer.blue(
                f"  Loop DB verification complete: {len(orig_map)} entries, {len(common_entry_ids)} common entries"
            )

        except Exception as e:
            self.warnings.append(f"Loop DB mapping check failed: {e}")

    def _check_relocalization_quality(
        self, original_slam, reloaded_slam, original_map, reloaded_map
    ):
        """Check relocalization quality by attempting to relocalize using keyframe images.

        This test verifies that relocalization behavior is consistent between original and reloaded maps.
        It tests relocalization by creating frames from keyframe images and attempting to relocalize them.
        """
        Printer.blue("\n7. Verifying Relocalization Quality:")
        try:
            # Check if loop closing is available for both SLAM instances
            if original_slam.loop_closing is None or reloaded_slam.loop_closing is None:
                self.warnings.append(
                    "Relocalization quality check skipped: loop_closing not available for one or both SLAM instances"
                )
                return

            # Get valid keyframes from both maps
            orig_kfs_list = [
                kf for kf in original_map.keyframes if kf is not None and not kf.is_bad()
            ]
            reload_kfs_list = [
                kf for kf in reloaded_map.keyframes if kf is not None and not kf.is_bad()
            ]

            if not orig_kfs_list or not reload_kfs_list:
                self.warnings.append(
                    "Relocalization quality check skipped: no valid keyframes found"
                )
                return

            # Sample keyframes: every Nth keyframe up to max
            # This provides better coverage than just first/middle/last
            sample_indices = []
            for i in range(0, len(orig_kfs_list), self.kRelocalizationSamplingInterval):
                sample_indices.append(i)
                if len(sample_indices) >= self.kRelocalizationMaxKeyframesToTest:
                    break

            # Always include the last keyframe if we haven't reached max
            if len(sample_indices) < self.kRelocalizationMaxKeyframesToTest:
                last_idx = len(orig_kfs_list) - 1
                if last_idx not in sample_indices:
                    sample_indices.append(last_idx)

            # Limit to max keyframes for testing
            sample_indices = sample_indices[: self.kRelocalizationMaxKeyframesToTest]

            num_tested = 0
            num_success_orig = 0
            num_success_reload = 0
            num_both_success = 0
            num_both_fail = 0
            num_mismatch = 0

            for idx in sample_indices:
                if idx >= len(orig_kfs_list):
                    continue

                orig_kf = orig_kfs_list[idx]
                # Find corresponding keyframe in reloaded map
                reload_kf = next((kf for kf in reload_kfs_list if kf.id == orig_kf.id), None)

                if reload_kf is None:
                    continue

                # Check if keyframe has stored image (needed for relocalization test)
                if orig_kf.img is None or reload_kf.img is None:
                    self.warnings.append(
                        f"Relocalization test skipped for KF {orig_kf.id}: image not stored (Frame.is_store_imgs may be False)"
                    )
                    continue

                num_tested += 1

                # Initialize success flags and pose errors
                orig_success = False
                reload_success = False
                orig_pose_error = None
                reload_pose_error = None

                try:
                    # Import Frame here to avoid circular imports
                    from pyslam.slam.frame import Frame

                    # Create test frames from keyframe images
                    # Use a different frame ID to avoid conflicts
                    test_frame_id_orig = original_map.max_frame_id + 1000 + idx
                    test_frame_id_reload = reloaded_map.max_frame_id + 1000 + idx

                    # Create frames from keyframe images
                    # Note: We need to enable image storage temporarily if needed
                    orig_frame = Frame(
                        camera=orig_kf.camera,
                        img=orig_kf.img,
                        img_right=orig_kf.img_right if hasattr(orig_kf, "img_right") else None,
                        depth=orig_kf.depth_img if hasattr(orig_kf, "depth_img") else None,
                        id=test_frame_id_orig,
                        timestamp=orig_kf.timestamp,
                        img_id=orig_kf.img_id,
                    )

                    reload_frame = Frame(
                        camera=reload_kf.camera,
                        img=reload_kf.img,
                        img_right=reload_kf.img_right if hasattr(reload_kf, "img_right") else None,
                        depth=reload_kf.depth_img if hasattr(reload_kf, "depth_img") else None,
                        id=test_frame_id_reload,
                        timestamp=reload_kf.timestamp,
                        img_id=reload_kf.img_id,
                    )

                    # Store original poses for comparison
                    orig_kf_pose = orig_kf.Tcw()
                    reload_kf_pose = reload_kf.Tcw()

                except Exception as e:
                    self.warnings.append(
                        f"Relocalization test: Failed to create test frames for KF {orig_kf.id}: {e}"
                    )
                    # Continue to comparison with both failures
                    orig_success = False
                    reload_success = False

                # Attempt relocalization on original SLAM (only if frame creation succeeded)
                # Note: Relocalization may fail if:
                # 1. The loop detector doesn't find enough candidates (the keyframe might not be in the database yet, or the frame's BoW doesn't match well)
                # 2. Feature matching fails (not enough matches between frame and candidate keyframes)
                # 3. PnP solving fails (not enough inliers)
                # This is expected behavior - we're testing consistency, not success rate
                orig_diagnostics = {}
                try:
                    if "orig_frame" in locals() and orig_kf_pose is not None:
                        # Check if frame has descriptors (needed for relocalization)
                        if orig_frame.des is None or len(orig_frame.des) == 0:
                            self.warnings.append(
                                f"Relocalization test: Frame for KF {orig_kf.id} has no descriptors - relocalization will likely fail"
                            )

                        # Get detection output for diagnostics BEFORE calling relocalize
                        # We need to do this separately because relocalize() modifies the frame state
                        from pyslam.loop_closing.loop_closing import (
                            LoopDetectorTaskType,
                            LoopDetectorTask,
                        )

                        # Create a copy of the frame for diagnostics to avoid modifying the original
                        # Note: We can't easily deep copy a Frame, so we'll get diagnostics from a separate call
                        # but use a fresh frame for the actual relocalization
                        task = LoopDetectorTask(
                            orig_frame,
                            orig_kf.img,
                            LoopDetectorTaskType.RELOCALIZATION,
                            covisible_keyframes=[],
                            connected_keyframes=[],
                        )
                        detection_output = (
                            original_slam.loop_closing.loop_detecting_process.relocalize(task)
                        )

                        # Capture diagnostic information
                        if detection_output is not None:
                            orig_diagnostics["num_candidates"] = (
                                len(detection_output.candidate_idxs)
                                if detection_output.candidate_idxs
                                else 0
                            )
                            orig_diagnostics["candidate_ids"] = (
                                detection_output.candidate_idxs[:5]
                                if detection_output.candidate_idxs
                                else []
                            )  # First 5
                            orig_diagnostics["top_score"] = (
                                detection_output.candidate_scores[0]
                                if detection_output.candidate_scores
                                and len(detection_output.candidate_scores) > 0
                                else None
                            )
                        else:
                            orig_diagnostics["num_candidates"] = 0
                            orig_diagnostics["candidate_ids"] = []
                            orig_diagnostics["top_score"] = None

                        # Now create a fresh frame for the actual relocalization
                        # (since the previous call may have modified the frame state)
                        orig_frame_reloc = Frame(
                            camera=orig_kf.camera,
                            img=orig_kf.img,
                            img_right=orig_kf.img_right if hasattr(orig_kf, "img_right") else None,
                            depth=orig_kf.depth_img if hasattr(orig_kf, "depth_img") else None,
                            id=test_frame_id_orig + 10000,  # Different ID to avoid conflicts
                            timestamp=orig_kf.timestamp,
                            img_id=orig_kf.img_id,
                        )

                        orig_success = original_slam.loop_closing.relocalize(
                            orig_frame_reloc, orig_kf.img
                        )
                        if orig_success:
                            num_success_orig += 1
                            # Check pose accuracy (should be close to keyframe pose)
                            reloc_pose = orig_frame_reloc.Tcw()
                            if reloc_pose is not None and orig_kf_pose is not None:
                                trans_diff = np.linalg.norm(reloc_pose[:3, 3] - orig_kf_pose[:3, 3])
                                rot_diff = reloc_pose[:3, :3] @ orig_kf_pose[:3, :3].T
                                angle_diff = (
                                    np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1))
                                    * 180
                                    / np.pi
                                )
                                orig_pose_error = (trans_diff, angle_diff)

                                # Check if relocalized pose is reasonable (not too far from keyframe)
                                if (
                                    trans_diff
                                    > self.kRelocalizationPoseReasonablenessTransThreshold
                                    or angle_diff
                                    > self.kRelocalizationPoseReasonablenessAngleThreshold
                                ):
                                    self.warnings.append(
                                        f"Relocalization on original SLAM for KF {orig_kf.id} succeeded but pose is far from keyframe: "
                                        f"trans_diff={trans_diff:.4f}m, angle_diff={angle_diff:.2f}deg"
                                    )
                except Exception as e:
                    self.warnings.append(
                        f"Relocalization test failed for original SLAM on KF {orig_kf.id}: {e}"
                    )
                    orig_success = False

                # Attempt relocalization on reloaded SLAM (only if frame creation succeeded)
                reload_diagnostics = {}
                try:
                    if "reload_frame" in locals() and reload_kf_pose is not None:
                        # Check if frame has descriptors (needed for relocalization)
                        if reload_frame.des is None or len(reload_frame.des) == 0:
                            self.warnings.append(
                                f"Relocalization test: Frame for KF {reload_kf.id} has no descriptors - relocalization will likely fail"
                            )

                        # Get detection output for diagnostics BEFORE calling relocalize
                        from pyslam.loop_closing.loop_closing import (
                            LoopDetectorTaskType,
                            LoopDetectorTask,
                        )

                        task = LoopDetectorTask(
                            reload_frame,
                            reload_kf.img,
                            LoopDetectorTaskType.RELOCALIZATION,
                            covisible_keyframes=[],
                            connected_keyframes=[],
                        )
                        detection_output = (
                            reloaded_slam.loop_closing.loop_detecting_process.relocalize(task)
                        )

                        # Capture diagnostic information
                        if detection_output is not None:
                            reload_diagnostics["num_candidates"] = (
                                len(detection_output.candidate_idxs)
                                if detection_output.candidate_idxs
                                else 0
                            )
                            reload_diagnostics["candidate_ids"] = (
                                detection_output.candidate_idxs[:5]
                                if detection_output.candidate_idxs
                                else []
                            )  # First 5
                            reload_diagnostics["top_score"] = (
                                detection_output.candidate_scores[0]
                                if detection_output.candidate_scores
                                and len(detection_output.candidate_scores) > 0
                                else None
                            )
                        else:
                            reload_diagnostics["num_candidates"] = 0
                            reload_diagnostics["candidate_ids"] = []
                            reload_diagnostics["top_score"] = None

                        # Now create a fresh frame for the actual relocalization
                        # (since the previous call may have modified the frame state)
                        reload_frame_reloc = Frame(
                            camera=reload_kf.camera,
                            img=reload_kf.img,
                            img_right=(
                                reload_kf.img_right if hasattr(reload_kf, "img_right") else None
                            ),
                            depth=reload_kf.depth_img if hasattr(reload_kf, "depth_img") else None,
                            id=test_frame_id_reload + 10000,  # Different ID to avoid conflicts
                            timestamp=reload_kf.timestamp,
                            img_id=reload_kf.img_id,
                        )

                        reload_success = reloaded_slam.loop_closing.relocalize(
                            reload_frame_reloc, reload_kf.img
                        )
                        if reload_success:
                            num_success_reload += 1
                            # Check pose accuracy (should be close to keyframe pose)
                            reloc_pose = reload_frame_reloc.Tcw()
                            if reloc_pose is not None and reload_kf_pose is not None:
                                trans_diff = np.linalg.norm(
                                    reloc_pose[:3, 3] - reload_kf_pose[:3, 3]
                                )
                                rot_diff = reloc_pose[:3, :3] @ reload_kf_pose[:3, :3].T
                                angle_diff = (
                                    np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1))
                                    * 180
                                    / np.pi
                                )
                                reload_pose_error = (trans_diff, angle_diff)

                                # Check if relocalized pose is reasonable (not too far from keyframe)
                                if (
                                    trans_diff
                                    > self.kRelocalizationPoseReasonablenessTransThreshold
                                    or angle_diff
                                    > self.kRelocalizationPoseReasonablenessAngleThreshold
                                ):
                                    self.warnings.append(
                                        f"Relocalization on reloaded SLAM for KF {reload_kf.id} succeeded but pose is far from keyframe: "
                                        f"trans_diff={trans_diff:.4f}m, angle_diff={angle_diff:.2f}deg"
                                    )
                except Exception as e:
                    self.warnings.append(
                        f"Relocalization test failed for reloaded SLAM on KF {reload_kf.id}: {e}"
                    )
                    reload_success = False

                # Log diagnostic information
                if orig_diagnostics or reload_diagnostics:
                    diag_msg = f"  KF {orig_kf.id} relocalization diagnostics:"
                    if orig_diagnostics:
                        top_score_str = (
                            f"{orig_diagnostics.get('top_score', 0):.4f}"
                            if orig_diagnostics.get("top_score") is not None
                            else "N/A"
                        )
                        diag_msg += (
                            f" orig(candidates={orig_diagnostics.get('num_candidates', 0)}, "
                        )
                        diag_msg += f"top_score={top_score_str}, "
                        diag_msg += f"success={orig_success})"
                    if reload_diagnostics:
                        top_score_str = (
                            f"{reload_diagnostics.get('top_score', 0):.4f}"
                            if reload_diagnostics.get("top_score") is not None
                            else "N/A"
                        )
                        diag_msg += (
                            f" reload(candidates={reload_diagnostics.get('num_candidates', 0)}, "
                        )
                        diag_msg += f"top_score={top_score_str}, "
                        diag_msg += f"success={reload_success})"
                    Printer.blue(diag_msg)

                # Compare results (always execute, even if exceptions occurred)
                try:
                    if orig_success and reload_success:
                        num_both_success += 1
                        # Compare pose errors if both succeeded
                        if orig_pose_error is not None and reload_pose_error is not None:
                            trans_diff_orig, angle_diff_orig = orig_pose_error
                            trans_diff_reload, angle_diff_reload = reload_pose_error

                            # Check if pose errors are similar (within reasonable tolerance)
                            trans_error_diff = abs(trans_diff_orig - trans_diff_reload)
                            angle_error_diff = abs(angle_diff_orig - angle_diff_reload)

                            if (
                                trans_error_diff
                                > self.kRelocalizationPoseErrorComparisonTransThreshold
                                or angle_error_diff
                                > self.kRelocalizationPoseErrorComparisonAngleThreshold
                            ):
                                self.warnings.append(
                                    f"Relocalization pose error mismatch for KF {orig_kf.id}: "
                                    f"orig error (trans={trans_diff_orig:.4f}m, angle={angle_diff_orig:.2f}deg), "
                                    f"reload error (trans={trans_diff_reload:.4f}m, angle={angle_diff_reload:.2f}deg)"
                                )
                    elif not orig_success and not reload_success:
                        num_both_fail += 1
                        # Both failed - this is acceptable and indicates consistent behavior
                        # Relocalization can fail for various reasons:
                        # - Loop detector doesn't find enough candidates (keyframe might not be in database, or BoW doesn't match)
                        # - Feature matching fails (not enough matches between frame and candidate keyframes)
                        # - PnP solving fails (not enough inliers for pose estimation)
                        # The important thing is that both original and reloaded SLAM behave consistently
                        # (no warning needed - this is expected behavior for some keyframes)
                    else:
                        num_mismatch += 1
                        # Mismatch: one succeeded, one failed
                        status_orig = "SUCCESS" if orig_success else "FAILED"
                        status_reload = "SUCCESS" if reload_success else "FAILED"
                        self.errors.append(
                            f"Relocalization result mismatch for KF {orig_kf.id}: "
                            f"original={status_orig}, reloaded={status_reload} "
                            f"(indicates potential difference in relocalization behavior)"
                        )
                except Exception as e:
                    self.warnings.append(
                        f"Relocalization test: Failed to compare results for KF {orig_kf.id}: {e}"
                    )

            # Print summary
            if num_tested > 0:
                Printer.blue(f"  Tested relocalization on {num_tested} keyframes:")
                Printer.blue(f"    Original SLAM: {num_success_orig}/{num_tested} successful")
                Printer.blue(f"    Reloaded SLAM: {num_success_reload}/{num_tested} successful")
                Printer.blue(f"    Both succeeded: {num_both_success}")
                Printer.blue(f"    Both failed: {num_both_fail}")
                Printer.blue(f"    Mismatches: {num_mismatch}")

                if num_mismatch > 0:
                    self.warnings.append(
                        f"Relocalization behavior differs between original and reloaded maps "
                        f"({num_mismatch}/{num_tested} keyframes with mismatched results). "
                        f"This may indicate differences in loop DB, feature matching, or pose optimization."
                    )
            else:
                self.warnings.append(
                    "Relocalization quality check: no keyframes with stored images found for testing"
                )

        except Exception as e:
            self.warnings.append(f"Relocalization quality check failed: {e}")
            self.warnings.append(f"Traceback: {traceback.format_exc()}")
