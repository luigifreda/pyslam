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

import sys
import cv2
import numpy as np


from pyslam.utilities.logging import Printer
from pyslam.local_features.feature_matcher import FeatureMatcherTypes
from pyslam.config_parameters import Parameters


import atexit
import traceback


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.local_features.feature_tracker import FeatureTracker
    from pyslam.local_features.feature_matcher import FeatureMatcher
    from pyslam.local_features.feature_manager import FeatureManager
    from pyslam.slam.slam import Slam


class SlamFeatureManagerInfo:
    """
    Minimal information about the feature manager used by the SLAM.
    Used by parallel processes to avoid pickling problems.
    """

    def __init__(self, slam: "Slam" = None, feature_manager: "FeatureManager" = None):
        self.feature_descriptor_type = None
        self.feature_descriptor_norm_type = None
        if slam is not None:
            assert slam.feature_tracker is not None
            assert slam.feature_tracker.feature_manager is not None
            self.feature_descriptor_type = slam.feature_tracker.feature_manager.descriptor_type
            self.feature_descriptor_norm_type = slam.feature_tracker.feature_manager.norm_type
        elif feature_manager is not None:
            self.feature_descriptor_type = feature_manager.descriptor_type
            self.feature_descriptor_norm_type = feature_manager.norm_type


# Shared frame stuff. Normally, this information is exclusively used by SLAM.
class FeatureTrackerShared:
    feature_tracker: "FeatureTracker | None" = None
    feature_manager: "FeatureManager | None" = None
    feature_matcher: "FeatureMatcher | None" = None
    descriptor_distance = None
    descriptor_distances = None
    oriented_features = False
    feature_tracker_right: "FeatureTracker | None" = None
    _is_cpp_used = False
    _is_cpp_available = False
    _is_cpp_initialized = False
    _cpp_module_parameters = None

    @staticmethod
    def set_feature_tracker(feature_tracker, force=False):
        from pyslam.slam import FrameBase, Frame  # Import at runtime

        FrameBase._id = 0  # reset the frame counter

        if not force and FeatureTrackerShared.feature_tracker is not None:
            raise Exception("FeatureTrackerShared: Tracker is already set!")
        FeatureTrackerShared.feature_tracker = feature_tracker
        FeatureTrackerShared.feature_manager = feature_tracker.feature_manager
        FeatureTrackerShared.feature_matcher = feature_tracker.matcher
        FeatureTrackerShared.descriptor_distance = (
            feature_tracker.feature_manager.descriptor_distance
        )
        FeatureTrackerShared.descriptor_distances = (
            feature_tracker.feature_manager.descriptor_distances
        )
        FeatureTrackerShared.oriented_features = feature_tracker.feature_manager.oriented_features

        # For the following guys we need to store the images since they need them at each matching step
        if FeatureTrackerShared.feature_matcher is not None and (
            FeatureTrackerShared.feature_matcher.matcher_type == FeatureMatcherTypes.LIGHTGLUE
            or FeatureTrackerShared.feature_matcher.matcher_type == FeatureMatcherTypes.XFEAT
            or FeatureTrackerShared.feature_matcher.matcher_type == FeatureMatcherTypes.LOFTR
        ):
            Frame.is_store_imgs = True

        # Initialize the C++ module with the feature tracker info
        FeatureTrackerShared.init_cpp_module(feature_tracker)

    @staticmethod
    def set_feature_tracker_right(feature_tracker, force=False):
        if not force and FeatureTrackerShared.feature_tracker_right is not None:
            raise Exception("FeatureTrackerShared: Tracker-right is already set!")
        FeatureTrackerShared.feature_tracker_right = feature_tracker

    @staticmethod
    def init_cpp_module(feature_tracker: "FeatureTracker"):
        """Initialize the C++ module with the feature tracker info"""
        try:
            from pyslam.slam.cpp import cpp_module
            from pyslam.slam.cpp import CPP_AVAILABLE
            from pyslam.slam import USE_CPP

            if not CPP_AVAILABLE:
                return

            FeatureTrackerShared._is_cpp_used = USE_CPP
            FeatureTrackerShared._is_cpp_available = True
            FeatureTrackerShared._cpp_module_parameters = cpp_module.Parameters

            # Set C++ FeatureSharedResources static properties from the feature manager
            py_fm = feature_tracker.feature_manager
            cpp_fsr = cpp_module.FeatureSharedResources

            def to_enum_value(e):
                return e.value

            def identity(x):
                return x

            def set_cpp_or_warn(attr, transform):
                if not hasattr(py_fm, attr):
                    Printer.red(f"FeatureTrackerShared: feature_manager.{attr} is missing")
                    return
                value = getattr(py_fm, attr)
                if value is None:
                    Printer.red(f"FeatureTrackerShared: feature_manager.{attr} is None")
                    return
                setattr(cpp_fsr, attr, transform(value))

            # --- attributes and their transformations ---
            attributes = [
                ("scale_factor", identity),
                ("inv_scale_factor", identity),
                ("log_scale_factor", identity),
                ("scale_factors", identity),
                ("inv_scale_factors", identity),
                ("level_sigmas", identity),
                ("level_sigmas2", identity),
                ("inv_level_sigmas2", identity),
                ("num_levels", identity),
                ("num_features", identity),
                ("detector_type", to_enum_value),
                ("descriptor_type", to_enum_value),
                ("norm_type", identity),
                ("oriented_features", identity),
            ]

            for attr, transform in attributes:
                set_cpp_or_warn(attr, transform)

            # Set the feature detection callbacks that will be used by pyslam::MapPoint
            FeatureTrackerShared.setup_feature_detection_callbacks("C++", cpp_module)

            # Register cleanup handler to be called on program exit
            # to prevent hanging on exit if the callbacks are not cleared
            FeatureTrackerShared._register_cleanup_handler()

            # Init the C++ module config parameters
            FeatureTrackerShared.init_cpp_module_config_parameters()

            # Set the C++ module initialized flag
            FeatureTrackerShared._is_cpp_initialized = True
        except Exception as e:
            Printer.orange(f"WARNING: FeatureTrackerShared: cannot set cpp_module: {e}")
            traceback.print_exc()

    @staticmethod
    def setup_feature_detection_callbacks(module_type: str, module: Any):
        """Setup feature detection callbacks using FeatureTrackerShared"""
        from pyslam.utilities.features import convert_keypoints_to_tuples

        if module_type != "C++":
            raise ValueError(f"FeatureTrackerShared: module_type must be 'C++', got {module_type}")

        # FeatureTrackerShared.clear_cpp_module_callbacks()

        # match pyslam::FeatureDetectAndComputeCallback signature
        def detect_and_compute_cb(image):
            kps, des = FeatureTrackerShared.feature_tracker.detectAndCompute(image)
            # Ensure the descriptor array is properly formatted for pybind11
            if des is not None and len(des) > 0:
                # Make sure it's contiguous, uint8, and has the right shape
                if des.dtype == np.float32:
                    des = np.ascontiguousarray(
                        des, dtype=np.float32
                    )  # Preserve float32 for ROOT_SIFT
                else:
                    des = np.ascontiguousarray(
                        des, dtype=np.uint8
                    )  # Keep uint8 for binary descriptors
                # Ensure it's 2D (N x descriptor_size)
                if des.ndim == 1:
                    des = des.reshape(1, -1)
            # Convert cv2.KeyPoint objects to tuples for C++ compatibility
            return convert_keypoints_to_tuples(kps), des

        # match pyslam::FeatureDetectAndComputeCallback signature
        def detect_and_compute_right_cb(image):
            kps, des = FeatureTrackerShared.feature_tracker_right.detectAndCompute(image)
            # Ensure the descriptor array is properly formatted for pybind11
            if des is not None and len(des) > 0:
                if des.dtype == np.float32:
                    des = np.ascontiguousarray(
                        des, dtype=np.float32
                    )  # Preserve float32 for ROOT_SIFT
                else:
                    des = np.ascontiguousarray(
                        des, dtype=np.uint8
                    )  # Keep uint8 for binary descriptors
                # Ensure it's 2D (N x descriptor_size)
                if des.ndim == 1:
                    des = des.reshape(1, -1)
            # Convert cv2.KeyPoint objects to tuples for C++ compatibility
            return convert_keypoints_to_tuples(kps), des

        # match pyslam::FeatureMatchingCallback signature
        # TODO: use a different dedicated feature matcher for stereo matching
        def stereo_matching_cb(
            image, image_right, des, des_r, kps, kps_r, ratio_test, row_matching, max_disparity
        ):
            results = FeatureTrackerShared.feature_matcher.match(
                image, image_right, des, des_r, kps, kps_r, ratio_test, row_matching, max_disparity
            )
            return results.idxs1, results.idxs2

        # match pyslam::FeatureMatchingCallback signature
        def feature_matching_cb(
            image, image_right, des, des_r, kps, kps_r, ratio_test, row_matching, max_disparity
        ):
            results = FeatureTrackerShared.feature_matcher.match(
                image, image_right, des, des_r, kps, kps_r, ratio_test, row_matching, max_disparity
            )
            return results.idxs1, results.idxs2

        module.FeatureSharedResources.set_feature_detect_and_compute_callback(detect_and_compute_cb)
        module.FeatureSharedResources.set_feature_detect_and_compute_right_callback(
            detect_and_compute_right_cb
        )
        module.FeatureSharedResources.set_stereo_matching_callback(stereo_matching_cb)
        module.FeatureSharedResources.set_feature_matching_callback(feature_matching_cb)

    @staticmethod
    def init_cpp_module_config_parameters():
        """Init the C++ module config parameters"""
        if not FeatureTrackerShared._is_cpp_used:
            return
        FeatureTrackerShared._cpp_module_parameters.kFeatureMatchDefaultRatioTest = (
            FeatureTrackerShared.feature_matcher.ratio_test
        )
        FeatureTrackerShared._cpp_module_parameters.kMaxDescriptorDistance = (
            Parameters.kMaxDescriptorDistance  # this needs to be updated dynamically
        )

    @staticmethod
    def update_cpp_module_dynamic_config_parameters():
        """Update the C++ module config parameters"""
        if not FeatureTrackerShared._is_cpp_used:
            return
        FeatureTrackerShared._cpp_module_parameters.kMaxDescriptorDistance = (
            Parameters.kMaxDescriptorDistance
        )

    @staticmethod
    def clear_cpp_module_callbacks():
        """Clear C++ module callbacks to prevent hanging on exit"""
        try:
            from pyslam.slam.cpp import cpp_module

            cpp_module.FeatureSharedResources.clear_callbacks()
            print("✅ C++ module callbacks cleared")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clear C++ module callbacks: {e}")

    @staticmethod
    def _register_cleanup_handler():
        """Register cleanup handler to be called on program exit"""
        # NOTE: The key insight is that the static callback functions in the C++ module are holding
        # references to Python objects, creating a circular reference that prevents the Python process
        # from exiting. By clearing these callbacks before the process exits, we break the circular
        # reference and allow clean shutdown.
        if not FeatureTrackerShared._is_cpp_initialized:
            atexit.register(FeatureTrackerShared.clear_cpp_module_callbacks)
            FeatureTrackerShared._is_cpp_initialized = True
