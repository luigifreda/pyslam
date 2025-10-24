/*
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
 */

#pragma once

#include "eigen_aliases.h"
#include "semantic_types.h"
#include "utils/descriptor_helpers.h"
#include <opencv2/opencv.hpp>

#include <vector>

namespace pyslam {

// pybind11 casting
// cv::KeyPoint <-> (pt.x, pt.y, size, angle, response, octave)
using KeyPointTuple = std::tuple<float, float, float, float, float, int>;

// (kps, des)
using FeatureDetectAndComputeOutput = std::pair<std::vector<KeyPointTuple>, cv::Mat>;

// (img) -> (kps, des)
using FeatureDetectAndComputeCallback =
    std::function<FeatureDetectAndComputeOutput(const cv::Mat &)>;

// (idxs1, idxs2)
using FeatureMatchingOutput = std::pair<std::vector<int>, std::vector<int>>;

// (img, img_right, des, des_r, kps, kps_r, ratio_test, row_matching, max_disparity)
// -> (idxs1, idxs2)
using FeatureMatchingCallback = std::function<FeatureMatchingOutput(
    const cv::Mat &, const cv::Mat &, const cv::Mat &, const cv::Mat &, MatNx2Ref<float>,
    MatNx2Ref<float>, float, bool, float)>;

// Feature shared info
// This class is used to share information between the Python and C++ code.
// At present, it contains part of the data that are stored in the python class
// FeatureTrackerShared.
class FeatureSharedResources {
  public:
    static float scale_factor;
    static float inv_scale_factor;
    static float log_scale_factor;

    static std::vector<float> scale_factors;
    static std::vector<float> inv_scale_factors;

    static std::vector<float> level_sigmas;
    static std::vector<float> level_sigmas2;
    static std::vector<float> inv_level_sigmas2;

    static int num_levels;
    static int num_features;

    static int detector_type;
    static int descriptor_type;
    static NormType norm_type;

    static bool oriented_features;

    static float feature_match_ratio_test; // ratio test used by all feature matchers

    static SemanticFeatureType
        semantic_feature_type; // semantic feature type (LABEL, PROBABILITY_VECTOR, FEATURE_VECTOR)

  public:
    // Python callbacks for feature management
    static FeatureDetectAndComputeCallback feature_detect_and_compute_callback;

    static FeatureDetectAndComputeCallback feature_detect_and_compute_right_callback;

    static FeatureMatchingCallback stereo_matching_callback; // dedicated to stereo matching

    static FeatureMatchingCallback
        feature_matching_callback; // dedicated to generic feature matching

  public:
    // Methods to set the Python callbacks

    static void set_feature_detect_and_compute_callback(
        const FeatureDetectAndComputeCallback &detect_compute_cb) {
        feature_detect_and_compute_callback = detect_compute_cb;
    }

    static void set_feature_detect_and_compute_right_callback(
        const FeatureDetectAndComputeCallback &detect_compute_cb) {
        feature_detect_and_compute_right_callback = detect_compute_cb;
    }

    static void set_stereo_matching_callback(const FeatureMatchingCallback &stereo_matching_cb) {
        stereo_matching_callback = stereo_matching_cb;
    }

    static void set_feature_matching_callback(const FeatureMatchingCallback &feature_matching_cb) {
        feature_matching_callback = feature_matching_cb;
    }

    static void clear_callbacks() {
        // NOTE : The key insight is that the static callback functions in this C++ module are
        // holding references to Python objects, creating a circular reference that prevents the
        // Python process from exiting.By clearing these callbacks before the process exits, we
        // break the circular reference and allow clean shutdown.
        // This function is automatically called when the Python process exits by
        // FeatureTrackerShared.clear_cpp_module_callbacks() that is registered on exit by
        // FeatureTrackerShared._register_cleanup_handler()
        feature_detect_and_compute_callback = nullptr;
        feature_detect_and_compute_right_callback = nullptr;
        stereo_matching_callback = nullptr;
        feature_matching_callback = nullptr;
    }
};

} // namespace pyslam