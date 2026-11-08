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

#include "slam_direct/direct_params.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace pyslam {
namespace {

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

// Accept nested Python settings, flat OpenCV YAML keys, or an already-extracted block.
// Nested settings take precedence; unrelated camera keys are ignored by the parser.
const ConfigDict parameter_block(const ConfigDict &config) {
    if (config.has("DIRECT_SLAM")) {
        return config.get("DIRECT_SLAM", ConfigDict());
    }
    ConfigDict prefixed;
    const std::string prefix = "DIRECT_SLAM.";
    for (const auto &entry : config) {
        if (entry.first.rfind(prefix, 0) == 0) {
            prefixed[entry.first.substr(prefix.size())] = entry.second;
        }
    }
    if (!prefixed.empty()) {
        return prefixed;
    }
    return config;
}

} // namespace

DirectSlamParams DirectSlamParams::from_config(const ConfigDict &config) {
    ConfigDict block = parameter_block(config);
    DirectSlamParams params;
    const std::string mode = lower_copy(block.get<std::string>("mode", "semi_direct"));
    if (mode == "semi_direct" || mode == "semidirect") {
        params.mode = DirectTrackingMode::SemiDirect;
    } else if (mode == "direct") {
        params.mode = DirectTrackingMode::Direct;
    } else {
        throw std::invalid_argument("DIRECT_SLAM.mode must be 'semi_direct' or 'direct', got '" +
                                    mode + "'");
    }

    // Missing keys retain the defaults documented in DirectSlamParams.
    params.num_features = block.get("num_features", params.num_features);
    params.gftt_quality = block.get("gftt_quality", params.gftt_quality);
    params.gftt_min_distance = block.get("gftt_min_distance", params.gftt_min_distance);
    params.gftt_block_size = block.get("gftt_block_size", params.gftt_block_size);
    params.border = block.get("border", params.border);
    params.lk_patch_size = block.get("lk_patch_size", params.lk_patch_size);
    params.lk_max_level = block.get("lk_max_level", params.lk_max_level);
    params.half_patch_size = block.get("half_patch_size", params.half_patch_size);
    params.num_iterations = block.get("num_iterations", params.num_iterations);
    params.eps_convergence = block.get("eps_convergence", params.eps_convergence);
    params.num_pyramid_levels = block.get("num_pyramid_levels", params.num_pyramid_levels);
    params.pyramid_scale = block.get("pyramid_scale", params.pyramid_scale);
    params.intensity_inlier_threshold =
        block.get("intensity_inlier_threshold", params.intensity_inlier_threshold);
    params.num_features_tracking = block.get("num_features_tracking", params.num_features_tracking);
    params.num_features_tracking_bad =
        block.get("num_features_tracking_bad", params.num_features_tracking_bad);
    params.num_features_needed_for_keyframe =
        block.get("num_features_needed_for_keyframe", params.num_features_needed_for_keyframe);
    params.pose_iterations = block.get("pose_iterations", params.pose_iterations);
    params.pose_inner_iterations = block.get("pose_inner_iterations", params.pose_inner_iterations);
    params.chi2_threshold = block.get("chi2_threshold", params.chi2_threshold);
    params.stereo_max_error = block.get("stereo_max_error", params.stereo_max_error);

    params.local_ba = block.get("local_ba", params.local_ba);
    params.num_active_keyframes = block.get("num_active_keyframes", params.num_active_keyframes);
    params.ba_iterations = block.get("ba_iterations", params.ba_iterations);
    params.max_keyframe_interval = block.get("max_keyframe_interval", params.max_keyframe_interval);
    params.ba_depth_sigma = block.get("ba_depth_sigma", params.ba_depth_sigma);

    // Reject invalid solver/image-window values before they reach Eigen or OpenCV.
    if (params.num_active_keyframes < 2 || params.ba_iterations < 1 ||
        params.max_keyframe_interval < 1 || !(params.ba_depth_sigma > 0.0) || params.num_features < 1 || params.half_patch_size < 0 || params.num_iterations < 1 ||
        params.num_pyramid_levels < 1 || !(params.pyramid_scale > 0.0 && params.pyramid_scale < 1.0) ||
        params.lk_patch_size < 3 || params.lk_max_level < 0 || params.border < 0 ||
        !(params.gftt_quality > 0.0 && params.gftt_quality <= 1.0) ||
        !(params.gftt_min_distance >= 0.0) || params.gftt_block_size < 2 ||
        !(params.eps_convergence > 0.0) || !(params.intensity_inlier_threshold > 0.0) ||
        params.pose_iterations < 1 || params.pose_inner_iterations < 1 ||
        !(params.chi2_threshold > 0.0) || !(params.stereo_max_error > 0.0) ||
        params.num_features_tracking_bad < 0 ||
        params.num_features_tracking < params.num_features_tracking_bad ||
        params.num_features_needed_for_keyframe < 1) {
        throw std::invalid_argument("DIRECT_SLAM parameters are out of range");
    }
    return params;
}

std::string DirectSlamParams::mode_name() const {
    return mode == DirectTrackingMode::Direct ? "direct" : "semi_direct";
}

} // namespace pyslam
