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

// Monocular input usually needs other values than stereo/RGBD from the same settings
// file (e.g. more corners and seeds, since only converged seeds create landmarks).
ConfigDict apply_monocular_overrides(ConfigDict block) {
    const ConfigDict nested = block.get("monocular", ConfigDict());
    ConfigDict overrides;
    for (const auto &entry : nested) overrides[entry.first] = entry.second;
    const std::string prefix = "monocular.";
    for (const auto &entry : block) {
        if (entry.first.rfind(prefix, 0) == 0) overrides[entry.first.substr(prefix.size())] = entry.second;
    }
    for (const auto &entry : overrides) block[entry.first] = entry.second;
    return block;
}

} // namespace

DirectSlamParams DirectSlamParams::from_config(const ConfigDict &config, bool monocular) {
    ConfigDict block = parameter_block(config);
    if (monocular) block = apply_monocular_overrides(std::move(block));
    DirectSlamParams params;
    const std::string mode = lower_copy(block.get<std::string>("mode", "klt"));
    // "direct" remains a deprecated alias of hybrid for older settings files.
    if (mode == "klt") {
        params.mode = DirectTrackingMode::Klt;
    } else if (mode == "semi_direct" || mode == "semidirect") {
        params.mode = DirectTrackingMode::SemiDirect;
    } else if (mode == "hybrid" || mode == "direct") {
        params.mode = DirectTrackingMode::Hybrid;
    } else {
        throw std::invalid_argument(
            "DIRECT_SLAM.mode must be 'klt', 'hybrid', or 'semi_direct', got '" + mode + "'");
    }
    const std::string detector = lower_copy(block.get<std::string>("corner_detector", "grid_fast"));
    if (detector == "gftt") {
        params.corner_detector = CornerDetector::Gftt;
    } else if (detector == "grid_fast" || detector == "fast_grid") {
        params.corner_detector = CornerDetector::GridFast;
    } else {
        throw std::invalid_argument(
            "DIRECT_SLAM.corner_detector must be 'gftt' or 'grid_fast', got '" + detector + "'");
    }

    // Missing keys retain the defaults documented in DirectSlamParams.
    params.num_features = block.get("num_features", params.num_features);
    params.grid_cols = block.get("grid_cols", params.grid_cols);
    params.grid_rows = block.get("grid_rows", params.grid_rows);
    params.fast_threshold = block.get("fast_threshold", params.fast_threshold);
    params.fast_nonmax_suppression =
        block.get("fast_nonmax_suppression", params.fast_nonmax_suppression);
    params.corner_subpix = block.get("corner_subpix", params.corner_subpix);
    params.subpixel_window = block.get("subpixel_window", params.subpixel_window);
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
    params.feature_half_patch = block.get("feature_half_patch", params.feature_half_patch);
    params.seed_min_depth = block.get("seed_min_depth", params.seed_min_depth);
    params.seed_max_depth = block.get("seed_max_depth", params.seed_max_depth);
    params.seed_convergence_sigma2 =
        block.get("seed_convergence_sigma2", params.seed_convergence_sigma2);
    params.max_seeds = block.get("max_seeds", params.max_seeds);
    params.max_seed_updates = block.get("max_seed_updates", params.max_seed_updates);
    params.init_min_disparity = block.get("init_min_disparity", params.init_min_disparity);
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
        params.grid_cols < 1 || params.grid_rows < 1 || params.fast_threshold < 1 ||
        params.fast_threshold > 255 ||
        (params.corner_subpix && (params.subpixel_window < 3 || params.subpixel_window % 2 == 0)) ||
        params.num_features_tracking_bad < 0 ||
        params.num_features_tracking < params.num_features_tracking_bad ||
        params.num_features_needed_for_keyframe < 1 || params.feature_half_patch < 1 ||
        !(params.seed_min_depth > 0.0) || !(params.seed_max_depth > params.seed_min_depth) ||
        !(params.seed_convergence_sigma2 > 0.0) || params.max_seeds < 1 || params.max_seed_updates < 1 ||
        !(params.init_min_disparity > 0.0)) {
        throw std::invalid_argument("DIRECT_SLAM parameters are out of range");
    }
    return params;
}

std::string DirectSlamParams::mode_name() const {
    switch (mode) {
    case DirectTrackingMode::Hybrid:
        return "hybrid";
    case DirectTrackingMode::SemiDirect:
        return "semi_direct";
    case DirectTrackingMode::Klt:
        return "klt";
    }
    return "klt";
}

} // namespace pyslam
