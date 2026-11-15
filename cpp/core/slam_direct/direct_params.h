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

#include "common/dictionary.h"

#include <string>

namespace pyslam {

// Klt: pyramidal LK tracks + reprojection-error pose (geometric, descriptor-free).
// Hybrid: Klt, then sparse photometric refinement and photometric fallback.
// SemiDirect: planar-patch photometric pose, per-feature alignment, then reprojection.
enum class DirectTrackingMode { Klt = 0, Hybrid = 1, SemiDirect = 2 };

// gftt: one global Shi-Tomasi extraction. grid_fast: FAST in a regular cell grid.
enum class CornerDetector { Gftt = 0, GridFast = 1 };

// Parameters for both frontends. A ConfigDict may be the DIRECT_SLAM block
// itself, or a camera settings dict that contains that block.
struct DirectSlamParams {
    // Tracking mode: klt uses LK + geometry; hybrid adds photometric refinement/fallback.
    // semi_direct aligns fronto-parallel patches, then refines each feature and runs the depth filter.
    // klt and hybrid require stereo or RGBD. semi_direct also accepts a monocular camera.
    DirectTrackingMode mode = DirectTrackingMode::Klt;

    // Shared: corner detection and live-track replenishment (features/).
    // gftt or grid_fast. grid_fast spreads FAST corners across a cell grid.
    CornerDetector corner_detector = CornerDetector::GridFast;
    // Maximum corners per detection; also the target live-track count when replenishing.
    int num_features = 200;
    // Grid FAST cell count. Reduced, keeping this aspect ratio, when it exceeds num_features.
    int grid_cols = 5;
    int grid_rows = 5;
    // FAST intensity difference. Lower values accept weaker corners.
    int fast_threshold = 10;
    // Keep only FAST corners stronger than their radius-3 neighbors.
    bool fast_nonmax_suppression = true;
    // Refine grid FAST corners with cornerSubPix. Ignored by GFTT, which is already subpixel.
    bool corner_subpix = true;
    // Odd subpixel window side length, in pixels. The point must sit at least half a window inside the image.
    int subpixel_window = 5;
    // Minimum corner response relative to the strongest corner (0 < value <= 1).
    double gftt_quality = 0.01;
    // Minimum distance between detected corners, in full-resolution pixels.
    double gftt_min_distance = 5.0;
    // GFTT corner-response neighborhood side length, in pixels.
    int gftt_block_size = 5;
    // Image-border exclusion margin for detection and LK validity, in pixels.
    int border = 20;

    // Indirect: pyramidal Lucas-Kanade (klt/). Temporal tracking in klt/hybrid, stereo matching in every mode.
    // LK search-window side length, in pixels at each pyramid level (left/right and temporal).
    int lk_patch_size = 11;
    // Highest zero-based LK pyramid level; 3 means levels 0 through 3.
    int lk_max_level = 3;

    // Direct: sparse photometric alignment (direct/direct_aligner), used by hybrid mode.
    // Photometric patch half-width; 1 samples a 3x3 patch at each pyramid level.
    int half_patch_size = 1;
    // Maximum photometric Gauss-Newton iterations per pyramid level.
    int num_iterations = 10;
    // Pose/photometric Gauss-Newton stop threshold for the SE3 update norm (mixed m/rad).
    double eps_convergence = 1e-3;
    // Number of photometric pyramid levels, including the full-resolution image.
    int num_pyramid_levels = 4;
    // Photometric image/intrinsic scale between adjacent levels (0 < value < 1).
    double pyramid_scale = 0.5;
    // Gray-level threshold: per-sample Huber cutoff and per-patch mean absolute inlier error.
    double intensity_inlier_threshold = 25.0;

    // Direct: fronto-parallel patches and inverse-depth seeds (direct/), used by semi_direct.
    // Half-width of the planar patch. 2 samples a 5x5 patch. hybrid keeps half_patch_size.
    int feature_half_patch = 2;
    // Inverse-depth prior and epipolar search range, in meters. Monocular scale follows this range.
    double seed_min_depth = 0.5;
    double seed_max_depth = 80.0;
    // Promote a seed when its inverse-depth variance drops below this.
    // The default is about a one-pixel depth interval at a few meters for a typical focal length.
    double seed_convergence_sigma2 = 1e-4;
    // Cap on immature seeds, and on epipolar searches (re-acquiring lost seed tracks) per accepted frame.
    int max_seeds = 400;
    int max_seed_updates = 100;
    // Monocular two-view initialization: median corner displacement (px) from the reference
    // frame before the essential matrix is tried. The map scale puts the median scene depth at
    // sqrt(seed_min_depth * seed_max_depth).
    double init_min_disparity = 20.0;

    // Tracking status and keyframe insertion.
    // TRACKING_GOOD requires strictly more geometric/accepted inliers than this count.
    int num_features_tracking = 50;
    // Accepted tracking requires strictly more inliers than this count; otherwise LOST.
    int num_features_tracking_bad = 20;
    // Replenish tracks and insert a keyframe when the live-track count falls below this value.
    int num_features_needed_for_keyframe = 80;

    // Geometric pose refinement and stereo validation.
    // Outer reprojection-pose rounds; each round reclassifies geometric outliers.
    int pose_iterations = 4;
    // Maximum Gauss-Newton steps per reprojection-pose round.
    int pose_inner_iterations = 10;
    // Squared pixel-error gate for pose/BA reprojection; its square root is the Huber cutoff.
    double chi2_threshold = 5.991;
    // Native-rig triangulation reprojection tolerance (px); rectified stereo uses a fixed 2 px y gate.
    double stereo_max_error = 2.5;

    // Local bundle adjustment; the archived map can outlive this bounded window.
    // Run synchronous local BA on keyframes; manual bundle_adjustment() remains available.
    bool local_ba = true;
    // Maximum recent keyframes in the BA window (>= 2); the oldest window pose is fixed.
    int num_active_keyframes = 7;
    // Initial robust LM iteration cap; refinement uses max(1, ba_iterations / 2) iterations.
    int ba_iterations = 10;
    // Force a keyframe after this input-frame ID gap when automatic BA is enabled.
    int max_keyframe_interval = 10;
    // RGBD depth sigma (m) = value * max(1, z*z); larger values weaken depth constraints.
    double ba_depth_sigma = 0.02;

    // With monocular true, keys of a "monocular" sub-block (nested dict or flat
    // "monocular.<key>") override the base values. Stereo/RGBD ignore that block.
    static DirectSlamParams from_config(const ConfigDict &config, bool monocular = false);
    std::string mode_name() const;
};

} // namespace pyslam
