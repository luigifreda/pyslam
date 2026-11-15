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

#include "camera/camera.h"
#include "slam_direct/direct_params.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace pyslam {

// SVO depth filter (Forster et al., ICRA 2014, Sec. IV-B), which uses the parametric
// model of Vogiatzis and Hernandez (IVC 2011): inverse depth is a Gaussian inlier
// hypothesis mixed with a uniform outlier hypothesis over [1/far, 1/near], and the
// inlier ratio has a Beta(a, b) posterior. The update matches rpg_svo updateSeed().
struct InverseDepthBelief {
    double mu = 0.0;
    double sigma2 = 1.0;
    double a = 10.0;
    double b = 10.0;
};

// One immature feature. pixel is on keyframe_id. parallax_px is the last accepted
// match displacement in the current image.
struct DepthSeed {
    int keyframe_id = -1;
    Vec2d pixel = Vec2d::Zero();
    InverseDepthBelief belief;
    int failures = 0;
    double parallax_px = 0.0;
    // Frame-to-frame LK position in the last accepted image. While tracked, the seed is
    // measured from this correspondence; once lost, by the epipolar search.
    Vec2d pixel_cur = Vec2d::Zero();
    bool tracked = false;
};

// Prior whose ±3σ inverse-depth interval covers [1/far, 1/near] (SVO: sigma2 = z_range^2 / 36).
// SVO centers a new seed on the keyframe's scene depth; pass it as mean_depth_m when known
// (clamped to [near, far]). Without it, the mean is the middle of the inverse-depth range.
InverseDepthBelief make_inverse_depth_prior(double near_m, double far_m, double mean_depth_m = 0.0);

// One Gaussian-uniform update with a measured inverse depth and its variance tau^2.
// On failure the belief is left unchanged.
bool fuse_inverse_depth(InverseDepthBelief &belief, double measured_inverse, double measured_variance,
                        double near_m, double far_m);

struct EpipolarMatch {
    bool searched = false; // False when the uncertainty segment is shorter than one pixel.
    bool matched = false;
    double depth = 0.0;            // Camera-frame z of the match in the reference keyframe.
    double inverse_variance = 0.0; // SVO tau^2: one-pixel angular error, in inverse depth.
    double parallax_px = 0.0;
};

// Project the seed's planar patch along its inverse-depth segment, keep one unique
// zero-mean SSD minimum and refine it to subpixel along the segment. The measurement
// variance follows SVO computeTau(): the depth change caused by a one-pixel error of
// the ray angle in the current image, by the law of sines.
EpipolarMatch match_seed_on_epipolar_line(const Camera &camera, const DirectSlamParams &params,
                                          const cv::Mat &img_ref, const cv::Mat &img_cur,
                                          const Vec2d &pixel_ref, double mu, double sigma2,
                                          const Eigen::Isometry3d &T_cur_ref);

// Depth measurement from a tracked correspondence: triangulate pixel_ref (reference keyframe)
// with pixel_cur. The variance is the same one-pixel tau as the epipolar search. A match that
// does not reproject within the chi2 gate, or falls outside the seed range, is a failed match;
// a ray pair without usable parallax is not searched.
EpipolarMatch measure_seed_from_track(const Camera &camera, const DirectSlamParams &params,
                                      const Vec2d &pixel_ref, const Vec2d &pixel_cur,
                                      const Eigen::Isometry3d &T_cur_ref);

enum class SeedObservation { Skipped, Updated, Failed };

// Skipped leaves the seed untouched. Failed (no match on a searched segment) counts
// as outlier evidence, as in SVO: b += 1 and the failure count grows. Updated fuses the
// match into the belief.
SeedObservation observe_seed(DepthSeed &seed, const EpipolarMatch &match, double near_m, double far_m);

// Too many failed searches, a high outlier ratio, or a variance that grew past the prior.
// The outlier gate is the expected inlier ratio a / (a + b) < 0.2 after 20 votes.
bool seed_expired(const DepthSeed &seed, double prior_sigma2);

// SVO computeTau(): depth (camera z) uncertainty for a point at depth z on the reference ray
// through pixel_ref, given the current camera, for an angular error px_error_angle.
// Returns a negative value when the geometry is degenerate (no baseline, ray at infinity).
double depth_uncertainty_one_pixel(const Camera &camera, const Eigen::Isometry3d &T_cur_ref,
                                   const Vec2d &pixel_ref, double depth, double px_error_angle);

} // namespace pyslam
