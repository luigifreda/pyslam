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

#include "slam_direct/direct/depth_filter.h"

#include "slam_direct/direct/direct_aligner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pyslam {
namespace {

constexpr int kMaxSeedFailures = 20;

float sample_gray(const cv::Mat &img, float x, float y) {
    if (img.cols < 2 || img.rows < 2) {
        return img.empty() ? 0.0f : static_cast<float>(img.at<uchar>(0, 0));
    }
    x = std::clamp(x, 0.0f, static_cast<float>(img.cols - 2));
    y = std::clamp(y, 0.0f, static_cast<float>(img.rows - 2));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float i00 = static_cast<float>(img.at<uchar>(y0, x0));
    const float i10 = static_cast<float>(img.at<uchar>(y0, x0 + 1));
    const float i01 = static_cast<float>(img.at<uchar>(y0 + 1, x0));
    const float i11 = static_cast<float>(img.at<uchar>(y0 + 1, x0 + 1));
    return (1.0f - fx) * (1.0f - fy) * i00 + fx * (1.0f - fy) * i10 + (1.0f - fx) * fy * i01 +
           fx * fy * i11;
}

double normal_density(double x, double mean, double variance) {
    if (!(variance > 0.0)) {
        return 0.0;
    }
    const double scale = std::sqrt(variance);
    const double delta = (x - mean) / scale;
    return std::exp(-0.5 * delta * delta) / (scale * std::sqrt(2.0 * 3.14159265358979323846));
}

// Project every pixel of the constant-depth patch and score it against the reference window.
bool patch_zero_mean_ssd(const Camera &camera, const Eigen::Isometry3d &T_cur_ref, const cv::Mat &img_ref,
                         const cv::Mat &img_cur, const Vec2d &pixel_ref, double depth, int half, double &score) {
    const int side = 2 * half + 1;
    std::vector<float> ref_values;
    std::vector<float> cur_values;
    ref_values.reserve(static_cast<size_t>(side * side));
    cur_values.reserve(static_cast<size_t>(side * side));
    for (int py = -half; py <= half; ++py) {
        for (int px = -half; px <= half; ++px) {
            const Vec2d ref_px = pixel_ref + Vec2d(px, py);
            if (ref_px.x() < 1.0 || ref_px.y() < 1.0 || ref_px.x() >= img_ref.cols - 2 ||
                ref_px.y() >= img_ref.rows - 2) {
                return false;
            }
            Vec3d point_ref = Vec3d::Zero();
            if (!lift_reference_point(camera, ref_px, depth, point_ref)) {
                return false;
            }
            const Vec3d point_cur = T_cur_ref * point_ref;
            if (!(point_cur.z() > 1e-8)) {
                return false;
            }
            const Vec2d cur_px = camera.project_point(point_cur).first;
            if (!cur_px.allFinite() || cur_px.x() < 1.0 || cur_px.y() < 1.0 || cur_px.x() >= img_cur.cols - 2 ||
                cur_px.y() >= img_cur.rows - 2) {
                return false;
            }
            ref_values.push_back(sample_gray(img_ref, static_cast<float>(ref_px.x()), static_cast<float>(ref_px.y())));
            cur_values.push_back(sample_gray(img_cur, static_cast<float>(cur_px.x()), static_cast<float>(cur_px.y())));
        }
    }
    if (ref_values.empty()) {
        return false;
    }
    double ref_mean = 0.0;
    double cur_mean = 0.0;
    for (size_t i = 0; i < ref_values.size(); ++i) {
        ref_mean += ref_values[i];
        cur_mean += cur_values[i];
    }
    ref_mean /= static_cast<double>(ref_values.size());
    cur_mean /= static_cast<double>(cur_values.size());
    double sum = 0.0;
    for (size_t i = 0; i < ref_values.size(); ++i) {
        const double delta = (ref_values[i] - ref_mean) - (cur_values[i] - cur_mean);
        sum += delta * delta;
    }
    score = sum;
    return std::isfinite(score);
}

// SVO: tau_inverse = 0.5 * (1 / max(eps, z - tau) - 1 / (z + tau)).
double inverse_depth_sigma(double depth, double depth_uncertainty) {
    if (!(depth > 0.0) || !(depth_uncertainty > 0.0) || !std::isfinite(depth_uncertainty)) {
        return 0.0;
    }
    return 0.5 * (1.0 / std::max(1e-7, depth - depth_uncertainty) - 1.0 / (depth + depth_uncertainty));
}

} // namespace

InverseDepthBelief make_inverse_depth_prior(double near_m, double far_m, double mean_depth_m) {
    const double near_inv = 1.0 / near_m;
    const double far_inv = 1.0 / far_m;
    const double low = std::min(near_inv, far_inv);
    const double high = std::max(near_inv, far_inv);
    InverseDepthBelief belief;
    belief.mu = mean_depth_m > 0.0 && std::isfinite(mean_depth_m) ? std::clamp(1.0 / mean_depth_m, low, high)
                                                                 : 0.5 * (low + high);
    const double span = high - low;
    belief.sigma2 = (span / 6.0) * (span / 6.0);
    belief.a = 10.0;
    belief.b = 10.0;
    return belief;
}

bool fuse_inverse_depth(InverseDepthBelief &belief, double measured_inverse, double measured_variance,
                        double near_m, double far_m) {
    if (!(belief.sigma2 > 0.0) || !(measured_variance > 0.0) || !std::isfinite(measured_inverse) ||
        !(belief.a > 0.0) || !(belief.b > 0.0) || !(near_m > 0.0) || !(far_m > near_m)) {
        return false;
    }
    const double low = 1.0 / far_m;
    const double high = 1.0 / near_m;
    const double span = high - low;
    if (!(span > 0.0)) {
        return false;
    }
    const double combined = belief.sigma2 + measured_variance;
    const double fused_variance = 1.0 / (1.0 / belief.sigma2 + 1.0 / measured_variance);
    const double fused_mean = fused_variance * (belief.mu / belief.sigma2 + measured_inverse / measured_variance);
    const double inlier_ratio = belief.a / (belief.a + belief.b);
    const double inlier_density = inlier_ratio * normal_density(measured_inverse, belief.mu, combined);
    const double outlier_density = (1.0 - inlier_ratio) / span;
    const double total = inlier_density + outlier_density;
    if (!(total > 0.0) || !std::isfinite(fused_mean) || !(fused_variance > 0.0)) {
        return false;
    }
    const double inlier_weight = inlier_density / total;
    const double outlier_weight = outlier_density / total;
    const double mean = inlier_weight * fused_mean + outlier_weight * belief.mu;
    const double variance = inlier_weight * (fused_variance + fused_mean * fused_mean) +
                            outlier_weight * (belief.sigma2 + belief.mu * belief.mu) - mean * mean;
    const double count = belief.a + belief.b;
    const double expected =
        inlier_weight * (belief.a + 1.0) / (count + 1.0) + outlier_weight * belief.a / (count + 1.0);
    const double second = inlier_weight * (belief.a + 1.0) * (belief.a + 2.0) / ((count + 1.0) * (count + 2.0)) +
                          outlier_weight * belief.a * (belief.a + 1.0) / ((count + 1.0) * (count + 2.0));
    if (!(variance > 0.0) || !std::isfinite(variance) || !(expected > 0.0) || !(expected < 1.0)) {
        return false;
    }
    const double moment_gap = expected - second / expected;
    if (!(std::abs(moment_gap) > 1e-12)) {
        return false;
    }
    const double a = (second - expected) / moment_gap;
    const double b = a * (1.0 - expected) / expected;
    if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    belief.mu = mean;
    belief.sigma2 = variance;
    belief.a = a;
    belief.b = b;
    return true;
}

EpipolarMatch match_seed_on_epipolar_line(const Camera &camera, const DirectSlamParams &params,
                                          const cv::Mat &img_ref, const cv::Mat &img_cur, const Vec2d &pixel_ref,
                                          double mu, double sigma2, const Eigen::Isometry3d &T_cur_ref) {
    EpipolarMatch match;
    if (!(sigma2 > 0.0) || img_ref.empty() || img_cur.empty()) {
        return match;
    }
    // The prior is a ±3σ interval, so the first search has to cover that whole range.
    // Later updates shrink sigma2 and the segment shrinks with it.
    const double radius = 3.0 * std::sqrt(sigma2);
    const double low = std::max(1.0 / params.seed_max_depth, mu - radius);
    const double high = std::min(1.0 / params.seed_min_depth, mu + radius);
    if (!(high > low)) {
        return match;
    }
    Vec3d near_ref = Vec3d::Zero();
    Vec3d far_ref = Vec3d::Zero();
    if (!lift_reference_point(camera, pixel_ref, 1.0 / high, near_ref) ||
        !lift_reference_point(camera, pixel_ref, 1.0 / low, far_ref)) {
        return match;
    }
    const Vec2d near_px = camera.project_point(T_cur_ref * near_ref).first;
    const Vec2d far_px = camera.project_point(T_cur_ref * far_ref).first;
    if (!near_px.allFinite() || !far_px.allFinite()) {
        return match;
    }
    const double segment = (near_px - far_px).norm();
    if (segment < 1.0) {
        return match;
    }
    match.searched = true;
    const int samples = std::max(2, static_cast<int>(std::ceil(segment)));
    const int side = 2 * params.feature_half_patch + 1;
    const double gate =
        params.intensity_inlier_threshold * params.intensity_inlier_threshold * static_cast<double>(side * side);
    std::vector<double> inverses(static_cast<size_t>(samples), 0.0);
    std::vector<double> scores(static_cast<size_t>(samples), std::numeric_limits<double>::infinity());
    const double step = (high - low) / static_cast<double>(samples - 1);
    for (int i = 0; i < samples; ++i) {
        const double inverse = low + step * static_cast<double>(i);
        inverses[static_cast<size_t>(i)] = inverse;
        double score = 0.0;
        if (patch_zero_mean_ssd(camera, T_cur_ref, img_ref, img_cur, pixel_ref, 1.0 / inverse,
                                params.feature_half_patch, score)) {
            scores[static_cast<size_t>(i)] = score;
        }
    }
    int best_index = -1;
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < samples; ++i) {
        if (scores[static_cast<size_t>(i)] < best) {
            best = scores[static_cast<size_t>(i)];
            best_index = i;
        }
    }
    double second = std::numeric_limits<double>::infinity();
    for (int i = 0; i < samples; ++i) {
        if (std::abs(i - best_index) <= 1) {
            continue;
        }
        second = std::min(second, scores[static_cast<size_t>(i)]);
    }
    // Adjacent samples sit in the same valley. A unique minimum is at least twice
    // the next score, and that next score has to clear a one-gray-level floor.
    const bool unique = std::isfinite(second) && second > std::max(2.0 * best, best + 25.0);
    if (best_index < 0 || !(best < gate) || !unique) {
        return match;
    }
    // Samples are at most one pixel apart. A parabola through the minimum and its two
    // neighbors gives the subpixel position along the segment (SVO refines it in 1D too).
    double inverse = inverses[static_cast<size_t>(best_index)];
    if (best_index > 0 && best_index + 1 < samples) {
        const double s_minus = scores[static_cast<size_t>(best_index - 1)];
        const double s_plus = scores[static_cast<size_t>(best_index + 1)];
        const double curvature = s_minus - 2.0 * best + s_plus;
        if (std::isfinite(s_minus) && std::isfinite(s_plus) && curvature > 0.0) {
            inverse += std::clamp(0.5 * (s_minus - s_plus) / curvature, -0.5, 0.5) * step;
        }
    }
    const double depth = 1.0 / inverse;
    // SVO measurement model: a one-pixel angular error on the current ray.
    const double px_error_angle = 2.0 * std::atan(1.0 / (2.0 * camera.fx));
    const double tau = depth_uncertainty_one_pixel(camera, T_cur_ref, pixel_ref, depth, px_error_angle);
    const double sigma = inverse_depth_sigma(depth, tau);
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        // Degenerate triangulation carries no depth information: skip, do not penalize.
        match.searched = false;
        return match;
    }
    match.matched = true;
    match.depth = depth;
    match.inverse_variance = sigma * sigma;
    // Pixel baseline between the match and the same reference ray at infinity.
    // This stays large after the mean has moved onto the measurement.
    Vec3d matched_ref = Vec3d::Zero();
    Vec3d bearing = Vec3d::Zero();
    if (lift_reference_point(camera, pixel_ref, depth, matched_ref) &&
        camera.unproject_bearing(pixel_ref, bearing, nullptr) && bearing.z() > 1e-8) {
        const Vec3d far_cur = T_cur_ref.linear() * bearing;
        const Vec2d matched_px = camera.project_point(T_cur_ref * matched_ref).first;
        const Vec2d far_px = far_cur.z() > 1e-8 ? camera.project_point(far_cur).first : Vec2d::Zero();
        if (matched_px.allFinite() && far_px.allFinite() && far_cur.z() > 1e-8) {
            match.parallax_px = (matched_px - far_px).norm();
        }
    }
    return match;
}

SeedObservation observe_seed(DepthSeed &seed, const EpipolarMatch &match, double near_m, double far_m) {
    if (!match.searched) {
        return SeedObservation::Skipped;
    }
    if (!match.matched) {
        // SVO: a searched segment without a match is outlier evidence for this seed.
        seed.belief.b += 1.0;
        ++seed.failures;
        return SeedObservation::Failed;
    }
    InverseDepthBelief updated = seed.belief;
    if (!fuse_inverse_depth(updated, 1.0 / match.depth, match.inverse_variance, near_m, far_m)) {
        ++seed.failures;
        return SeedObservation::Failed;
    }
    seed.belief = updated;
    seed.parallax_px = match.parallax_px;
    return SeedObservation::Updated;
}

bool seed_expired(const DepthSeed &seed, double prior_sigma2) {
    if (seed.failures >= kMaxSeedFailures) {
        return true;
    }
    const double total = seed.belief.a + seed.belief.b;
    if (total > 20.0 && seed.belief.b / total > 0.8) {
        return true;
    }
    return seed.belief.sigma2 > prior_sigma2 * 4.0;
}

double depth_uncertainty_one_pixel(const Camera &camera, const Eigen::Isometry3d &T_cur_ref,
                                   const Vec2d &pixel_ref, double depth, double px_error_angle) {
    Vec3d f = Vec3d::Zero();
    if (!(depth > 0.0) || !camera.unproject_bearing(pixel_ref, f, nullptr) || !f.allFinite()) {
        return -1.0;
    }
    f.normalize();
    if (!(f.z() > 1e-8)) {
        return -1.0;
    }
    // rpg_svo DepthFilter::computeTau, with t the current camera center in the reference
    // frame and the distance along the unit ray f. The result is converted back to z.
    const Vec3d t = T_cur_ref.inverse().translation();
    const double t_norm = t.norm();
    const double range = depth / f.z();
    const Vec3d a = f * range - t;
    const double a_norm = a.norm();
    if (!(t_norm > 1e-12) || !(a_norm > 1e-12)) {
        return -1.0;
    }
    const double alpha = std::acos(std::clamp(f.dot(t) / t_norm, -1.0, 1.0));
    const double beta = std::acos(std::clamp(a.dot(-t) / (t_norm * a_norm), -1.0, 1.0));
    const double beta_plus = beta + px_error_angle;
    const double gamma_plus = 3.14159265358979323846 - alpha - beta_plus;
    if (!(gamma_plus > 1e-12)) {
        return -1.0;
    }
    const double range_plus = t_norm * std::sin(beta_plus) / std::sin(gamma_plus);
    const double tau = (range_plus - range) * f.z();
    return std::isfinite(tau) && tau > 0.0 ? tau : -1.0;
}

} // namespace pyslam
