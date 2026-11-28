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

#include "slam/direct/photometric/depth_filter.h"

#include "slam/direct/photometric/direct_aligner.h"
#include "slam/direct/core/image_utils.h"
#include "slam/direct/geometry/ray_geometry.h"
#include "slam/direct/geometry/triangulation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pyslam {
namespace {

constexpr int kMaxSeedFailures = 20;
// Minimum ray angle of a tracked correspondence, in one-pixel angular errors.
constexpr double kMinTrackRayAnglePx = 3.0;

double normal_density(double x, double mean, double variance) {
    if (!(variance > 0.0)) {
        return 0.0;
    }
    const double scale = std::sqrt(variance);
    const double delta = (x - mean) / scale;
    return std::exp(-0.5 * delta * delta) / (scale * std::sqrt(2.0 * 3.14159265358979323846));
}

// The reference window of one epipolar search: the pixels, bearings and intensities that do
// not depend on the sampled depth, computed once instead of once per sample.
class ReferencePatch {
  public:
    ReferencePatch(const Camera &camera, const cv::Mat &img_ref, const Vec2d &pixel_ref, int half) {
        const int side = 2 * half + 1;
        bearings_.reserve(static_cast<size_t>(side * side));
        ref_values_.reserve(static_cast<size_t>(side * side));
        constexpr double kMinBearingZ = 1e-4; // Same gate as lift_reference_point().
        for (int py = -half; py <= half; ++py) {
            for (int px = -half; px <= half; ++px) {
                const Vec2d ref_px = pixel_ref + Vec2d(px, py);
                if (ref_px.x() < 1.0 || ref_px.y() < 1.0 || ref_px.x() >= img_ref.cols - 2 ||
                    ref_px.y() >= img_ref.rows - 2) {
                    return; // Leaves the patch invalid: every sample would fail.
                }
                Vec3d bearing;
                if (!camera.unproject_bearing(ref_px, bearing, nullptr) || !bearing.allFinite() ||
                    !(bearing.z() > kMinBearingZ)) {
                    return;
                }
                bearings_.push_back(bearing);
                ref_values_.push_back(
                    sample_gray(img_ref, static_cast<float>(ref_px.x()), static_cast<float>(ref_px.y())));
            }
        }
        for (const float value : ref_values_) {
            ref_mean_ += value;
        }
        ref_mean_ /= static_cast<double>(ref_values_.size());
        cur_values_.resize(ref_values_.size());
        valid_ = !ref_values_.empty();
    }

    bool valid() const { return valid_; }

    // Project every pixel of the constant-depth patch and score it against the reference window.
    // Same arithmetic, in the same order, as lifting each pixel with lift_reference_point().
    bool zero_mean_ssd(const Camera &camera, const Eigen::Isometry3d &T_cur_ref, const cv::Mat &img_cur,
                       double depth, double &score) {
        if (!valid_ || !(depth > 0.0) || !std::isfinite(depth)) {
            return false;
        }
        for (size_t i = 0; i < bearings_.size(); ++i) {
            const Vec3d &bearing = bearings_[i];
            const Vec3d point_ref = bearing * (depth / bearing.z());
            if (!point_ref.allFinite()) {
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
            cur_values_[i] = sample_gray(img_cur, static_cast<float>(cur_px.x()), static_cast<float>(cur_px.y()));
        }
        double cur_mean = 0.0;
        for (const float value : cur_values_) {
            cur_mean += value;
        }
        cur_mean /= static_cast<double>(cur_values_.size());
        double sum = 0.0;
        for (size_t i = 0; i < ref_values_.size(); ++i) {
            const double delta = (ref_values_[i] - ref_mean_) - (cur_values_[i] - cur_mean);
            sum += delta * delta;
        }
        score = sum;
        return std::isfinite(score);
    }

  private:
    std::vector<Vec3d> bearings_;
    std::vector<float> ref_values_;
    std::vector<float> cur_values_;
    double ref_mean_ = 0.0;
    bool valid_ = false;
};

// Coarse probes used to find a visible part of the search range when neither end is visible.
constexpr int kVisibleRangeProbes = 32;
constexpr int kVisibleRangeBisections = 30;
// Upper bound on epipolar samples, per pixel of image width plus height. One-pixel spacing
// near the current camera can still need many samples; this caps the cost of one search.
constexpr int kMaxSamplesPerImageSide = 2;

// Narrow [low, high] (inverse depths on the reference ray through pixel_ref) to the part
// that projects in front of the current camera and inside its image, with room for the
// patch. The projection moves monotonically along the epipolar line, so this part is one
// interval. Without clipping, a near bound almost on the current image plane projects
// thousands of pixels away. The search then samples one patch per pixel of that segment,
// and patch_zero_mean_ssd rejects every one of them.
bool clip_to_visible_range(const Camera &camera, const Eigen::Isometry3d &T_cur_ref, const Vec2d &pixel_ref,
                           const cv::Size &size, int half, double &low, double &high) {
    const double margin = 1.0 + half;
    const auto visible = [&](double inverse) {
        Vec3d point_ref = Vec3d::Zero();
        if (!(inverse > 0.0) || !lift_reference_point(camera, pixel_ref, 1.0 / inverse, point_ref)) {
            return false;
        }
        const Vec3d point_cur = T_cur_ref * point_ref;
        if (!(point_cur.z() > 1e-8)) {
            return false;
        }
        const Vec2d px = camera.project_point(point_cur).first;
        return px.allFinite() && px.x() >= margin && px.y() >= margin && px.x() < size.width - 1 - margin &&
               px.y() < size.height - 1 - margin;
    };
    double inside = std::numeric_limits<double>::quiet_NaN();
    if (visible(low)) {
        inside = low;
    } else if (visible(high)) {
        inside = high;
    } else {
        for (int i = 1; i < kVisibleRangeProbes; ++i) {
            const double probe = low + (high - low) * static_cast<double>(i) / kVisibleRangeProbes;
            if (visible(probe)) {
                inside = probe;
                break;
            }
        }
    }
    if (!std::isfinite(inside)) {
        return false;
    }
    // Bisect each boundary between a visible and an invisible inverse depth.
    const auto boundary = [&](double in, double out) {
        if (visible(out)) {
            return out;
        }
        for (int i = 0; i < kVisibleRangeBisections; ++i) {
            const double mid = 0.5 * (in + out);
            (visible(mid) ? in : out) = mid;
        }
        return in;
    };
    const double new_low = boundary(inside, low);
    const double new_high = boundary(inside, high);
    low = new_low;
    high = new_high;
    return high > low;
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
    double low = std::max(1.0 / params.seed_max_depth, mu - radius);
    double high = std::min(1.0 / params.seed_min_depth, mu + radius);
    if (!(high > low)) {
        return match;
    }
    // Pixel length of the epipolar segment between the projections of two inverse depths.
    const auto segment_length = [&](double inverse_low, double inverse_high) {
        Vec3d near_ref = Vec3d::Zero();
        Vec3d far_ref = Vec3d::Zero();
        if (!lift_reference_point(camera, pixel_ref, 1.0 / inverse_high, near_ref) ||
            !lift_reference_point(camera, pixel_ref, 1.0 / inverse_low, far_ref)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const Vec2d near_px = camera.project_point(T_cur_ref * near_ref).first;
        const Vec2d far_px = camera.project_point(T_cur_ref * far_ref).first;
        if (!near_px.allFinite() || !far_px.allFinite()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (near_px - far_px).norm();
    };
    const double full_segment = segment_length(low, high);
    if (!std::isfinite(full_segment) || full_segment < 1.0) {
        return match;
    }
    match.searched = true;
    // Only the visible part can match. A segment entirely outside the image is a failed search.
    if (!clip_to_visible_range(camera, T_cur_ref, pixel_ref, img_cur.size(), params.feature_half_patch, low,
                               high)) {
        return match;
    }
    // Samples are uniform in inverse depth, but their pixel spacing is not. The pixel speed
    // |du/dw| of a perspective projection along the ray is monotonic in w, so its larger end
    // value bounds the spacing: sampling at that speed keeps every step within one pixel.
    const double span = high - low;
    const double probe = 1e-3 * span;
    const double speed = std::max(segment_length(low, low + probe), segment_length(high - probe, high)) / probe;
    if (!std::isfinite(speed)) {
        return match;
    }
    const int max_samples = kMaxSamplesPerImageSide * (img_cur.cols + img_cur.rows);
    // Clamp before converting: a near-singular projection can exceed the integer range.
    const int samples = static_cast<int>(std::clamp(std::ceil(speed * span) + 1.0, 2.0,
                                                   static_cast<double>(max_samples)));
    const int side = 2 * params.feature_half_patch + 1;
    const double gate =
        params.intensity_inlier_threshold * params.intensity_inlier_threshold * static_cast<double>(side * side);
    std::vector<double> inverses(static_cast<size_t>(samples), 0.0);
    std::vector<double> scores(static_cast<size_t>(samples), std::numeric_limits<double>::infinity());
    const double step = (high - low) / static_cast<double>(samples - 1);
    ReferencePatch patch(camera, img_ref, pixel_ref, params.feature_half_patch);
    for (int i = 0; i < samples; ++i) {
        const double inverse = low + step * static_cast<double>(i);
        inverses[static_cast<size_t>(i)] = inverse;
        double score = 0.0;
        if (patch.zero_mean_ssd(camera, T_cur_ref, img_cur, 1.0 / inverse, score)) {
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
    bool has_other_valley = false; // A finite score at least two samples from the best one.
    for (int i = 0; i < samples; ++i) {
        if (std::abs(i - best_index) <= 1) {
            continue;
        }
        if (std::isfinite(scores[static_cast<size_t>(i)])) has_other_valley = true;
        second = std::min(second, scores[static_cast<size_t>(i)]);
    }
    // Adjacent samples sit in the same valley. A unique minimum is at least twice
    // the next score, and that next score has to clear a one-gray-level floor. A short
    // segment (a few samples, e.g. a converging seed) can have no other valley at all:
    // nothing competes with the best score, which still has to pass the intensity gate.
    const bool unique = !has_other_valley || second > std::max(2.0 * best, best + 25.0);
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

namespace {
// Pixel baseline between pixel_cur and the reference ray at infinity, in the current image.
double parallax_to_infinity(const Camera &camera, const Eigen::Isometry3d &T_cur_ref, const Vec2d &pixel_ref,
                            const Vec2d &pixel_cur) {
    Vec3d bearing = Vec3d::Zero();
    if (!camera.unproject_bearing(pixel_ref, bearing, nullptr)) return 0.0;
    const Vec3d far_cur = T_cur_ref.linear() * bearing;
    if (!(far_cur.z() > 1e-8)) return 0.0;
    const Vec2d far_px = camera.project_point(far_cur).first;
    return far_px.allFinite() ? (pixel_cur - far_px).norm() : 0.0;
}
} // namespace

EpipolarMatch measure_seed_from_track(const Camera &camera, const DirectSlamParams &params,
                                      const Vec2d &pixel_ref, const Vec2d &pixel_cur,
                                      const Eigen::Isometry3d &T_cur_ref) {
    EpipolarMatch match;
    Vec3d f_ref = Vec3d::Zero();
    Vec3d f_cur = Vec3d::Zero();
    if (!camera.unproject_bearing(pixel_ref, f_ref, nullptr) || !camera.unproject_bearing(pixel_cur, f_cur, nullptr) ||
        !f_ref.allFinite() || !f_cur.allFinite() || !(T_cur_ref.translation().norm() > 1e-12)) {
        return match;
    }
    f_ref.normalize();
    f_cur.normalize();
    // Rays closer than a few pixels of angle carry no depth information yet (far points,
    // short baseline): skip instead of counting a noisy triangulation as an outlier.
    const double px_error_angle = 2.0 * std::atan(1.0 / (2.0 * camera.fx));
    const double ray_angle = std::acos(std::clamp((T_cur_ref.linear() * f_ref).dot(f_cur), -1.0, 1.0));
    if (ray_angle < kMinTrackRayAnglePx * px_error_angle) {
        return match;
    }
    match.searched = true;
    Vec3d X_ref = Vec3d::Zero();
    if (!triangulate_midpoint(f_ref, f_cur, T_cur_ref.linear(), T_cur_ref.translation(), X_ref)) {
        return match;
    }
    const double depth = X_ref.z();
    if (depth > params.seed_max_depth) {
        match.searched = false; // Beyond the far end of the prior: no evidence either way.
        return match;
    }
    const Vec2d reprojected = camera.project_point(T_cur_ref * X_ref).first;
    if (!(depth >= params.seed_min_depth) || !reprojected.allFinite() ||
        (reprojected - pixel_cur).squaredNorm() > params.chi2_threshold) {
        return match;
    }
    const double tau = depth_uncertainty_one_pixel(camera, T_cur_ref, pixel_ref, depth, px_error_angle);
    const double sigma = inverse_depth_sigma(depth, tau);
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        match.searched = false; // No usable parallax yet: skip, do not penalize.
        return match;
    }
    match.matched = true;
    match.depth = depth;
    match.inverse_variance = sigma * sigma;
    match.parallax_px = parallax_to_infinity(camera, T_cur_ref, pixel_ref, pixel_cur);
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

} // namespace pyslam
