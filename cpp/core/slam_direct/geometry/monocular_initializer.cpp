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

#include "slam_direct/geometry/monocular_initializer.h"

#include "slam_direct/depth_measurement.h"
#include "slam_direct/direct/depth_filter.h"
#include "slam_direct/features/corner_detection.h"
#include "slam_direct/geometry/triangulation.h"
#include "slam_direct/klt/klt_tracker.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cmath>

namespace pyslam {
namespace {

constexpr double kPi = 3.14159265358979323846;
// Per-point triangulation angle; flatter rays are left to the depth filter.
constexpr double kMinPointParallaxDeg = 0.5;
// The accepted set must have this median triangulation angle (ORB-SLAM uses 1 degree).
constexpr double kMinMedianParallaxDeg = 1.0;
// Rays this far off-axis do not map to the normalized plane used by the essential matrix.
constexpr double kMinBearingZ = 0.2;
constexpr size_t kMaxTrail = 64;

double median_of(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

double scene_depth(const Camera &camera, const Vec3d &X) {
    return camera.projects_raw_keypoints() ? X.norm() : X.z();
}

} // namespace

MonocularInitializer::MonocularInitializer(CameraPtr camera, const DirectSlamParams &params)
    : camera_(std::move(camera)), params_(params) {}

void MonocularInitializer::reset() {
    ref_gray_.release();
    last_gray_.release();
    ref_img_id_ = -1;
    last_img_id_ = -1;
    ref_timestamp_ = 0.0;
    tracks_.clear();
    result_ = MonocularInitResult();
}

void MonocularInitializer::set_reference(const cv::Mat &gray, int img_id, double timestamp) {
    ref_gray_ = gray.clone();
    last_gray_ = ref_gray_;
    ref_img_id_ = img_id;
    last_img_id_ = img_id;
    ref_timestamp_ = timestamp;
    tracks_.clear();
    const std::vector<Vec2d> corners = detect_corners(gray, params_);
    tracks_.reserve(corners.size());
    for (size_t i = 0; i < corners.size(); ++i) {
        LivePoint track;
        track.id = static_cast<int64_t>(i);
        track.uv = corners[i];
        track.uv_ref = corners[i];
        track.trail.emplace_back(std::floor(corners[i].x()), std::floor(corners[i].y()));
        track.observations = 1;
        tracks_.push_back(std::move(track));
    }
}

bool MonocularInitializer::add_frame(const cv::Mat &gray, int img_id, double timestamp) {
    if (ref_gray_.empty()) {
        set_reference(gray, img_id, timestamp);
        return false;
    }
    std::vector<Vec2d> previous;
    previous.reserve(tracks_.size());
    for (const LivePoint &track : tracks_) previous.push_back(track.uv);
    const LkTrack lk = track_pyramidal_lk(last_gray_, gray, previous, previous, params_);
    const bool contiguous = img_id == last_img_id_ + 1;
    std::vector<LivePoint> kept;
    kept.reserve(tracks_.size());
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!lk.status[i]) continue;
        LivePoint track = std::move(tracks_[i]);
        track.uv = lk.points_cur[i];
        if (!contiguous) track.trail.clear();
        track.trail.insert(track.trail.begin(), cv::Point(std::floor(track.uv.x()), std::floor(track.uv.y())));
        if (track.trail.size() > kMaxTrail) track.trail.resize(kMaxTrail);
        ++track.observations;
        kept.push_back(std::move(track));
    }
    tracks_ = std::move(kept);
    last_gray_ = gray.clone();
    last_img_id_ = img_id;

    // Accepting needs more than num_features_tracking triangulated points.
    if (num_tracks() <= params_.num_features_tracking) {
        set_reference(gray, img_id, timestamp);
        return false;
    }
    std::vector<double> disparity;
    disparity.reserve(tracks_.size());
    for (const LivePoint &track : tracks_) disparity.push_back((track.uv - track.uv_ref).norm());
    const double median_disparity = median_of(disparity);
    if (median_disparity < params_.init_min_disparity) return false;
    if (try_two_view()) return true;
    // Large image motion without usable parallax (e.g. pure rotation): start over here.
    if (median_disparity > 4.0 * params_.init_min_disparity) set_reference(gray, img_id, timestamp);
    return false;
}

bool MonocularInitializer::try_two_view() {
    const size_t n = tracks_.size();
    std::vector<Vec3d> f_ref(n), f_cur(n);
    std::vector<cv::Point2d> p_ref, p_cur;
    std::vector<size_t> index;
    p_ref.reserve(n);
    p_cur.reserve(n);
    index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Vec3d a = Vec3d::Zero();
        Vec3d b = Vec3d::Zero();
        if (!camera_->unproject_bearing(tracks_[i].uv_ref, a, nullptr) ||
            !camera_->unproject_bearing(tracks_[i].uv, b, nullptr) || !a.allFinite() || !b.allFinite()) {
            continue;
        }
        a.normalize();
        b.normalize();
        f_ref[i] = a;
        f_cur[i] = b;
        if (a.z() < kMinBearingZ || b.z() < kMinBearingZ) continue;
        p_ref.emplace_back(a.x() / a.z(), a.y() / a.z());
        p_cur.emplace_back(b.x() / b.z(), b.y() / b.z());
        index.push_back(i);
    }
    if (static_cast<int>(index.size()) <= std::max(params_.num_features_tracking, 5)) return false;

    // RANSAC threshold on the normalized plane: the reprojection gate converted by the focal length.
    const double threshold = std::sqrt(params_.chi2_threshold) / std::max(camera_->fx, camera_->fy);
    cv::Mat mask;
    const cv::Mat E = cv::findEssentialMat(p_ref, p_cur, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.999,
                                           threshold, mask);
    if (E.rows != 3 || E.cols != 3) return false; // Missing or multiple solutions.
    cv::Mat R_cv, t_cv;
    const int good = cv::recoverPose(E, p_ref, p_cur, R_cv, t_cv, 1.0, cv::Point2d(0.0, 0.0), mask);
    if (good <= params_.num_features_tracking) return false;
    Mat3d R;
    Vec3d t;
    cv::cv2eigen(R_cv, R);
    cv::cv2eigen(t_cv, t);
    if (!R.allFinite() || !t.allFinite() || !(t.norm() > 0.0)) return false;
    t.normalize();

    // Triangulate with a unit baseline, then rescale.
    const Vec3d center_cur = -R.transpose() * t; // Current camera center, in the reference frame.
    const double min_cos = std::cos(kMinPointParallaxDeg * kPi / 180.0);
    std::vector<size_t> accepted;
    std::vector<Vec3d> points;
    std::vector<double> depths, parallax;
    for (size_t k = 0; k < index.size(); ++k) {
        if (!mask.at<uchar>(static_cast<int>(k))) continue;
        const size_t i = index[k];
        Vec3d X = Vec3d::Zero();
        if (!triangulate_midpoint(f_ref[i], f_cur[i], R, t, X)) continue;
        const double cos_parallax = X.normalized().dot((X - center_cur).normalized());
        if (!(cos_parallax < min_cos)) continue;
        const Vec3d X_cur = R * X + t;
        const Vec2d uv_ref = camera_->project_point(X).first;
        const Vec2d uv_cur = camera_->project_point(X_cur).first;
        if (!uv_ref.allFinite() || !uv_cur.allFinite() ||
            (uv_ref - tracks_[i].uv_ref).squaredNorm() > params_.chi2_threshold ||
            (uv_cur - tracks_[i].uv).squaredNorm() > params_.chi2_threshold) {
            continue;
        }
        accepted.push_back(i);
        points.push_back(X);
        depths.push_back(scene_depth(*camera_, X));
        parallax.push_back(std::acos(std::clamp(cos_parallax, -1.0, 1.0)) * 180.0 / kPi);
    }
    if (static_cast<int>(accepted.size()) <= params_.num_features_tracking ||
        median_of(parallax) < kMinMedianParallaxDeg) {
        return false;
    }
    const double median_depth = median_of(depths);
    if (!(median_depth > 0.0)) return false;
    const double scale = params_.monocular_init_median_depth() / median_depth;

    MonocularInitResult result;
    result.ref_img_id = ref_img_id_;
    result.ref_timestamp = ref_timestamp_;
    result.ref_gray = ref_gray_;
    result.T_cur_ref.linear() = R;
    result.T_cur_ref.translation() = scale * t;
    const double px_error_angle = 2.0 * std::atan(1.0 / (2.0 * camera_->fx));
    for (size_t k = 0; k < accepted.size(); ++k) {
        const Vec3d X = scale * points[k];
        if (!metric_point_ok(*camera_, params_, X)) continue;
        const LivePoint &track = tracks_[accepted[k]];
        // Inverse-depth variance of a one-pixel error, as the depth filter measures it.
        double sigma2 = 0.0;
        const double z = X.z();
        const double tau = depth_uncertainty_one_pixel(*camera_, result.T_cur_ref, track.uv_ref, z, px_error_angle);
        if (tau > 0.0) {
            const double sigma = 0.5 * (1.0 / std::max(1e-7, z - tau) - 1.0 / (z + tau));
            if (sigma > 0.0 && std::isfinite(sigma)) sigma2 = sigma * sigma;
        }
        result.uv_ref.push_back(track.uv_ref);
        result.uv_cur.push_back(track.uv);
        result.X_ref.push_back(X);
        result.inverse_depth_sigma2.push_back(sigma2);
        result.trails.push_back(track.trail);
    }
    if (static_cast<int>(result.X_ref.size()) <= params_.num_features_tracking) return false;
    result_ = std::move(result);
    return true;
}

} // namespace pyslam
