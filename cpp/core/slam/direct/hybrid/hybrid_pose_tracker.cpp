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

#include "slam/direct/hybrid/hybrid_pose_tracker.h"

#include "slam/direct/core/depth_measurement.h"
#include "slam/direct/photometric/direct_aligner.h"
#include "slam/direct/core/image_utils.h"

#include <cmath>
#include <limits>

namespace pyslam {

TrackingResult HybridPoseTracker::track(const TrackingInput &input) {
    TrackingResult result = klt_.track(input);
    const std::vector<LivePoint> &points = input.points;
    const cv::Mat &gray = input.frame.gray;
    const Eigen::Isometry3d &T_prev_w = input.T_prev_w;
    std::vector<char> &tracked_ok = result.tracked_ok;
    std::vector<Vec2d> &tracked_uv = result.tracked_uv;
    Eigen::Isometry3d &T_cur_w = result.T_cur_w;
    int &inliers = result.inliers;
    // Photometric reference: live points at their last-image pixel and depth.
    struct Reference {
        std::vector<Vec2d> pixels;
        std::vector<double> depths;
    };
    const auto make_reference = [&](bool only_tracked) {
        Reference reference;
        for (size_t i = 0; i < points.size(); ++i) {
            if (only_tracked && !tracked_ok[i]) continue;
            reference.pixels.push_back(points[i].uv);
            reference.depths.push_back((T_prev_w * points[i].position_w).z());
        }
        return reference;
    };

    if (inliers <= params_.num_features_tracking_bad) {
        // Photometric tracking can still resolve motion when corner flow fails
        // (for example at a repetitive pattern's half-period displacement).
        const Reference reference = make_reference(false);
        const auto aligned = align_direct_multilayer(*camera_, params_, input.last_gray, gray, reference.pixels,
                                                     reference.depths, input.T_pred_w * T_prev_w.inverse());
        if (aligned.inliers > params_.num_features_tracking_bad &&
            aligned.inliers >= 0.6 * points.size()) {
            T_cur_w = aligned.T_cur_ref * T_prev_w;
            inliers = 0;
            for (size_t i = 0; i < points.size(); ++i) {
                const Vec3d X = T_cur_w * points[i].position_w;
                tracked_uv[i] = camera_->project_point(X).first;
                tracked_ok[i] = aligned.inlier_mask[i] && X.z() > kMinTrackingDepth &&
                                inside_border(tracked_uv[i], gray.size(), params_.border);
                inliers += tracked_ok[i];
            }
        }
    } else {
        // Use geometric tracks to seed and validate photometric refinement. Outdoor
        // brightness changes and moving objects can produce a low photometric cost
        // at the wrong pose, so retain the geometric solution if refinement disagrees.
        const Reference reference = make_reference(true);
        const auto aligned = align_direct_multilayer(*camera_, params_, input.last_gray, gray, reference.pixels,
                                                     reference.depths, T_cur_w * T_prev_w.inverse());
        const Eigen::Isometry3d candidate = aligned.T_cur_ref * T_prev_w;
        double geometric_cost = 0.0, candidate_cost = 0.0;
        int candidate_inliers = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            if (!tracked_ok[i]) continue;
            const Vec3d X = candidate * points[i].position_w;
            const Vec2d uv = camera_->project_point(X).first;
            const double error = (uv - tracked_uv[i]).squaredNorm();
            if (X.z() > 0.0 && std::isfinite(error)) {
                candidate_cost += error;
                if (error <= params_.chi2_threshold) ++candidate_inliers;
            } else {
                candidate_cost = std::numeric_limits<double>::infinity();
            }
            geometric_cost += (camera_->project_point(T_cur_w * points[i].position_w).first -
                               tracked_uv[i]).squaredNorm();
        }
        if (aligned.inliers > params_.num_features_tracking_bad &&
            candidate_inliers >= 0.9 * inliers && candidate_cost <= geometric_cost + 0.25 * inliers) {
            T_cur_w = candidate;
            inliers = 0;
            for (size_t i = 0; i < points.size(); ++i) {
                if (!tracked_ok[i]) continue;
                const Vec3d X = T_cur_w * points[i].position_w;
                tracked_ok[i] = (camera_->project_point(X).first - tracked_uv[i]).squaredNorm() <=
                                params_.chi2_threshold;
                inliers += tracked_ok[i];
            }
        }
    }
    result.accepted = inliers > params_.num_features_tracking_bad && T_cur_w.matrix().allFinite();
    return result;
}

AcceptOutcome HybridPoseTracker::after_accept(const TrackingFrame &frame, const Eigen::Isometry3d &T_cur_w,
                                              TrackerMapState &state) {
    if (params_.local_ba) {
        return {};
    }
    // Photometric warps need depth at the current reference pixel. Refresh it
    // from the current sensor measurement instead of carrying stereo errors forever.
    std::vector<LivePoint> &live_points = state.live_points;
    const Eigen::Isometry3d Twc = T_cur_w.inverse();
    const std::vector<PixelMeasurement> measured_depth =
        measure_live_points(*camera_, params_, frame.gray, frame.gray_right, frame.depth, T_cur_w, live_points);
    std::vector<LivePoint> measured;
    for (size_t i = 0; i < live_points.size(); ++i) {
        if (!measured_depth[i].ok) continue;
        LivePoint updated = live_points[i];
        updated.position_w = Twc * measured_depth[i].X;
        measured.push_back(updated);
    }
    live_points = std::move(measured);
    return {};
}

} // namespace pyslam
