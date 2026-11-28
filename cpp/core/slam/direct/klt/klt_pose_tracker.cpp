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

#include "slam/direct/klt/klt_pose_tracker.h"

#include "slam/direct/core/depth_measurement.h"
#include "slam/direct/geometry/reprojection_pose.h"
#include "slam/direct/lk/lk_tracker.h"

#include <cmath>

namespace pyslam {

TrackingResult KltPoseTracker::track(const TrackingInput &input) {
    const std::vector<LivePoint> &points = input.points;
    const Eigen::Isometry3d &T_pred_w = input.T_pred_w;
    TrackingResult result = TrackingResult::for_points(points.size());
    result.T_cur_w = T_pred_w;

    std::vector<Vec2d> pixels_last;
    std::vector<Vec2d> pixels_init;
    pixels_last.reserve(points.size());
    pixels_init.reserve(points.size());
    for (const LivePoint &point : points) {
        const Vec3d Xc = T_pred_w * point.position_w;
        Vec2d predicted = point.uv;
        if (Xc.z() > kMinTrackingDepth) {
            const auto projection = camera_->project_point(Xc);
            if (std::isfinite(projection.first.x()) && std::isfinite(projection.first.y())) {
                predicted = projection.first;
            }
        }
        pixels_last.push_back(point.uv);
        pixels_init.push_back(predicted);
    }
    const LkTrack tracked = track_pyramidal_lk(input.last_gray, input.frame.gray, pixels_last, pixels_init, params_);
    std::vector<PoseObservation> observations;
    observations.reserve(points.size());
    // Geometric observations omit failed LK tracks; retain a map back to the live-point indices.
    std::vector<size_t> observation_indices;
    for (size_t i = 0; i < points.size(); ++i) {
        result.tracked_ok[i] = tracked.status[i];
        result.tracked_uv[i] = tracked.points_cur[i];
        if (!tracked.status[i]) {
            continue;
        }
        PoseObservation observation;
        observation.point_ref = points[i].position_w;
        observation.pixel_cur = tracked.points_cur[i];
        observations.push_back(observation);
        observation_indices.push_back(i);
    }
    const ReprojectionPoseResult pose = estimate_pose_reprojection(*camera_, observations, T_pred_w, params_);
    result.T_cur_w = pose.T_cur_ref;
    result.inliers = pose.inliers;
    // Only geometric inliers may carry a landmark into the next image.
    for (size_t j = 0; j < observation_indices.size(); ++j) {
        result.tracked_ok[observation_indices[j]] = pose.inlier_mask[j];
    }
    // A prediction is only an optimizer seed. Never integrate unmeasured motion
    // or triangulate landmarks at a predicted pose after tracking has failed.
    result.accepted = result.inliers > params_.num_features_tracking_bad && result.T_cur_w.matrix().allFinite();
    return result;
}

} // namespace pyslam
