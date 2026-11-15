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

#include "slam_direct/direct/semi_direct_pose_tracker.h"

#include "slam_direct/depth_measurement.h"
#include "slam_direct/direct/planar_patch.h"
#include "slam_direct/features/corner_detection.h"
#include "slam_direct/geometry/reprojection_pose.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace pyslam {

SemiDirectPoseTracker::SemiDirectPoseTracker(CameraPtr camera, const DirectSlamParams &params)
    : PoseTracker(std::move(camera), params) {
    prior_sigma2_ = make_inverse_depth_prior(params_.seed_min_depth, params_.seed_max_depth).sigma2;
}

void SemiDirectPoseTracker::reset() {
    seeds_.clear();
}

TrackingResult SemiDirectPoseTracker::track(const TrackingInput &input) {
    const std::vector<LivePoint> &live_points = input.points;
    TrackingResult result;
    result.tracked_ok.assign(live_points.size(), 0);
    result.tracked_uv.assign(live_points.size(), Vec2d::Zero());

    const DirectKeyframe *reference = input.map.latest_keyframe();
    if (!reference || reference->gray.empty()) {
        return result;
    }
    Eigen::Isometry3d T_ref_w;
    T_ref_w.matrix() = reference->pose.Tcw();
    const Eigen::Isometry3d T_init = input.T_pred_w * T_ref_w.inverse();

    std::vector<Vec2d> pixels;
    std::vector<double> depths;
    std::vector<Vec3d> world_points;
    std::vector<size_t> live_index;
    for (size_t i = 0; i < live_points.size(); ++i) {
        const LivePoint &point = live_points[i];
        const Vec3d X_ref = T_ref_w * point.position_w;
        if (!(X_ref.z() > kMinTrackingDepth)) {
            continue;
        }
        // The patch must be read on the image it is paired with. A seed promoted after
        // a later keyframe still points at its older keyframe; project it instead.
        Vec2d pixel_ref = point.uv_ref;
        if (point.reference_keyframe != reference->id) {
            const Vec2d projected = camera_->project_point(X_ref).first;
            if (!projected.allFinite() || projected.x() < 0.0 || projected.y() < 0.0 ||
                projected.x() >= reference->gray.cols || projected.y() >= reference->gray.rows) {
                continue;
            }
            pixel_ref = projected;
        }
        pixels.push_back(pixel_ref);
        depths.push_back(X_ref.z());
        world_points.push_back(point.position_w);
        live_index.push_back(i);
    }
    if (pixels.size() < 6) {
        result.inliers = static_cast<int>(pixels.size());
        return result;
    }

    const PlanarPoseResult aligned =
        align_planar_patch_pose(*camera_, params_, reference->gray, input.frame.gray, pixels, depths, T_init);
    const Eigen::Isometry3d T_cur_ref =
        aligned.inliers > params_.num_features_tracking_bad ? aligned.T_cur_ref : T_init;
    const PatchOffsetResult offsets =
        align_planar_patch_offsets(*camera_, params_, reference->gray, input.frame.gray, pixels, depths, T_cur_ref);
    std::vector<PoseObservation> observations;
    std::vector<size_t> slots;
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (!offsets.status[i]) {
            continue;
        }
        observations.push_back(PoseObservation{world_points[i], offsets.pixels_cur[i]});
        slots.push_back(i);
    }
    const ReprojectionPoseResult pose =
        estimate_pose_reprojection(*camera_, observations, T_cur_ref * T_ref_w, params_);
    result.T_cur_w = T_cur_ref * T_ref_w;
    if (pose.inliers > params_.num_features_tracking_bad && pose.T_cur_ref.matrix().allFinite()) {
        result.T_cur_w = pose.T_cur_ref;
        for (size_t j = 0; j < slots.size(); ++j) {
            if (!pose.inlier_mask[j]) {
                continue;
            }
            const size_t index = live_index[slots[j]];
            result.tracked_ok[index] = 1;
            result.tracked_uv[index] = offsets.pixels_cur[slots[j]];
            ++result.inliers;
        }
    }
    result.accepted = result.inliers > params_.num_features_tracking_bad && result.T_cur_w.matrix().allFinite();
    return result;
}

AcceptOutcome SemiDirectPoseTracker::after_accept(const TrackingFrame &frame, const Eigen::Isometry3d &T_cur_w,
                                                  TrackerMapState &state) {
    update_depth_seeds(frame.gray, T_cur_w, state.map);
    promote_depth_seeds(T_cur_w, state);
    return {};
}

void SemiDirectPoseTracker::on_keyframe(const TrackingFrame &frame, int keyframe_id, TrackerMapState &state) {
    spawn_depth_seeds(frame, keyframe_id, state);
    release_idle_keyframe_images(state.map);
}

void SemiDirectPoseTracker::spawn_depth_seeds(const TrackingFrame &frame, int keyframe_id,
                                              const TrackerMapState &state) {
    const std::vector<LivePoint> &live_points = state.live_points;
    if (static_cast<int>(seeds_.size()) >= params_.max_seeds) {
        return;
    }
    const cv::Mat &gray = frame.gray;
    cv::Mat occupied(gray.size(), CV_8U, cv::Scalar(255));
    const int radius = std::max(1, static_cast<int>(std::lround(params_.gftt_min_distance)));
    auto block = [&](const Vec2d &uv) {
        cv::circle(occupied,
                   cv::Point(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y()))),
                   radius, cv::Scalar(0), cv::FILLED);
    };
    for (const LivePoint &point : live_points) {
        block(point.uv);
    }
    for (const DepthSeed &seed : seeds_) {
        if (seed.keyframe_id == keyframe_id) {
            block(seed.pixel);
        }
    }
    const int room = params_.max_seeds - static_cast<int>(seeds_.size());
    const std::vector<Vec2d> pixels = detect_corners(gray, params_, occupied, room);
    if (pixels.empty()) {
        return;
    }
    // A corner with a sensor measurement is a landmark candidate, not a seed.
    const bool stereo = camera_->sensor_type == SensorType::STEREO;
    const std::vector<PixelMeasurement> measured = measure_pixels(
        *camera_, params_, gray, stereo ? frame.gray_right : cv::Mat(), frame.depth, pixels);
    // SVO centers new seeds on the keyframe's median scene depth. Without landmarks the
    // prior stays at the middle of the inverse-depth range.
    double scene_depth = 0.0;
    const auto &keyframes = state.map.keyframes();
    if (!live_points.empty() && keyframe_id >= 0 && static_cast<size_t>(keyframe_id) < keyframes.size()) {
        Eigen::Isometry3d T_kf_w;
        T_kf_w.matrix() = keyframes[static_cast<size_t>(keyframe_id)].pose.Tcw();
        std::vector<double> z;
        z.reserve(live_points.size());
        for (const LivePoint &point : live_points) {
            const double depth = (T_kf_w * point.position_w).z();
            if (depth > kMinTrackingDepth && std::isfinite(depth)) {
                z.push_back(depth);
            }
        }
        if (!z.empty()) {
            std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
            scene_depth = z[z.size() / 2];
        }
    }
    for (size_t i = 0; i < pixels.size() && static_cast<int>(seeds_.size()) < params_.max_seeds; ++i) {
        if (measured[i].ok) {
            continue;
        }
        DepthSeed seed;
        seed.keyframe_id = keyframe_id;
        seed.pixel = pixels[i];
        seed.belief = make_inverse_depth_prior(params_.seed_min_depth, params_.seed_max_depth, scene_depth);
        seeds_.push_back(seed);
    }
}

void SemiDirectPoseTracker::update_depth_seeds(const cv::Mat &gray, const Eigen::Isometry3d &T_cur_w,
                                               const DirectMap &map) {
    const auto &keyframes = map.keyframes();
    std::vector<DepthSeed> kept;
    kept.reserve(seeds_.size());
    int searches = 0;
    for (DepthSeed seed : seeds_) {
        if (seed.keyframe_id < 0 || static_cast<size_t>(seed.keyframe_id) >= keyframes.size() ||
            keyframes[static_cast<size_t>(seed.keyframe_id)].gray.empty() || searches >= params_.max_seed_updates) {
            kept.push_back(std::move(seed));
            continue;
        }
        const DirectKeyframe &keyframe = keyframes[static_cast<size_t>(seed.keyframe_id)];
        Eigen::Isometry3d T_ref_w;
        T_ref_w.matrix() = keyframe.pose.Tcw();
        const EpipolarMatch match = match_seed_on_epipolar_line(
            *camera_, params_, keyframe.gray, gray, seed.pixel, seed.belief.mu, seed.belief.sigma2,
            T_cur_w * T_ref_w.inverse());
        if (match.searched) {
            ++searches;
        }
        observe_seed(seed, match, params_.seed_min_depth, params_.seed_max_depth);
        if (!seed_expired(seed, prior_sigma2_)) {
            kept.push_back(std::move(seed));
        }
    }
    seeds_ = std::move(kept);
}

int SemiDirectPoseTracker::promote_depth_seeds(const Eigen::Isometry3d &T_cur_w, TrackerMapState &state) {
    auto &keyframes = state.map.keyframes_mutable();
    std::vector<DepthSeed> remaining;
    int promoted = 0;
    for (const DepthSeed &seed : seeds_) {
        const bool converged = seed.belief.sigma2 < params_.seed_convergence_sigma2 && seed.parallax_px >= 2.0;
        if (!converged || seed.keyframe_id < 0 || static_cast<size_t>(seed.keyframe_id) >= keyframes.size()) {
            remaining.push_back(seed);
            continue;
        }
        const double depth = 1.0 / seed.belief.mu;
        Vec3d X_ref = Vec3d::Zero();
        if (!lift_pixel_depth(*camera_, seed.pixel, depth, X_ref) || !metric_point_ok(*camera_, X_ref)) {
            remaining.push_back(seed);
            continue;
        }
        Eigen::Isometry3d T_ref_w;
        T_ref_w.matrix() = keyframes[static_cast<size_t>(seed.keyframe_id)].pose.Tcw();
        LivePoint live;
        live.id = state.next_point_id++;
        live.uv_ref = seed.pixel;
        live.reference_keyframe = seed.keyframe_id;
        live.position_w = T_ref_w.inverse() * X_ref;
        const Vec3d X_cur = T_cur_w * live.position_w;
        const Vec2d projected = camera_->project_point(X_cur).first;
        live.uv = projected.allFinite() ? projected : seed.pixel;
        live.trail.emplace_back(std::floor(live.uv.x()), std::floor(live.uv.y()));
        live.observations = 1;
        DirectLandmark landmark;
        landmark.position_w = live.position_w;
        if (camera_->sensor_type == SensorType::MONOCULAR) {
            landmark.has_inverse_prior = true;
            landmark.inverse_depth_mu = seed.belief.mu;
            landmark.inverse_depth_sigma2 = std::max(seed.belief.sigma2, 1e-8);
            landmark.prior_keyframe = seed.keyframe_id;
        }
        state.map.landmarks_mutable()[live.id] = landmark;
        DirectPoint observation;
        observation.id = live.id;
        observation.uv = seed.pixel;
        observation.position_c = X_ref;
        observation.depth = X_ref.z();
        keyframes[static_cast<size_t>(seed.keyframe_id)].points.push_back(observation);
        state.map.note_observation(seed.keyframe_id, live.id);
        state.live_points.push_back(std::move(live));
        ++promoted;
    }
    seeds_ = std::move(remaining);
    return promoted;
}

void SemiDirectPoseTracker::release_idle_keyframe_images(DirectMap &map) {
    auto &keyframes = map.keyframes_mutable();
    if (keyframes.size() < 2) {
        return;
    }
    for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
        const int id = keyframes[i].id;
        const bool owns_seed = std::any_of(seeds_.begin(), seeds_.end(), [&](const DepthSeed &seed) {
            return seed.keyframe_id == id;
        });
        if (!owns_seed) {
            keyframes[i].gray.release();
        }
    }
}

} // namespace pyslam
