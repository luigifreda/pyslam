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
#include "slam_direct/image_utils.h"
#include "slam_direct/klt/klt_tracker.h"

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
    prev_gray_.release();
}

TrackingResult SemiDirectPoseTracker::track(const TrackingInput &input) {
    const std::vector<LivePoint> &live_points = input.points;
    TrackingResult result = TrackingResult::for_points(live_points.size());

    const DirectKeyframe *reference = input.map.latest_keyframe();
    if (!reference || reference->gray.empty()) {
        return result;
    }
    const Eigen::Isometry3d &T_ref_w = reference->pose.isometry3d();
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

    // Both alignments run on the same two images: build each pyramid once.
    require_gray(reference->gray, "planar patch reference");
    require_gray(input.frame.gray, "planar patch current");
    const ImagePyramid pyramid_ref = build_pyramid(reference->gray, params_);
    const ImagePyramid pyramid_cur = build_pyramid(input.frame.gray, params_);
    const PlanarPoseResult aligned =
        align_planar_patch_pose(*camera_, params_, pyramid_ref, pyramid_cur, pixels, depths, T_init);
    const Eigen::Isometry3d T_cur_ref =
        aligned.inliers > params_.num_features_tracking_bad ? aligned.T_cur_ref : T_init;
    const PatchOffsetResult offsets =
        align_planar_patch_offsets(*camera_, params_, pyramid_ref, pyramid_cur, pixels, depths, T_cur_ref);
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
    prev_gray_ = frame.gray.clone();
    promote_depth_seeds(T_cur_w, state);
    return {};
}

void SemiDirectPoseTracker::on_keyframe(const TrackingFrame &frame, int keyframe_id, TrackerMapState &state) {
    spawn_depth_seeds(frame, keyframe_id, state);
    prev_gray_ = frame.gray.clone();
    release_idle_keyframe_images(state.map);
}

void SemiDirectPoseTracker::apply_keyframe_scales(const std::vector<double> &scales) {
    // Seed inverse depths live in their keyframe camera: depth / s means inverse depth * s.
    for (DepthSeed &seed : seeds_) {
        if (seed.keyframe_id < 0 || static_cast<size_t>(seed.keyframe_id) >= scales.size()) continue;
        const double scale = scales[static_cast<size_t>(seed.keyframe_id)];
        seed.belief.mu *= scale;
        seed.belief.sigma2 *= scale * scale;
    }
}

void SemiDirectPoseTracker::spawn_depth_seeds(const TrackingFrame &frame, int keyframe_id,
                                              const TrackerMapState &state) {
    const std::vector<LivePoint> &live_points = state.live_points;
    if (static_cast<int>(seeds_.size()) >= params_.max_seeds) {
        return;
    }
    const cv::Mat &gray = frame.gray;
    OccupancyMask occupied(gray.size(), params_);
    for (const LivePoint &point : live_points) {
        occupied.block(point.uv);
    }
    for (const DepthSeed &seed : seeds_) {
        if (seed.tracked) {
            occupied.block(seed.pixel_cur); // Tracked seeds are in this image, whatever their keyframe.
        } else if (seed.keyframe_id == keyframe_id) {
            occupied.block(seed.pixel);
        }
    }
    const int room = params_.max_seeds - static_cast<int>(seeds_.size());
    const std::vector<Vec2d> pixels = detect_corners(gray, params_, occupied.mask(), room);
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
        const Eigen::Isometry3d &T_kf_w = keyframes[static_cast<size_t>(keyframe_id)].pose.isometry3d();
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
        seed.pixel_cur = pixels[i];
        seed.tracked = true;
        seed.belief = make_inverse_depth_prior(params_.seed_min_depth, params_.seed_max_depth, scene_depth);
        seeds_.push_back(seed);
    }
}

void SemiDirectPoseTracker::update_depth_seeds(const cv::Mat &gray, const Eigen::Isometry3d &T_cur_w,
                                               const DirectMap &map) {
    const auto &keyframes = map.keyframes();
    // Small inter-frame appearance changes keep LK reliable under forward motion, where a
    // fronto-parallel patch matched against an older keyframe image no longer fits.
    std::vector<size_t> tracked_index;
    std::vector<Vec2d> tracked_prev;
    for (size_t i = 0; i < seeds_.size(); ++i) {
        if (seeds_[i].tracked) {
            tracked_index.push_back(i);
            tracked_prev.push_back(seeds_[i].pixel_cur);
        }
    }
    if (!tracked_index.empty()) {
        if (prev_gray_.empty() || prev_gray_.size() != gray.size()) {
            for (size_t i : tracked_index) seeds_[i].tracked = false;
        } else {
            const LkTrack lk = track_pyramidal_lk(prev_gray_, gray, tracked_prev, tracked_prev, params_);
            for (size_t k = 0; k < tracked_index.size(); ++k) {
                DepthSeed &seed = seeds_[tracked_index[k]];
                seed.tracked = lk.status[k] != 0;
                if (seed.tracked) seed.pixel_cur = lk.points_cur[k];
            }
        }
    }

    std::vector<DepthSeed> kept;
    kept.reserve(seeds_.size());
    int searches = 0;
    for (DepthSeed seed : seeds_) {
        const bool known_keyframe = seed.keyframe_id >= 0 && static_cast<size_t>(seed.keyframe_id) < keyframes.size();
        if (!known_keyframe) {
            kept.push_back(std::move(seed));
            continue;
        }
        const DirectKeyframe &keyframe = keyframes[static_cast<size_t>(seed.keyframe_id)];
        const Eigen::Isometry3d &T_ref_w = keyframe.pose.isometry3d();
        const Eigen::Isometry3d T_cur_ref = T_cur_w * T_ref_w.inverse();
        EpipolarMatch match;
        if (seed.tracked) {
            match = measure_seed_from_track(*camera_, params_, seed.pixel, seed.pixel_cur, T_cur_ref);
        } else {
            // A lost track gets one epipolar search on the keyframe image (capped per frame).
            // A match re-acquires the seed; otherwise it is dropped, so stale seeds do not
            // hold max_seeds slots that new keyframes need.
            if (keyframe.gray.empty() || searches >= params_.max_seed_updates) {
                kept.push_back(std::move(seed));
                continue;
            }
            match = match_seed_on_epipolar_line(*camera_, params_, keyframe.gray, gray, seed.pixel,
                                                seed.belief.mu, seed.belief.sigma2, T_cur_ref);
            ++searches;
            Vec3d X_ref = Vec3d::Zero();
            if (!match.matched || !lift_pixel_depth(*camera_, seed.pixel, match.depth, X_ref)) {
                continue;
            }
            const Vec2d reacquired = camera_->project_point(T_cur_ref * X_ref).first;
            if (!reacquired.allFinite() || !inside_border(reacquired, gray.size(), params_.border)) {
                continue;
            }
            seed.pixel_cur = reacquired;
            seed.tracked = true;
        }
        observe_seed(seed, match, params_.seed_min_depth, params_.seed_max_depth);
        if (!seed_expired(seed, prior_sigma2_)) {
            kept.push_back(std::move(seed));
        }
    }
    seeds_ = std::move(kept);
}

int SemiDirectPoseTracker::promote_depth_seeds(const Eigen::Isometry3d &T_cur_w, TrackerMapState &state) {
    const auto &keyframes = state.map.keyframes();
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
        if (!lift_pixel_depth(*camera_, seed.pixel, depth, X_ref) || !metric_point_ok(*camera_, params_, X_ref)) {
            remaining.push_back(seed);
            continue;
        }
        const Eigen::Isometry3d &T_ref_w = keyframes[static_cast<size_t>(seed.keyframe_id)].pose.isometry3d();
        const Vec3d position_w = T_ref_w.inverse() * X_ref;
        // A tracked seed is already in the current image; otherwise project the new landmark.
        const Vec2d projected = camera_->project_point(T_cur_w * position_w).first;
        const Vec2d uv = seed.tracked ? seed.pixel_cur : (projected.allFinite() ? projected : seed.pixel);
        LivePoint live = make_live_point(state.next_point_id++, uv, position_w);
        // The patch reference stays on the seed's keyframe.
        live.uv_ref = seed.pixel;
        live.reference_keyframe = seed.keyframe_id;
        DirectLandmark landmark;
        landmark.position_w = position_w;
        if (camera_->sensor_type == SensorType::MONOCULAR) {
            landmark.set_inverse_prior(seed.belief.mu, std::max(seed.belief.sigma2, 1e-8), seed.keyframe_id);
        }
        state.map.landmarks_mutable()[live.id] = landmark;
        DirectPoint observation;
        observation.id = live.id;
        observation.uv = seed.pixel;
        observation.position_c = X_ref;
        observation.depth = X_ref.z();
        state.map.add_observation(seed.keyframe_id, std::move(observation));
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
