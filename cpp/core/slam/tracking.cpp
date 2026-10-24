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
#include "tracking.h"

#include "config_parameters.h"
#include "frontend_helpers.h"
#include "geometry_matchers.h"
#include "optimizer_g2o.h"
#include "optimizer_gtsam.h"
#include "rotation_histogram.h"
#include "slam.h"
#include "tracking_core.h"
#include "utils/descriptor_helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace pyslam {

namespace {

std::pair<bool, float> detect_blur_laplacian(const cv::Mat &img, double threshold) {
    if (img.empty()) {
        return {false, 0.f};
    }
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img;
    }
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    const float variance = static_cast<float>(stddev[0] * stddev[0]);
    return {variance < threshold, variance};
}

float far_threshold_or_inf(const std::optional<double> &value) {
    if (!value) {
        return std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(*value);
}

} // namespace

Tracking::Tracking(Slam *slam_in)
    : slam(slam_in), initializer(slam_in ? slam_in->sensor_type : SensorType::MONOCULAR),
      dyn_config(slam_in ? slam_in->feature_max_descriptor_distance() : Parameters::kMaxDescriptorDistance.load()) {
    descriptor_distance_sigma = dyn_config.descriptor_distance_sigma;
    reproj_err_frame_map_sigma = Parameters::kMaxReprojectionDistanceMap;
    if (slam && slam->sensor_type == SensorType::RGBD) {
        reproj_err_frame_map_sigma = Parameters::kMaxReprojectionDistanceMapRgbd;
    }
    if (slam && slam->camera && slam->camera->fps > 0) {
        max_frames_between_kfs = slam->camera->fps;
    }
    max_frames_between_kfs_after_reloc = max_frames_between_kfs;
}

CameraPtr Tracking::camera() const { return slam ? slam->camera : nullptr; }

Tracking::Snapshot Tracking::snapshot() const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return snapshot_;
}

void Tracking::publish_snapshot() {
    Snapshot next{f_cur, kf_ref, predicted_pose, cur_R, cur_t, has_current_pose};
    next.num_matched_kps = num_matched_kps;
    next.num_inliers = num_inliers;
    next.num_matched_map_points = num_matched_map_points;
    next.num_kf_ref_tracked_points = num_kf_ref_tracked_points;
    next.last_num_static_stereo_map_points = last_num_static_stereo_map_points;
    next.total_num_static_stereo_map_points = total_num_static_stereo_map_points;
    next.mean_pose_opt_chi2_error = mean_pose_opt_chi2_error;
    next.time_track = time_track;
    // Destroy the old FramePtr outside the lock: its image owner may need the GIL.
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        std::swap(snapshot_, next);
    }
}

bool Tracking::use_motion_model() const {
    return Parameters::kUseMotionModel || Parameters::kUseSearchFrameByProjection;
}

bool Tracking::use_search_frame_by_projection() const {
    return Parameters::kUseSearchFrameByProjection && !Parameters::kUseEssentialMatrixFitting;
}

void Tracking::reset() {
    initializer.reset();
    motion_model.reset();
    state = SlamState::NO_IMAGES_YET;
    num_matched_kps = 0;
    num_inliers = 0;
    num_matched_map_points = 0;
    num_matched_map_points_in_last_pose_opt = 0;
    num_kf_ref_tracked_points = 0;
    last_num_static_stereo_map_points = 0;
    total_num_static_stereo_map_points = 0;
    pose_is_ok = false;
    mean_pose_opt_chi2_error = 0.0;
    predicted_pose.reset();
    f_cur.reset();
    f_ref.reset();
    idxs_cur.clear();
    idxs_ref.clear();
    kf_ref.reset();
    kf_last.reset();
    local_keyframes.clear();
    local_points.clear();
    vo_points.clear();
    last_reloc_frame_id = -1;
    tracking_history.reset();
    has_current_pose = false;
    publish_snapshot();
}

void Tracking::clean_vo_points() {
    for (auto &point : vo_points) {
        if (!point) {
            continue;
        }
        point->set_bad();
        point->delete_point();
    }
    vo_points.clear();
}

bool Tracking::pose_optimization(FramePtr &frame, const std::string &) {
    const Eigen::Matrix4d pose_before = frame->pose();
    PoseOptimizationResult result = Parameters::kOptimizationFrontEndUseGtsam
                                        ? OptimizerGTSAM::pose_optimization(frame, false, 10)
                                        : OptimizerG2o::pose_optimization(frame, false, 10);
    mean_pose_opt_chi2_error = result.mean_squared_error;
    pose_is_ok = result.is_ok;
    num_matched_map_points_in_last_pose_opt = result.num_valid_points;
    if (!pose_is_ok) {
        frame->update_pose(pose_before);
    }
    return pose_is_ok;
}

void Tracking::track_reference_frame(const FramePtr &f_ref_in, FramePtr &f_cur_in,
                                     const std::string &) {
    if (!f_ref_in || !f_cur_in) {
        pose_is_ok = false;
        return;
    }
    const std::vector<int> good_idxs = f_ref_in->get_matched_good_points_idxs();
    const cv::Mat des_ref = descriptor_rows(f_ref_in->des, good_idxs);
    const MatNx2f kps_ref = keypoint_rows(f_ref_in->kps, good_idxs);
    const float ratio = FeatureSharedResources::feature_match_ratio_test > 0.f
                            ? FeatureSharedResources::feature_match_ratio_test
                            : Parameters::kFeatureMatchDefaultRatioTest;
    const auto matches =
        match_descriptors(f_cur_in->img, f_ref_in->img, f_cur_in->des, des_ref, f_cur_in->kps, kps_ref,
                          ratio);
    idxs_cur = matches.first;
    idxs_ref = apply_match_indices(good_idxs, matches.second);
    if (idxs_cur.size() != idxs_ref.size()) {
        const size_t n = std::min(idxs_cur.size(), idxs_ref.size());
        idxs_cur.resize(n);
        idxs_ref.resize(n);
    }
    num_matched_kps = static_cast<int>(idxs_cur.size());
    if (FeatureSharedResources::oriented_features && num_matched_kps > 0) {
        std::vector<int> valid = RotationHistogram::filter_matches_with_histogram_orientation(
            idxs_cur, idxs_ref, angles_as_double(f_cur_in->angles), angles_as_double(f_ref_in->angles));
        const std::vector<float> distances =
            descriptor_distances_for_matches(f_cur_in->des, idxs_cur, f_ref_in->des, idxs_ref);
        std::vector<int> kept;
        const float max_distance = 0.5f * descriptor_distance_sigma;
        for (int k : valid) {
            if (k >= 0 && k < static_cast<int>(distances.size()) && distances[static_cast<size_t>(k)] <= max_distance) {
                kept.push_back(k);
            }
        }
        // Python uses np.intersect1d, which sorts by the original match index.
        std::sort(kept.begin(), kept.end());
        idxs_cur = apply_match_indices(idxs_cur, kept);
        idxs_ref = apply_match_indices(idxs_ref, kept);
        num_matched_kps = static_cast<int>(idxs_cur.size());
    }
    if (num_matched_kps < Parameters::kMinNumMatchedFeaturesSearchReferenceFrame) {
        pose_is_ok = false;
        return;
    }
    if ((f_cur_in->is_blurry || f_ref_in->is_blurry) && !f_cur_in->is_keyframe && !f_ref_in->is_keyframe &&
        num_matched_kps < Parameters::kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography) {
        auto homo = TrackingCore::find_homography_with_ransac(f_cur_in, f_ref_in, idxs_cur, idxs_ref);
        idxs_cur = std::get<1>(homo);
        idxs_ref = std::get<2>(homo);
        num_matched_kps = std::get<3>(homo);
    }
    if (num_matched_kps < Parameters::kMinNumMatchedFeaturesSearchReferenceFrame) {
        pose_is_ok = false;
        return;
    }
    if (Parameters::kUseEssentialMatrixFitting) {
        auto fitted = TrackingCore::estimate_pose_by_fitting_ess_mat(f_ref_in, f_cur_in, idxs_ref, idxs_cur);
        idxs_ref = std::get<0>(fitted);
        idxs_cur = std::get<1>(fitted);
        num_inliers = std::get<2>(fitted);
        num_matched_kps = static_cast<int>(idxs_cur.size());
    }
    if (Parameters::kUseDynamicDesDistanceTh) {
        descriptor_distance_sigma =
            dyn_config.update_descriptor_stats(f_ref_in, f_cur_in, idxs_ref, idxs_cur);
    }
    const float max_descriptor_distance =
        f_ref_in->is_keyframe ? 0.5f * descriptor_distance_sigma.load() : descriptor_distance_sigma.load();
    auto propagated = TrackingCore::propagate_map_point_matches(f_ref_in, f_cur_in, idxs_ref, idxs_cur,
                                                                max_descriptor_distance);
    const Eigen::Matrix4d pose_before = f_cur_in->pose();
    pose_optimization(f_cur_in, "match-frame");
    num_matched_map_points = f_cur_in->clean_outlier_map_points();
    if (!pose_is_ok || num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame) {
        f_cur_in->remove_frame_views(idxs_cur);
        f_cur_in->reset_points();
        pose_is_ok = false;
        f_cur_in->update_pose(pose_before);
    }
    (void)propagated;
}

void Tracking::track_keyframe(const FramePtr &keyframe, FramePtr &f_cur_in, const std::string &name) {
    if (f_ref) {
        f_cur_in->update_pose(f_ref->pose());
    }
    track_reference_frame(keyframe, f_cur_in, name);
}

void Tracking::track_previous_frame(const FramePtr &f_ref_in, FramePtr &f_cur_in) {
    bool failed = false;
    const bool use_projection =
        motion_model.is_ok && use_search_frame_by_projection() && use_motion_model();
    if (use_projection) {
        float search_radius = Parameters::kMaxReprojectionDistanceFrame;
        if (slam->sensor_type != SensorType::STEREO) {
            search_radius = Parameters::kMaxReprojectionDistanceFrameNonStereo;
        }
        f_cur_in->reset_points();
        auto searched = ProjectionMatcher::search_frame_by_projection(
            f_ref_in, f_cur_in, search_radius, descriptor_distance_sigma,
            Parameters::kMatchRatioTestFrameByProjection, slam->sensor_type == SensorType::MONOCULAR);
        idxs_ref = std::get<0>(searched);
        idxs_cur = std::get<1>(searched);
        num_matched_kps = static_cast<int>(idxs_cur.size());
        if (num_matched_kps < Parameters::kMinNumMatchedFeaturesSearchFrameByProjection) {
            f_cur_in->remove_frame_views(idxs_cur);
            f_cur_in->reset_points();
            searched = ProjectionMatcher::search_frame_by_projection(
                f_ref_in, f_cur_in, 2.f * search_radius, descriptor_distance_sigma,
                Parameters::kMatchRatioTestFrameByProjection, slam->sensor_type == SensorType::MONOCULAR);
            idxs_ref = std::get<0>(searched);
            idxs_cur = std::get<1>(searched);
            num_matched_kps = static_cast<int>(idxs_cur.size());
        }
        if ((f_cur_in->is_blurry || f_ref_in->is_blurry) &&
            num_matched_kps < Parameters::kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography) {
            auto homo = TrackingCore::find_homography_with_ransac(f_cur_in, f_ref_in, idxs_cur, idxs_ref);
            idxs_cur = std::get<1>(homo);
            idxs_ref = std::get<2>(homo);
            num_matched_kps = std::get<3>(homo);
        }
        if (num_matched_kps < Parameters::kMinNumMatchedFeaturesSearchFrameByProjection) {
            f_cur_in->remove_frame_views(idxs_cur);
            f_cur_in->reset_points();
            failed = true;
        } else {
            const Eigen::Matrix4d pose_before = f_cur_in->pose();
            pose_optimization(f_cur_in, "proj-frame-frame");
            num_matched_map_points = f_cur_in->clean_outlier_map_points();
            if (!pose_is_ok || num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame) {
                pose_is_ok = false;
                f_cur_in->update_pose(pose_before);
                failed = true;
            } else if (Parameters::kUseDynamicDesDistanceTh) {
                descriptor_distance_sigma =
                    dyn_config.update_descriptor_stats(f_ref_in, f_cur_in, idxs_ref, idxs_cur);
            }
        }
    }
    if (!use_projection || failed) {
        track_reference_frame(f_ref_in, f_cur_in, "match-frame-frame");
    }
}

void Tracking::update_local_map() {
    if (f_cur) {
        f_cur->clean_bad_map_points();
    }
    auto covisibles = slam->map->local_map->get_frame_covisibles(f_cur);
    kf_ref = std::get<0>(covisibles);
    local_keyframes = std::get<1>(covisibles);
    local_points = std::get<2>(covisibles);
    if (kf_ref && f_cur) {
        f_cur->kf_ref = kf_ref;
    }
}

void Tracking::track_local_map(FramePtr &f_cur_in) {
    if (!slam->map->local_map || slam->map->local_map->is_empty()) {
        return;
    }
    update_local_map();
    if (local_points.empty()) {
        pose_is_ok = false;
        return;
    }
    reproj_err_frame_map_sigma = Parameters::kMaxReprojectionDistanceMap;
    if (slam->sensor_type == SensorType::RGBD) {
        reproj_err_frame_map_sigma = Parameters::kMaxReprojectionDistanceMapRgbd;
    }
    if (f_cur_in->id < last_reloc_frame_id + 2) {
        reproj_err_frame_map_sigma = Parameters::kMaxReprojectionDistanceMapReloc;
    }
    auto searched = ProjectionMatcher::search_map_by_projection(
        local_points, f_cur_in, reproj_err_frame_map_sigma, descriptor_distance_sigma,
        Parameters::kMatchRatioTestMap, far_threshold_or_inf(far_points_threshold));
    const Eigen::Matrix4d pose_before = f_cur_in->pose();
    pose_optimization(f_cur_in, "proj-map-frame");
    num_matched_map_points = f_cur_in->update_map_points_statistics(slam->sensor_type);
    if (!pose_is_ok || num_matched_map_points < kNumMinInliersPoseOptimizationTrackLocalMap) {
        pose_is_ok = false;
        f_cur_in->update_pose(pose_before);
    }
    (void)searched;
}

bool Tracking::need_new_keyframe(const FramePtr &f_cur_in) {
    if (!slam->local_mapping || slam->local_mapping->is_stopped() ||
        slam->local_mapping->is_stop_requested()) {
        return false;
    }
    const int num_keyframes = slam->map->num_keyframes();
    if (f_cur_in->id < last_reloc_frame_id + max_frames_between_kfs_after_reloc &&
        num_keyframes > max_frames_between_kfs) {
        return false;
    }
    int n_min_obs = kNumMinObsForKeyFrameDefault;
    if (num_keyframes <= 2) {
        n_min_obs = 2;
    }
    num_kf_ref_tracked_points = kf_ref ? kf_ref->num_tracked_points(n_min_obs) : 0;
    const int num_f_cur_tracked_points = f_cur_in->num_matched_inlier_map_points();
    const bool is_local_mapping_idle = slam->local_mapping->is_idle();
    const int local_mapping_queue_size = slam->local_mapping->queue_size();

    int num_tracked_close = 0;
    int num_non_tracked_close = 0;
    std::vector<bool> tracked_mask;
    bool is_need_to_insert_close = false;
    if (slam->sensor_type != SensorType::MONOCULAR) {
        auto counted =
            TrackingCore::count_tracked_and_non_tracked_close_points(f_cur_in, slam->sensor_type);
        num_tracked_close = std::get<0>(counted);
        num_non_tracked_close = std::get<1>(counted);
        tracked_mask = std::get<2>(counted);
        is_need_to_insert_close =
            num_tracked_close < Parameters::kNumMinTrackedClosePointsForNewKfNonMonocular &&
            num_non_tracked_close > Parameters::kNumMaxNonTrackedClosePointsForNewKfNonMonocular;
    }
    double th_ref_ratio = Parameters::kThNewKfRefRatioStereo;
    if (num_keyframes < 2) {
        th_ref_ratio = 0.4;
    }
    if (slam->sensor_type == SensorType::MONOCULAR) {
        th_ref_ratio = Parameters::kThNewKfRefRatioMonocular;
    }
    if (!Parameters::kLocalMappingOnSeparateThread && slam->sensor_type != SensorType::MONOCULAR) {
        min_frames_between_kfs = 3;
    }
    const bool cond1a = kf_last && f_cur_in->id >= (kf_last->id + max_frames_between_kfs);
    const bool cond1b = kf_last && f_cur_in->id >= (kf_last->id + min_frames_between_kfs) &&
                        is_local_mapping_idle;
    const bool cond1c = slam->sensor_type != SensorType::MONOCULAR &&
                        (num_f_cur_tracked_points <
                             num_kf_ref_tracked_points * Parameters::kThNewKfRefRatioNonMonocular ||
                         is_need_to_insert_close);
    bool cond1d = false;
    if (Parameters::kUseFeatureCoverageControlForNewKf && camera()) {
        if (tracked_mask.empty()) {
            tracked_mask = f_cur_in->get_tracked_mask();
        }
        std::vector<int> tracked_idxs;
        for (int i = 0; i < static_cast<int>(tracked_mask.size()); ++i) {
            if (tracked_mask[static_cast<size_t>(i)]) {
                tracked_idxs.push_back(i);
            }
        }
        ImageGrid grid(camera()->width, camera()->height, 3, 2);
        grid.add_points(keypoint_rows(f_cur_in->kps, tracked_idxs));
        cond1d = grid.num_cells_uncovered(1) > 1;
    }
    const bool cond2 = (num_f_cur_tracked_points < num_kf_ref_tracked_points * th_ref_ratio ||
                        is_need_to_insert_close) &&
                       (num_f_cur_tracked_points > Parameters::kNumMinPointsForNewKf);
    bool cond3 = false;
    if (use_fov_centers_based_kf_generation && num_f_cur_tracked_points > Parameters::kNumMinPointsForNewKf) {
        std::vector<KeyFramePtr> close_kfs = local_keyframes;
        if (kf_last && std::find(close_kfs.begin(), close_kfs.end(), kf_last) == close_kfs.end()) {
            close_kfs.push_back(kf_last);
        }
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto &kf : close_kfs) {
            if (!kf) {
                continue;
            }
            min_dist = std::min(min_dist, (kf->fov_center_w - f_cur_in->fov_center_w).norm());
        }
        cond3 = min_dist > max_fov_centers_distance;
    }
    const bool condition_checks = ((cond1a || cond1b || cond1c || cond1d) && cond2) || cond3;
    if (!condition_checks) {
        return false;
    }
    if (is_local_mapping_idle) {
        return true;
    }
    slam->local_mapping->interrupt_optimization();
    if (slam->sensor_type == SensorType::MONOCULAR) {
        return local_mapping_queue_size <= 3;
    }
    return false;
}

void Tracking::create_new_keyframe(const FramePtr &f_cur_in, const cv::Mat &img,
                                   const cv::Mat &img_right, const cv::Mat &depth) {
    if (!slam->local_mapping->set_do_not_stop(true)) {
        return;
    }
    KeyFramePtr kf_new = std::make_shared<KeyFrame>(f_cur_in, img, img_right, depth);
    kf_last = kf_new;
    kf_ref = kf_new;
    f_cur_in->kf_ref = kf_new;
    slam->map->add_keyframe(kf_new);
    if (slam->sensor_type != SensorType::MONOCULAR) {
        const int added =
            TrackingCore::create_and_add_stereo_map_points_on_new_kf(f_cur, kf_new, slam->map, img);
        last_num_static_stereo_map_points = added;
        total_num_static_stereo_map_points += added;
    }
    slam->local_mapping->push_keyframe(kf_new, kf_new->img, kf_new->img_right, kf_new->depth_img);
    slam->local_mapping->set_do_not_stop(false);
}

bool Tracking::relocalize(const FramePtr &f_cur_in, const cv::Mat &img) {
    if (!slam->has_loop_closing()) {
        return false;
    }
    return slam->loop_closing_relocalize(f_cur_in, img);
}

void Tracking::create_vo_points_on_last_frame() {
    if (slam->sensor_type == SensorType::MONOCULAR || !kf_last || !f_ref || kf_last->id == f_ref->id ||
        f_ref->depths.empty()) {
        return;
    }
    if (!Parameters::kUseVisualOdometryPoints) {
        return;
    }
    auto created = TrackingCore::create_vo_points(f_ref, Parameters::kMaxNumVisualOdometryPoints);
    vo_points.insert(vo_points.end(), created.begin(), created.end());
}

void Tracking::wait_for_local_mapping(double timeout, bool) {
    if (!Parameters::kLocalMappingOnSeparateThread || !slam->local_mapping) {
        return;
    }
    if (Parameters::kTrackingWaitForLocalMappingToGetIdle) {
        if (!slam->local_mapping->is_idle()) {
            while (slam->local_mapping->queue_size() > 0 && !slam->local_mapping->is_idle()) {
                slam->local_mapping->wait_idle(std::nullopt);
            }
        }
        return;
    }
    if (num_matched_map_points > 0 &&
        num_matched_map_points < kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle) {
        if (!slam->local_mapping->is_idle() || slam->local_mapping->queue_size() > 0) {
            slam->local_mapping->wait_idle(timeout);
        }
        if (Parameters::kUseParallelProcessLBA && !slam->local_mapping->is_idle()) {
            slam->local_mapping->wait_idle(Parameters::kParallelLBAWaitIdleTimeout);
        }
    }
    if (slam->local_mapping->queue_size() > 0) {
        slam->local_mapping->wait_idle(timeout);
    }
}

void Tracking::update_tracking_history() {
    if (state == SlamState::OK && f_cur && f_cur->kf_ref) {
        const Eigen::Isometry3d Tcr = f_cur->isometry3d() * f_cur->kf_ref->isometry3d().inverse();
        tracking_history.relative_frame_poses.push_back(Tcr);
        tracking_history.kf_references.push_back(f_cur->kf_ref);
        tracking_history.timestamps.push_back(f_cur->timestamp);
        tracking_history.ids.push_back(f_cur->id);
    } else if (!tracking_history.relative_frame_poses.empty()) {
        tracking_history.relative_frame_poses.push_back(tracking_history.relative_frame_poses.back());
        tracking_history.kf_references.push_back(tracking_history.kf_references.back());
        tracking_history.timestamps.push_back(tracking_history.timestamps.back());
        tracking_history.ids.push_back(tracking_history.ids.back());
    }
    tracking_history.slam_states.push_back(state);
}

void Tracking::update_history() {
    if (!slam->map) {
        return;
    }
    FramePtr current = slam->map->get_frame(-1);
    if (!current) {
        return;
    }
    const Eigen::Matrix4d pose = current->pose();
    cur_R = pose.block<3, 3>(0, 0).transpose();
    cur_t = -cur_R * pose.block<3, 1>(0, 3);
    has_current_pose = true;
}

void Tracking::finish_tracked_frame(const FramePtr &f_cur_in, const cv::Mat &img,
                                    const cv::Mat &img_right, const cv::Mat &depth) {
    if (pose_is_ok) {
        track_local_map(f_cur);
    }
    if (pose_is_ok) {
        state = SlamState::OK;
    } else if (state == SlamState::OK) {
        state = SlamState::LOST;
    }
    if (f_cur_in->id <= last_reloc_frame_id + 1) {
        motion_model.is_ok = false;
    } else {
        motion_model.is_ok = pose_is_ok;
    }
    if (pose_is_ok) {
        const Eigen::Vector3d position = f_cur_in->position();
        const Eigen::Quaterniond orientation = f_cur_in->quaternion();
        motion_model.update_pose(f_cur_in->timestamp, position, orientation);
        f_cur_in->clean_vo_matches();
        clean_vo_points();
        const bool need_kf = need_new_keyframe(f_cur_in);
        if (need_kf) {
            create_new_keyframe(f_cur_in, img, img_right, depth);
        }
        f_cur_in->clean_outlier_map_points();
        if (need_kf && !Parameters::kLocalMappingOnSeparateThread) {
            slam->local_mapping->is_running = true;
            while (slam->local_mapping->queue_size() > 0) {
                slam->local_mapping->step();
                for (const auto &kf : slam->map->local_map->get_keyframes()) {
                    if (kf) {
                        kf->update_connections();
                    }
                }
            }
        }
    }
}

void Tracking::track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
                     std::optional<double> timestamp, const cv::Mat &mask, const cv::Mat &mask_right) {
    struct PublishOnExit {
        Tracking &tracking;
        ~PublishOnExit() { tracking.publish_snapshot(); }
    } publish_on_exit{*this};
    const auto time_start = std::chrono::steady_clock::now();
    if (!camera()) {
        throw std::runtime_error("Tracking::track: camera is not set");
    }
    if (img.rows != camera()->height || img.cols != camera()->width) {
        throw std::runtime_error("Tracking::track: image size does not match the camera");
    }
    slam->set_feature_tracker_num_features(state != SlamState::OK);

    bool is_blurry = false;
    float laplacian_var = 0.f;
    if (Parameters::kUseMotionBlurDection) {
        auto blur = detect_blur_laplacian(img, Parameters::kMotionBlurDetectionLalacianVarianceThreshold);
        is_blurry = blur.first;
        laplacian_var = blur.second;
    }

    const double stamp = timestamp.value_or(0.0);
    f_cur = std::make_shared<Frame>(camera(), img, img_right, depth, CameraPose(), -1, stamp, img_id,
                                    cv::Mat(), mask, mask_right);
    if (is_blurry) {
        f_cur->is_blurry = true;
        f_cur->laplacian_var = laplacian_var;
    }
    idxs_ref.clear();
    idxs_cur.clear();

    if (state == SlamState::NO_IMAGES_YET) {
        initializer.init(f_cur, img, img_right, depth);
        state = SlamState::NOT_INITIALIZED;
        return;
    }
    if (state == SlamState::NOT_INITIALIZED) {
        auto initialized = initializer.initialize(f_cur, img, img_right, depth);
        if (initialized.second) {
            InitializerOutput &result = initialized.first;
            KeyFramePtr kf_ref_init = result.kf_ref;
            KeyFramePtr kf_cur_init = result.kf_cur;
            slam->map->keyframe_origins.insert(kf_ref_init);
            FramePtr ref_frame = kf_ref_init;
            FramePtr cur_frame = kf_cur_init;
            slam->map->add_frame(ref_frame);
            slam->map->add_frame(cur_frame);
            slam->map->add_keyframe(kf_ref_init);
            slam->map->add_keyframe(kf_cur_init);
            kf_ref_init->init_observations();
            kf_cur_init->init_observations();
            slam->map->add_points(result.pts, std::nullopt, kf_cur_init, kf_ref_init, result.idxs_cur,
                                  result.idxs_ref, img, false, Parameters::kCosMaxParallax, std::nullopt);
            kf_ref_init->update_connections();
            kf_cur_init->update_connections();
            f_cur = kf_cur_init;
            f_cur->kf_ref = kf_ref_init;
            kf_ref = kf_cur_init;
            kf_last = kf_cur_init;
            slam->map->local_map->update(kf_ref);
            state = SlamState::OK;
            update_tracking_history();
            const Eigen::Vector3d position = kf_cur_init->position();
            const Eigen::Quaterniond orientation = kf_cur_init->quaternion();
            motion_model.update_pose(kf_cur_init->timestamp, position, orientation);
            motion_model.is_ok = false;
            initializer.reset();
            if (Parameters::kUseDynamicDesDistanceTh) {
                descriptor_distance_sigma = dyn_config.update_descriptor_stats(
                    kf_ref_init, kf_cur_init, result.idxs_ref, result.idxs_cur);
            }
            slam->loop_closing_add_keyframe(kf_ref_init, kf_ref_init->img);
            slam->loop_closing_add_keyframe(kf_cur_init, kf_cur_init->img);
            slam->volumetric_add_keyframe(kf_ref_init, kf_ref_init->img, kf_ref_init->img_right,
                                          kf_ref_init->depth_img);
            slam->volumetric_add_keyframe(kf_cur_init, kf_cur_init->img, kf_cur_init->img_right,
                                          kf_cur_init->depth_img);
            slam->semantic_push_keyframe(kf_ref_init, kf_ref_init->img, kf_ref_init->img_right,
                                         kf_ref_init->depth_img);
            slam->semantic_push_keyframe(kf_cur_init, kf_cur_init->img, kf_cur_init->img_right,
                                         kf_cur_init->depth_img);
        }
        return;
    }

    f_ref = slam->map->get_frame(-1);
    if (f_ref && f_ref->img_id != f_cur->img_id - 1) {
        std::cerr << "Tracking::track: reference img id " << f_ref->img_id
                  << " is not the frame before " << f_cur->img_id << std::endl;
    }
    slam->map->add_frame(f_cur);
    f_cur->kf_ref = kf_ref;
    pose_is_ok = false;

    if (slam->loop_closing_is_closing()) {
        slam->local_mapping->wait_idle(std::nullopt);
        slam->loop_closing_wait_if_closing();
    }
    wait_for_local_mapping(Parameters::kWaitForLocalMappingTimeout);

    {
        std::unique_lock<MapMutex> map_lock(slam->map->update_lock());
        bool pending_relocalize = false;
        if (state == SlamState::OK) {
            f_ref->check_replaced_map_points();
            if (use_motion_model() && motion_model.is_ok) {
                if (!tracking_history.relative_frame_poses.empty() && f_ref->kf_ref) {
                    f_ref->update_pose(tracking_history.relative_frame_poses.back() *
                                       f_ref->kf_ref->isometry3d());
                }
                create_vo_points_on_last_frame();
                const Eigen::Vector3d position = f_ref->position();
                const Eigen::Quaterniond orientation = f_ref->orientation();
                auto predicted = motion_model.predict_pose(timestamp, &position, &orientation);
                predicted_pose = predicted.first;
                f_cur->update_pose(predicted_pose->matrix());
            } else {
                f_cur->update_pose(f_ref->pose());
            }
            if (!motion_model.is_ok || f_cur->id < last_reloc_frame_id + 2) {
                track_keyframe(kf_ref, f_cur, "match-frame-keyframe");
            } else {
                track_previous_frame(f_ref, f_cur);
                if (!pose_is_ok) {
                    track_keyframe(kf_ref, f_cur, "match-frame-keyframe");
                }
            }
        } else {
            pending_relocalize = true;
        }

        bool relocalized = false;
        if (pending_relocalize) {
            if (state != SlamState::INIT_RELOCALIZE) {
                state = SlamState::RELOCALIZE;
            }
            map_lock.unlock();
            relocalized = relocalize(f_cur, img);
            map_lock.lock();
            if (relocalized) {
                last_reloc_frame_id = f_cur->id;
                state = SlamState::OK;
                pose_is_ok = true;
                kf_ref = f_cur->kf_ref;
                kf_last = kf_ref;
                if (kf_ref) {
                    slam->map->local_map->update(kf_ref);
                }
                motion_model.reset();
                const Eigen::Vector3d position = f_cur->position();
                const Eigen::Quaterniond orientation = f_cur->quaternion();
                motion_model.update_pose(timestamp, position, orientation);
            } else {
                pose_is_ok = false;
            }
        }
        finish_tracked_frame(f_cur, img, img_right, depth);
    }

    const bool need_reset =
        slam->reset_requested || (state == SlamState::LOST && slam->map->num_keyframes_session() <= 5);
    if (need_reset) {
        const SlamState state_before_reset = state;
        slam->reset_session();
        if (state_before_reset == SlamState::LOST && slam->map->is_reloaded()) {
            state = SlamState::INIT_RELOCALIZE;
            motion_model.is_ok = false;
        }
        return;
    }
    if (f_cur && !f_cur->kf_ref) {
        f_cur->kf_ref = kf_ref;
    }
    update_tracking_history();
    update_history();
    time_track = std::chrono::duration<double>(std::chrono::steady_clock::now() - time_start).count();
}

} // namespace pyslam
