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
#include "frame.h"
#include "initializer.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "utils/cv_ops.h"
#include "utils/messages.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace pyslam {

// Constructor
Tracking::Tracking(Slam *slam)
    : slam_(slam), state_(SlamState::NO_IMAGES_YET),
      initializer_(std::make_unique<Initializer>(slam->get_sensor_type())),
      motion_model_(std::make_unique<MotionModel>()),
      dyn_config_(std::make_unique<SLAMDynamicConfig>(0.5)) // Default descriptor distance
      ,
      descriptor_distance_sigma_(0.5), reproj_err_frame_map_sigma_(2.0), max_frames_between_kfs_(1),
      max_frames_between_kfs_after_reloc_(1), min_frames_between_kfs_(0),
      far_points_threshold_(-1.0), use_fov_centers_based_kf_generation_(false),
      max_fov_centers_distance_(-1.0), num_matched_kps_(0), num_inliers_(0),
      num_matched_map_points_(0), num_matched_map_points_in_last_pose_opt_(0),
      num_kf_ref_tracked_points_(0), last_num_static_stereo_map_points_(0),
      total_num_static_stereo_map_points_(0),
      last_reloc_frame_id_(-std::numeric_limits<int>::max()), pose_is_ok_(false),
      mean_pose_opt_chi2_error_(0.0), predicted_pose_(Eigen::Isometry3d::Identity()),
      velocity_(Eigen::Vector3d::Zero()), f_cur_(nullptr), f_ref_(nullptr), kf_ref_(nullptr),
      kf_last_(nullptr), kid_last_BA_(-1), init_history_(true), t0_est_(Eigen::Vector3d::Zero()),
      t0_gt_(Eigen::Vector3d::Zero()), cur_R_(Eigen::Matrix3d::Identity()),
      cur_t_(Eigen::Vector3d::Zero()), gt_x_(0.0), gt_y_(0.0), gt_z_(0.0), time_track_(0.0) {

    // Initialize max frames between keyframes based on camera FPS
    if (slam_->get_camera()->fps > 0) {
        max_frames_between_kfs_ = static_cast<int>(slam_->get_camera()->fps);
        max_frames_between_kfs_after_reloc_ = max_frames_between_kfs_;
    }
}

// Reset method
void Tracking::reset() {
    MSG_INFO("Tracking: reset...");

    initializer_->reset();
    motion_model_->reset();

    state_ = SlamState::NO_IMAGES_YET;

    // Reset statistics
    num_matched_kps_ = 0;
    num_inliers_ = 0;
    num_matched_map_points_ = 0;
    num_matched_map_points_in_last_pose_opt_ = 0;
    num_kf_ref_tracked_points_ = 0;
    last_num_static_stereo_map_points_ = 0;
    total_num_static_stereo_map_points_ = 0;

    // Reset pose state
    pose_is_ok_ = false;
    mean_pose_opt_chi2_error_ = 0.0;
    predicted_pose_ = Eigen::Isometry3d::Identity();
    velocity_ = Eigen::Vector3d::Zero();

    // Reset frames
    f_cur_ = nullptr;
    f_ref_ = nullptr;
    idxs_cur_.clear();
    idxs_ref_.clear();

    // Reset keyframes
    kf_ref_ = nullptr;
    kf_last_ = nullptr;
    kid_last_BA_ = -1;

    // Reset local map
    local_keyframes_.clear();
    local_points_.clear();
    vo_points_.clear();

    // Reset tracking history
    tracking_history_.reset();

    // Reset trajectory history
    init_history_ = true;
    poses_.clear();
    pose_timestamps_.clear();
    t0_est_ = Eigen::Vector3d::Zero();
    t0_gt_ = Eigen::Vector3d::Zero();
    traj3d_est_.clear();
    traj3d_gt_.clear();

    // Reset current pose
    cur_R_ = Eigen::Matrix3d::Identity();
    cur_t_ = Eigen::Vector3d::Zero();
    gt_x_ = gt_y_ = gt_z_ = 0.0;
}

// Main tracking method
void Tracking::track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
                     double timestamp) {

    MSG_INFO_STREAM("Tracking " << static_cast<int>(slam_->get_sensor_type())
                                << ", img id: " << img_id << ", frame id: " << Frame::next_id()
                                << ", state: " << static_cast<int>(state_));

    time_start_ = std::chrono::high_resolution_clock::now();

    // Validate image size
    MSG_ASSERT(img.rows == slam_->get_camera()->height && img.cols == slam_->get_camera()->width,
               "Image size mismatch with camera parameters");

    // Create current frame
    f_cur_ = std::make_shared<Frame>(slam_->get_camera(), img, img_right, depth, CameraPose(), -1,
                                     timestamp, img_id);

    // Reset match indices
    idxs_cur_.clear();
    idxs_ref_.clear();

    // Process based on current state
    switch (state_) {
    case SlamState::NO_IMAGES_YET:
        process_initialization();
        break;

    case SlamState::NOT_INITIALIZED:
        process_initialization();
        break;

    case SlamState::OK:
    case SlamState::LOST:
    case SlamState::RELOCALIZE:
    case SlamState::INIT_RELOCALIZE:
        process_tracking();
        break;
    }

    // Finalize tracking
    finalize_tracking();

    // Update timing
    auto time_end = std::chrono::high_resolution_clock::now();
    time_track_ = std::chrono::duration<double>(time_end - time_start_).count();

    MSG_INFO_STREAM("Tracking elapsed time: " << time_track_);
}

// Pose optimization
std::pair<bool, double> Tracking::pose_optimization(const FramePtr &f_cur,
                                                    const std::string &name) {
    MSG_INFO_STREAM("Pose optimization " << name);

    CameraPose pose_before = f_cur->get_pose();

    // Perform pose optimization using g2o
    PoseOptimizationResult result = OptimizerG2o::pose_optimization(f_cur, false);

    mean_pose_opt_chi2_error_ = result.mean_squared_error;
    pose_is_ok_ = result.is_ok;
    num_matched_map_points_in_last_pose_opt_ = result.num_valid_points;

    MSG_INFO_STREAM("Error^2: " << mean_pose_opt_chi2_error_
                                << ", ok: " << static_cast<int>(pose_is_ok_));

    if (!pose_is_ok_) {
        // Reset pose if optimization failed
        f_cur->update_pose(pose_before);
    }

    return {pose_is_ok_, mean_pose_opt_chi2_error_};
}

// Track previous frame
void Tracking::track_previous_frame(const FramePtr &f_ref, const FramePtr &f_cur) {
    MSG_INFO("Tracking previous frame...");

    bool is_search_frame_by_projection_failure = false;
    bool use_search_frame_by_projection = motion_model_->is_ok;

    if (use_search_frame_by_projection) {
        MSG_INFO("Search frame by projection");

        double search_radius = 2.0; // Default reprojection distance
        if (slam_->get_sensor_type() == SensorType::RGBD) {
            search_radius = 1.5; // Smaller radius for RGBD
        }

        f_cur->reset_points();

        auto [idxs_ref, idxs_cur, num_found_map_pts] = tracking_utils::search_frame_by_projection(
            f_ref, f_cur, search_radius, descriptor_distance_sigma_, 0.75,
            slam_->get_sensor_type() == SensorType::MONOCULAR);

        num_matched_kps_ = static_cast<int>(idxs_cur.size());
        MSG_INFO_STREAM("Matched map points in prev frame: " << num_matched_kps_);

        // Check if we have enough matches
        if (num_matched_kps_ < 15) { // Minimum threshold
            f_cur->remove_frame_views(idxs_cur);
            f_cur->reset_points();
            is_search_frame_by_projection_failure = true;
            MSG_WARN_STREAM(
                "Not enough matches in search frame by projection: " << num_matched_kps_);
        } else {
            // Update descriptor statistics
            descriptor_distance_sigma_ =
                dyn_config_->update_descriptor_stats(f_ref, f_cur, idxs_ref, idxs_cur);

            // Store tracking info
            idxs_ref_ = idxs_ref;
            idxs_cur_ = idxs_cur;

            CameraPose pose_before_pos_opt = f_cur->get_pose();

            // Perform pose optimization
            pose_optimization(f_cur, "proj-frame-frame");

            // Update matched map points
            num_matched_map_points_ = f_cur->clean_outlier_map_points();

            if (!pose_is_ok_ ||
                num_matched_map_points_ < kNumMinInliersPoseOptimizationTrackFrame) {
                MSG_WARN_STREAM("Failure in tracking previous frame, matched map points: "
                                << num_matched_map_points_);
                pose_is_ok_ = false;
                f_cur->update_pose(pose_before_pos_opt);
                is_search_frame_by_projection_failure = true;
            }
        }
    }

    if (!use_search_frame_by_projection || is_search_frame_by_projection_failure) {
        MSG_INFO("Using frame-frame matching");
        track_reference_frame(f_ref, f_cur, "match-frame-frame");
    }
}

// Track reference frame
void Tracking::track_reference_frame(const FramePtr &f_ref, const FramePtr &f_cur,
                                     const std::string &name) {
    std::string frame_str = f_ref->is_keyframe ? "keyframe" : "frame";
    MSG_INFO_STREAM("Tracking reference " << frame_str << " " << f_ref->id);

    if (!f_ref) {
        pose_is_ok_ = false;
        MSG_ERROR("f_ref is None");
        return;
    }

    // Find keypoint matches between f_cur and f_ref
    MSG_INFO("Matching keypoints");

    // Get reference keypoints that correspond to map points
    std::vector<int> idxs_ref_map_points;
    for (int i = 0; i < static_cast<int>(f_ref->points.size()); ++i) {
        if (f_ref->points[i] && !f_ref->points[i]->is_bad()) {
            idxs_ref_map_points.push_back(i);
        }
    }

    if (idxs_ref_map_points.empty()) {
        pose_is_ok_ = false;
        MSG_WARN("No valid map points in reference frame");
        return;
    }

    // Extract descriptors and keypoints for matching
    cv::Mat des_ref = cv::Mat::zeros(static_cast<int>(idxs_ref_map_points.size()), f_ref->des.cols,
                                     f_ref->des.type());
    MatNx2f kps_ref = MatNx2f::Zero(static_cast<int>(idxs_ref_map_points.size()), 2);

    for (size_t i = 0; i < idxs_ref_map_points.size(); ++i) {
        int idx = idxs_ref_map_points[i];
        f_ref->des.row(idx).copyTo(des_ref.row(static_cast<int>(i)));
        kps_ref(i, 0) = f_ref->kps(idx, 0);
        kps_ref(i, 1) = f_ref->kps(idx, 1);
    }

    // Perform descriptor matching (simplified - would use proper matcher)
    std::vector<int> idxs_cur, idxs_ref;
    // This is a placeholder - in practice would use FeatureTrackerShared or similar
    // For now, we'll assume some matches are found

    num_matched_kps_ = static_cast<int>(idxs_cur.size());

    if (num_matched_kps_ < 15) { // Minimum threshold
        pose_is_ok_ = false;
        MSG_WARN_STREAM("Not enough matches: " << num_matched_kps_);
        return;
    }

    MSG_INFO_STREAM("Keypoints matched: " << num_matched_kps_);

    // Update descriptor statistics
    descriptor_distance_sigma_ =
        dyn_config_->update_descriptor_stats(f_ref, f_cur, idxs_ref, idxs_cur);

    // Propagate map point matches
    auto [num_found_map_pts, idx_ref_prop, idx_cur_prop] =
        tracking_utils::propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur,
                                                    descriptor_distance_sigma_);

    MSG_INFO_STREAM("Matched map points in reference frame: " << num_found_map_pts);

    // Store tracking info
    idxs_ref_ = idxs_ref;
    idxs_cur_ = idxs_cur;

    CameraPose pose_before_pos_opt = f_cur->get_pose();

    // Perform pose optimization
    pose_optimization(f_cur, name);

    // Update matched map points
    num_matched_map_points_ = f_cur->clean_outlier_map_points();

    if (!pose_is_ok_ || num_matched_map_points_ < kNumMinInliersPoseOptimizationTrackFrame) {
        f_cur->remove_frame_views(idxs_cur);
        f_cur->reset_points();
        MSG_WARN_STREAM("Failure in tracking reference "
                        << f_ref->id << ", matched map points: " << num_matched_map_points_);
        pose_is_ok_ = false;
        f_cur->update_pose(pose_before_pos_opt);
    }
}

// Track keyframe
void Tracking::track_keyframe(const KeyFramePtr &keyframe, const FramePtr &f_cur,
                              const std::string &name) {
    f_cur->update_pose(f_ref_->get_pose()); // Start from last frame pose
    track_reference_frame(keyframe, f_cur, name);
}

// Track local map
void Tracking::track_local_map(const FramePtr &f_cur) {
    if (slam_->get_map()->get_local_map()->is_empty()) {
        return;
    }

    MSG_INFO("Tracking local map...");

    update_local_map();

    if (local_points_.empty()) {
        pose_is_ok_ = false;
        return;
    }

    reproj_err_frame_map_sigma_ = 2.0; // Default reprojection distance
    if (slam_->get_sensor_type() == SensorType::RGBD) {
        reproj_err_frame_map_sigma_ = 1.5;
    }
    if (f_cur->id < last_reloc_frame_id_ + 2) {
        reproj_err_frame_map_sigma_ = 3.0; // Larger radius after relocalization
    }

    // Search for matches between local map points and unmatched keypoints
    auto [num_found_map_pts, matched_points_frame_idxs] = tracking_utils::search_map_by_projection(
        local_points_, f_cur, reproj_err_frame_map_sigma_, descriptor_distance_sigma_, 0.75,
        far_points_threshold_);

    MSG_INFO_STREAM("Matched map points in local map: "
                    << num_found_map_pts
                    << ", percentage: " << (100.0 * num_found_map_pts / local_points_.size()));

    CameraPose pose_before_pos_opt = f_cur->get_pose();

    // Perform pose optimization with all matched local map points
    pose_optimization(f_cur, "proj-map-frame");

    // Update map points statistics
    num_matched_map_points_ = f_cur->update_map_points_statistics(slam_->get_sensor_type());

    if (!pose_is_ok_ || num_matched_map_points_ < kNumMinInliersPoseOptimizationTrackLocalMap) {
        MSG_WARN_STREAM(
            "Failure in tracking local map, matched map points: " << num_matched_map_points_);
        pose_is_ok_ = false;
        f_cur->update_pose(pose_before_pos_opt);
    }
}

// Update local map
void Tracking::update_local_map() {
    f_cur_->clean_bad_map_points();

    auto [kf_ref, local_keyframes, local_points] =
        slam_->get_map()->get_local_map()->get_frame_covisibles(f_cur_);

    kf_ref_ = kf_ref;
    local_keyframes_ = local_keyframes;
    local_points_ = local_points;

    if (kf_ref_) {
        f_cur_->kf_ref = kf_ref_;
    }
}

// Need new keyframe check
bool Tracking::need_new_keyframe(const FramePtr &f_cur) {
    // Check if local mapping is stopped
    if (slam_->get_local_mapping()->is_stopped() ||
        slam_->get_local_mapping()->is_stop_requested()) {
        return false;
    }

    int num_keyframes = slam_->get_map()->num_keyframes();

    // Don't insert keyframes too close to last relocalization
    if (f_cur->id < last_reloc_frame_id_ + max_frames_between_kfs_after_reloc_ &&
        num_keyframes > max_frames_between_kfs_) {
        MSG_INFO_STREAM("Not inserting keyframe " << f_cur->id
                                                  << " because too close to last reloc frame "
                                                  << last_reloc_frame_id_);
        return false;
    }

    int nMinObs = kNumMinObsForKeyFrameDefault;
    if (num_keyframes <= 2) {
        nMinObs = 2;
    }

    int num_kf_ref_tracked_points = kf_ref_->num_tracked_points(nMinObs);
    int num_f_cur_tracked_points = f_cur->num_matched_inlier_map_points();

    MSG_INFO_STREAM("F(" << f_cur->id << ") matched points: " << num_f_cur_tracked_points << ", KF("
                         << kf_ref_->id << ") matched points: " << num_kf_ref_tracked_points);

    num_kf_ref_tracked_points_ = num_kf_ref_tracked_points;

    bool is_local_mapping_idle = slam_->get_local_mapping()->is_idle();
    int local_mapping_queue_size = slam_->get_local_mapping()->queue_size();

    MSG_INFO_STREAM("Local mapping idle: " << is_local_mapping_idle
                                           << ", queue size: " << local_mapping_queue_size);

    // Check conditions for new keyframe
    bool cond1a = f_cur->id >= (kf_last_->id + max_frames_between_kfs_);
    bool cond1b = (f_cur->id >= (kf_last_->id + min_frames_between_kfs_)) && is_local_mapping_idle;

    double thRefRatio = 0.75; // Default threshold
    if (num_keyframes < 2) {
        thRefRatio = 0.4;
    }
    if (slam_->get_sensor_type() == SensorType::MONOCULAR) {
        thRefRatio = 0.9;
    }

    bool cond2 = (num_f_cur_tracked_points < num_kf_ref_tracked_points * thRefRatio) &&
                 (num_f_cur_tracked_points > 15); // Minimum points threshold

    bool condition_checks = (cond1a || cond1b) && cond2;

    if (condition_checks) {
        if (is_local_mapping_idle) {
            return true;
        } else {
            if (slam_->get_sensor_type() == SensorType::MONOCULAR) {
                if (local_mapping_queue_size <= 3) {
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}

// Create new keyframe
void Tracking::create_new_keyframe(const FramePtr &f_cur, const cv::Mat &img,
                                   const cv::Mat &img_right, const cv::Mat &depth) {
    if (!slam_->get_local_mapping()->set_do_not_stop(true)) {
        return;
    }

    KeyFramePtr kf_new = std::make_shared<KeyFrame>(f_cur, img, img_right, depth);
    kf_last_ = kf_new;
    kf_ref_ = kf_new;
    f_cur->kf_ref = kf_new;

    MSG_INFO_STREAM("Adding new KF with id " << kf_new->id);

    // Add keyframe to map
    slam_->get_map()->add_keyframe(kf_new);

    // Create stereo map points if not monocular
    if (slam_->get_sensor_type() != SensorType::MONOCULAR) {
        create_and_add_stereo_map_points_on_new_kf(f_cur, kf_new, img);
    }

    // Push to local mapping
    slam_->get_local_mapping()->push_keyframe(kf_new, img, img_right, depth);

    slam_->get_local_mapping()->set_do_not_stop(false);
}

// Essential matrix pose estimation
std::pair<std::vector<int>, std::vector<int>>
Tracking::estimate_pose_by_fitting_ess_mat(const FramePtr &f_ref, const FramePtr &f_cur,
                                           const std::vector<int> &idxs_ref,
                                           const std::vector<int> &idxs_cur) {

    if (idxs_ref.empty() || idxs_cur.empty()) {
        MSG_WARN("idxs_ref or idxs_cur is empty");
        mask_match_ = cv::Mat::zeros(static_cast<int>(idxs_ref.size()), 1, CV_8U);
        num_inliers_ = 0;
        return {idxs_ref, idxs_cur};
    }

    // Extract normalized keypoints
    MatNx2f kps_ref_norm = MatNx2f::Zero(static_cast<int>(idxs_ref.size()), 2);
    MatNx2f kps_cur_norm = MatNx2f::Zero(static_cast<int>(idxs_cur.size()), 2);

    for (size_t i = 0; i < idxs_ref.size(); ++i) {
        kps_ref_norm(i, 0) = f_ref->kpsn(idxs_ref[i], 0);
        kps_ref_norm(i, 1) = f_ref->kpsn(idxs_ref[i], 1);
        kps_cur_norm(i, 0) = f_cur->kpsn(idxs_cur[i], 0);
        kps_cur_norm(i, 1) = f_cur->kpsn(idxs_cur[i], 1);
    }

    // Estimate essential matrix
    auto [Mrc, mask] = tracking_utils::estimate_pose_ess_mat(
        kps_ref_norm, kps_cur_norm, cv::USAC_MAGSAC, kRansacProb, kRansacThresholdNormalized);

    mask_match_ = mask;

    // Compute inverse transformation
    Eigen::Matrix4d Mcr = Mrc.inverse();
    Eigen::Matrix4d estimated_Tcw = Mcr * f_ref->get_pose().Tcw();

    // Remove outliers
    std::vector<int> valid_idxs;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i) == 1) {
            valid_idxs.push_back(i);
        }
    }

    num_inliers_ = static_cast<int>(valid_idxs.size());
    MSG_INFO_STREAM("Number of inliers: " << num_inliers_);

    std::vector<int> filtered_idxs_ref, filtered_idxs_cur;
    for (int idx : valid_idxs) {
        filtered_idxs_ref.push_back(idxs_ref[idx]);
        filtered_idxs_cur.push_back(idxs_cur[idx]);
    }

    if (num_inliers_ < kNumMinInliersEssentialMat) {
        MSG_WARN("Essential mat: not enough inliers!");
    } else {
        // Use estimated pose as initial guess
        Eigen::Matrix3d Rcw = estimated_Tcw.block<3, 3>(0, 0);
        Eigen::Vector3d tcw = f_ref->get_pose().Tcw().block<3, 1>(0, 3);
        f_cur->update_rotation_and_translation(Rcw, tcw);
    }

    return {filtered_idxs_ref, filtered_idxs_cur};
}

// Homography RANSAC
std::tuple<bool, std::vector<int>, std::vector<int>, int, int>
Tracking::find_homography_with_ransac(const FramePtr &f_cur, const FramePtr &f_ref,
                                      const std::vector<int> &idxs_cur,
                                      const std::vector<int> &idxs_ref, double reproj_threshold,
                                      int min_num_inliers) {

    if (idxs_cur.empty() || idxs_cur.size() != idxs_ref.size()) {
        MSG_WARN("find_homography_with_ransac: invalid input sizes");
        return {false, {}, {}, 0, 0};
    }

    // Extract keypoints
    std::vector<cv::Point2f> kps_cur, kps_ref;
    for (size_t i = 0; i < idxs_cur.size(); ++i) {
        kps_cur.emplace_back(f_cur->kps(idxs_cur[i], 0), f_cur->kps(idxs_cur[i], 1));
        kps_ref.emplace_back(f_ref->kps(idxs_ref[i], 0), f_ref->kps(idxs_ref[i], 1));
    }

    cv::Mat H, mask;
    H = cv::findHomography(kps_cur, kps_ref, cv::USAC_MAGSAC, reproj_threshold);

    int num_inliers = cv::countNonZero(mask);
    int num_outliers = static_cast<int>(idxs_cur.size()) - num_inliers;

    if (num_inliers < min_num_inliers) {
        return {false, {}, {}, 0, 0};
    }

    // Filter valid matches
    std::vector<int> valid_idxs_cur, valid_idxs_ref;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i) == 1) {
            valid_idxs_cur.push_back(idxs_cur[i]);
            valid_idxs_ref.push_back(idxs_ref[i]);
        }
    }

    int num_matched_kps = static_cast<int>(valid_idxs_cur.size());
    return {true, valid_idxs_cur, valid_idxs_ref, num_matched_kps, num_outliers};
}

// Create VO points on last frame
void Tracking::create_vo_points_on_last_frame() {
    if (slam_->get_sensor_type() == SensorType::MONOCULAR || !kf_last_ || !f_ref_ ||
        kf_last_->id == f_ref_->id || f_ref_->depths.empty()) {
        return;
    }

    MSG_INFO("Creating VO points...");

    // Find valid depths
    std::vector<int> valid_indices;
    for (int i = 0; i < static_cast<int>(f_ref_->depths.size()); ++i) {
        if (f_ref_->depths[i] > kMinDepth) {
            valid_indices.push_back(i);
        }
    }

    if (valid_indices.empty()) {
        return;
    }

    // Sort by depth
    std::sort(valid_indices.begin(), valid_indices.end(),
              [&](int a, int b) { return f_ref_->depths[a] < f_ref_->depths[b]; });

    // Configuration
    int max_num_vo_points = 100;
    double depth_threshold = f_ref_->camera->depth_threshold;

    // Select points to create
    std::vector<int> selected_indices;
    for (int idx : valid_indices) {
        if (selected_indices.size() >= max_num_vo_points)
            break;
        if (f_ref_->depths[idx] < depth_threshold || selected_indices.size() < max_num_vo_points) {
            selected_indices.push_back(idx);
        }
    }

    // Filter points that need new map points
    std::vector<int> final_indices;
    for (int idx : selected_indices) {
        auto p = f_ref_->points[idx];
        if (!p || p->num_observations() < 1) {
            final_indices.push_back(idx);
        }
    }

    // Create new map points
    auto [pts3d, pts3d_mask] = f_ref_->unproject_points_3d(final_indices, true);
    if (pts3d.empty()) {
        return;
    }

    Vec3b color(0, 0, 255); // Red color for VO points

    for (size_t i = 0; i < final_indices.size(); ++i) {
        if (!pts3d_mask[i])
            continue;

        MapPointPtr mp = std::make_shared<MapPoint>(pts3d[i], color, f_ref_, final_indices[i]);
        f_ref_->points[final_indices[i]] = mp;
        vo_points_.push_back(mp);
    }

    MSG_INFO_STREAM("Added " << vo_points_.size() << " new VO points");
}

// Clean VO points
void Tracking::clean_vo_points() {
    for (auto &p : vo_points_) {
        p->set_bad();
        p->delete_point();
    }
    vo_points_.clear();
}

// Create and add stereo map points on new keyframe
void Tracking::create_and_add_stereo_map_points_on_new_kf(const FramePtr &f, const KeyFramePtr &kf,
                                                          const cv::Mat &img) {
    if (slam_->get_sensor_type() == SensorType::MONOCULAR || kf->depths.empty()) {
        return;
    }

    // Find valid depths and sort them
    std::vector<std::pair<float, int>> valid_depths_and_idxs;
    for (int i = 0; i < static_cast<int>(kf->depths.size()); ++i) {
        if (kf->depths[i] > kMinDepth) {
            valid_depths_and_idxs.emplace_back(kf->depths[i], i);
        }
    }

    if (valid_depths_and_idxs.empty()) {
        MSG_WARN("No valid depths found for stereo map point creation");
        return;
    }

    std::sort(valid_depths_and_idxs.begin(), valid_depths_and_idxs.end());

    // Extract sorted indices
    std::vector<int> sorted_idx_values;
    for (const auto &pair : valid_depths_and_idxs) {
        sorted_idx_values.push_back(pair.second);
    }

    int N = 100; // Maximum number of points to create
    double depth_threshold = kf->camera->depth_threshold;

    // Select points to create
    std::vector<int> selected_indices;
    for (size_t i = 0; i < sorted_idx_values.size() && selected_indices.size() < N; ++i) {
        int idx = sorted_idx_values[i];
        if (kf->depths[idx] < depth_threshold || selected_indices.size() < N) {
            selected_indices.push_back(idx);
        }
    }

    // Filter points that need new map points
    std::vector<int> final_indices;
    for (int idx : selected_indices) {
        auto p = kf->points[idx];
        if (!p || p->num_observations() < 1) {
            final_indices.push_back(idx);
        }
    }

    // Reset points in original frame
    for (int idx : final_indices) {
        f->points[idx] = nullptr;
    }

    // Create 3D points and add to map
    auto [pts3d, pts3d_mask] = f->unproject_points_3d(final_indices, true);
    int num_added_points =
        slam_->get_map()->add_stereo_points(pts3d, pts3d_mask, f, kf, final_indices, img);

    last_num_static_stereo_map_points_ = num_added_points;
    total_num_static_stereo_map_points_ += num_added_points;
}

// Update tracking history
void Tracking::update_tracking_history() {
    if (state_ == SlamState::OK) {
        Eigen::Isometry3d isometry3d_Tcr =
            f_cur_->isometry3d() * f_cur_->kf_ref->isometry3d().inverse();
        tracking_history_.relative_frame_poses.push_back(isometry3d_Tcr);
        tracking_history_.kf_references.push_back(kf_ref_);
        tracking_history_.timestamps.push_back(f_cur_->timestamp);
        tracking_history_.ids.push_back(f_cur_->id);
    } else {
        if (!tracking_history_.relative_frame_poses.empty()) {
            tracking_history_.relative_frame_poses.push_back(
                tracking_history_.relative_frame_poses.back());
            tracking_history_.kf_references.push_back(tracking_history_.kf_references.back());
            tracking_history_.timestamps.push_back(tracking_history_.timestamps.back());
            tracking_history_.ids.push_back(tracking_history_.ids.back());
        }
    }
    tracking_history_.slam_states.push_back(state_);
}

// Update history
void Tracking::update_history() {
    FramePtr f_cur = slam_->get_map()->get_frame(-1);
    CameraPose f_cur_pose = f_cur->get_pose();

    cur_R_ = f_cur_pose.Rcw().transpose();
    cur_t_ = -cur_R_ * f_cur_pose.tcw();

    if (init_history_) {
        t0_est_ = cur_t_;
        if (gt_x_ != 0.0 || gt_y_ != 0.0 || gt_z_ != 0.0) {
            t0_gt_ = Eigen::Vector3d(gt_x_, gt_y_, gt_z_);
        }
    }

    if (t0_est_ != Eigen::Vector3d::Zero()) {
        Eigen::Vector3d p = cur_t_ - t0_est_;
        traj3d_est_.push_back(p);

        if (t0_gt_ != Eigen::Vector3d::Zero()) {
            traj3d_gt_.push_back(
                Eigen::Vector3d(gt_x_ - t0_gt_.x(), gt_y_ - t0_gt_.y(), gt_z_ - t0_gt_.z()));
        }

        poses_.push_back(poseRt(cur_R_, p));
        pose_timestamps_.push_back(f_cur->timestamp);
    }
}

// Wait for local mapping
void Tracking::wait_for_local_mapping(double timeout) {
    // Simplified implementation - would check local mapping status
    // and wait if necessary
}

// Relocalize
bool Tracking::relocalize(const FramePtr &f_cur, const cv::Mat &img) {
    MSG_INFO_STREAM("Relocalizing frame id: " << f_cur->id);

    if (slam_->get_loop_closing()) {
        return slam_->get_loop_closing()->relocalize(f_cur, img);
    } else {
        MSG_WARN("No loop closing / relocalize method set!");
        return false;
    }
}

// Process initialization
void Tracking::process_initialization() {
    if (state_ == SlamState::NO_IMAGES_YET) {
        // Push first frame to initializer
        initializer_->init(f_cur_, f_cur_->img, f_cur_->img_right, f_cur_->depth_img);
        state_ = SlamState::NOT_INITIALIZED;
        return;
    }

    if (state_ == SlamState::NOT_INITIALIZED) {
        // Try to initialize
        auto [initializer_output, is_ok] =
            initializer_->initialize(f_cur_, f_cur_->img, f_cur_->img_right, f_cur_->depth_img);

        if (is_ok) {
            KeyFramePtr kf_ref = initializer_output.kf_ref;
            KeyFramePtr kf_cur = initializer_output.kf_cur;

            // Add initialized frames to map
            slam_->get_map()->keyframe_origins.insert(kf_ref);
            slam_->get_map()->add_frame(kf_ref);
            slam_->get_map()->add_frame(kf_cur);
            slam_->get_map()->add_keyframe(kf_ref);
            slam_->get_map()->add_keyframe(kf_cur);

            kf_ref->init_observations();
            kf_cur->init_observations();

            // Add points to map
            auto [new_pts_count, _, __] = slam_->get_map()->add_points(
                initializer_output.pts, std::nullopt, kf_cur, kf_ref, initializer_output.idxs_cur,
                initializer_output.idxs_ref, f_cur_->img, false);

            MSG_INFO_STREAM("Map initialized with KFs " << kf_ref->id << ", " << kf_cur->id
                                                        << " and " << new_pts_count
                                                        << " new map points");

            // Update covisibility graph
            kf_ref->update_connections();
            kf_cur->update_connections();

            // Update tracking info
            f_cur_ = kf_cur;
            f_cur_->kf_ref = kf_ref;
            kf_ref_ = kf_cur;
            kf_last_ = kf_cur;
            slam_->get_map()->get_local_map()->update(kf_ref_);
            state_ = SlamState::OK;

            update_tracking_history();
            motion_model_->update_pose(kf_cur->timestamp, kf_cur->position(), kf_cur->quaternion());
            motion_model_->is_ok = false; // Reset after initialization

            initializer_->reset();

            if (true) { // kUseDynamicDesDistanceTh
                descriptor_distance_sigma_ = dyn_config_->update_descriptor_stats(
                    kf_ref, kf_cur, initializer_output.idxs_ref, initializer_output.idxs_cur);
            }
        }
    }
}

// Process tracking
void Tracking::process_tracking() {
    // Get previous frame as reference
    f_ref_ = slam_->get_map()->get_frame(-1);
    MSG_ASSERT(f_ref_->img_id == f_cur_->img_id - 1, "Frame ID mismatch");

    // Add current frame to map
    slam_->get_map()->add_frame(f_cur_);
    f_cur_->kf_ref = kf_ref_;

    // Reset pose state
    pose_is_ok_ = false;

    // Wait for local mapping if needed
    wait_for_local_mapping();

    // Process based on state
    if (state_ == SlamState::OK) {
        // Check for map point replacements
        f_ref_->check_replaced_map_points();

        // Set initial pose guess
        if (motion_model_->is_ok) {
            MSG_INFO("Using motion model for pose prediction");
            f_ref_->update_pose(tracking_history_.relative_frame_poses.back() *
                                f_ref_->kf_ref->isometry3d());

            if (true) { // kUseVisualOdometryPoints
                create_vo_points_on_last_frame();
            }

            // Predict pose using motion model
            auto [predicted_pose, is_prediction_ok] = motion_model_->predict_pose(
                f_cur_->timestamp, f_ref_->position(), f_ref_->orientation());
            f_cur_->update_pose(predicted_pose);
        } else {
            MSG_INFO("Setting f_cur.pose <-- f_ref.pose");
            f_cur_->update_pose(f_ref_->get_pose());
        }

        // Track based on motion model status
        if (!motion_model_->is_ok || f_cur_->id < last_reloc_frame_id_ + 2) {
            MSG_ASSERT(kf_ref_, "Reference keyframe is null");
            track_keyframe(kf_ref_, f_cur_);
        } else {
            track_previous_frame(f_ref_, f_cur_);

            if (!pose_is_ok_) {
                track_keyframe(kf_ref_, f_cur_);
            }
        }
    } else {
        // SLAM is NOT OK - try relocalization
        if (state_ != SlamState::INIT_RELOCALIZE) {
            state_ = SlamState::RELOCALIZE;
        }

        if (relocalize(f_cur_, f_cur_->img)) {
            if (state_ != SlamState::INIT_RELOCALIZE) {
                last_reloc_frame_id_ = f_cur_->id;
            }
            state_ = SlamState::OK;
            pose_is_ok_ = true;
            kf_ref_ = f_cur_->kf_ref;
            kf_last_ = kf_ref_;
            MSG_INFO_STREAM("Relocalization successful, last reloc frame id: " << f_cur_->id);
        } else {
            pose_is_ok_ = false;
            MSG_WARN("Relocalization failed");
            if (!slam_->get_loop_closing()) {
                MSG_WARN("No loop closing / relocalize method set!");
            }
        }
    }

    // Track local map if pose is OK
    if (pose_is_ok_) {
        track_local_map(f_cur_);
    }

    // Update SLAM state
    if (pose_is_ok_) {
        state_ = SlamState::OK;
    } else {
        if (state_ == SlamState::OK) {
            state_ = SlamState::LOST;
            MSG_WARN("Tracking failure");
        }
    }

    // Update motion model state
    motion_model_->is_ok = pose_is_ok_;

    if (pose_is_ok_) {
        // Update motion model
        motion_model_->update_pose(f_cur_->timestamp, f_cur_->position(), f_cur_->quaternion());

        // Clean VO matches and points
        f_cur_->clean_vo_matches();
        clean_vo_points();

        // Check if we need a new keyframe
        bool need_new_kf = need_new_keyframe(f_cur_);

        if (need_new_kf) {
            MSG_INFO("NEW KF");
            create_new_keyframe(f_cur_, f_cur_->img, f_cur_->img_right, f_cur_->depth_img);
            MSG_INFO_STREAM("New keyframe created: " << f_cur_->id);
        }

        // Clean outliers after keyframe generation
        f_cur_->clean_outlier_map_points();

        if (need_new_kf) {
            // Process local mapping if not on separate thread
            if (!true) { // kLocalMappingOnSeparateThread
                slam_->get_local_mapping()->is_running = true;
                while (slam_->get_local_mapping()->queue_size() > 0) {
                    slam_->get_local_mapping()->step();
                    for (const auto &kf : slam_->get_map()->get_local_map()->get_keyframes()) {
                        kf->update_connections();
                    }
                }
            }
        }
    }
}

// Finalize tracking
void Tracking::finalize_tracking() {
    // Check for reset
    check_for_reset();

    if (f_cur_->kf_ref == nullptr) {
        f_cur_->kf_ref = kf_ref_;
    }

    update_tracking_history();
    update_history();

    MSG_INFO_STREAM("Map: " << slam_->get_map()->num_points() << " points, "
                            << slam_->get_map()->num_keyframes() << " keyframes");
}

// Check for reset
void Tracking::check_for_reset() {
    bool need_reset = slam_->is_reset_requested() ||
                      (state_ == SlamState::LOST && slam_->get_map()->num_keyframes_session() <= 5);

    if (need_reset) {
        MSG_WARN("Tracking: SLAM resetting...");
        SlamState state_before_reset = state_;
        slam_->reset_session();

        if (state_before_reset == SlamState::LOST && slam_->get_map()->is_reloaded()) {
            state_ = SlamState::INIT_RELOCALIZE;
            motion_model_->is_ok = false;
        }
        MSG_WARN("Tracking: SLAM reset done");
    }
}

} // namespace pyslam