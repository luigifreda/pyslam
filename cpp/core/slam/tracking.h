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

#include "frame.h"
#include "initializer.h"
#include "keyframe.h"
#include "motion_model.h"
#include "slam_commons.h"
#include "slam_dynamic_config.h"

#include <Eigen/Geometry>
#include <atomic>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace pyslam {

class Slam;

struct TrackingHistory {
    std::vector<Eigen::Isometry3d> relative_frame_poses;
    std::vector<KeyFramePtr> kf_references;
    std::vector<double> timestamps;
    std::vector<int> ids;
    std::vector<SlamState> slam_states;

    void reset() {
        relative_frame_poses.clear();
        kf_references.clear();
        timestamps.clear();
        ids.clear();
        slam_states.clear();
    }
};

// Same control flow as pyslam/slam/tracking.py. TrackingCore stays the static helper.
class Tracking {
  public:
    explicit Tracking(Slam *slam);

    void reset();
    void track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
               std::optional<double> timestamp, const cv::Mat &mask, const cv::Mat &mask_right);
    bool pose_optimization(FramePtr &frame, const std::string &name);
    void track_previous_frame(const FramePtr &f_ref_in, FramePtr &f_cur_in);
    void track_reference_frame(const FramePtr &f_ref_in, FramePtr &f_cur_in, const std::string &name);
    void track_keyframe(const FramePtr &keyframe, FramePtr &f_cur_in, const std::string &name);
    void update_local_map();
    void track_local_map(FramePtr &f_cur_in);
    bool need_new_keyframe(const FramePtr &f_cur_in);
    void create_new_keyframe(const FramePtr &f_cur_in, const cv::Mat &img, const cv::Mat &img_right,
                             const cv::Mat &depth);
    bool relocalize(const FramePtr &f_cur_in, const cv::Mat &img);
    void create_vo_points_on_last_frame();
    void wait_for_local_mapping(double timeout, bool check_on_exit = false);
    void update_tracking_history();
    void update_history();
    void clean_vo_points();

    CameraPtr camera() const;

    // Publish only the small viewer snapshot; never hold this lock across Python calls.
    struct Snapshot {
        FramePtr f_cur;
        KeyFramePtr kf_ref;
        std::optional<Eigen::Isometry3d> predicted_pose;
        Eigen::Matrix3d cur_R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d cur_t = Eigen::Vector3d::Zero();
        bool has_current_pose = false;
        int num_matched_kps = 0;
        int num_inliers = 0;
        int num_matched_map_points = 0;
        int num_kf_ref_tracked_points = 0;
        int last_num_static_stereo_map_points = 0;
        int total_num_static_stereo_map_points = 0;
        double mean_pose_opt_chi2_error = 0;
        double time_track = 0;
    };
    Snapshot snapshot() const;
    void publish_snapshot();

    Slam *slam = nullptr;
    Initializer initializer;
    MotionModel motion_model;
    SLAMDynamicConfig dyn_config;
    std::atomic<float> descriptor_distance_sigma{0.f};
    float reproj_err_frame_map_sigma = 0.f;

    int max_frames_between_kfs = 1;
    int max_frames_between_kfs_after_reloc = 1;
    int min_frames_between_kfs = 0;

    std::optional<double> far_points_threshold;
    bool use_fov_centers_based_kf_generation = false;
    double max_fov_centers_distance = -1.0;

    std::atomic<SlamState> state{SlamState::NO_IMAGES_YET};
    int num_matched_kps = 0;
    int num_inliers = 0;
    int num_matched_map_points = 0;
    int num_matched_map_points_in_last_pose_opt = 0;
    int num_kf_ref_tracked_points = 0;
    int last_num_static_stereo_map_points = 0;
    int total_num_static_stereo_map_points = 0;
    bool pose_is_ok = false;
    double mean_pose_opt_chi2_error = 0.0;
    std::optional<Eigen::Isometry3d> predicted_pose;
    FramePtr f_cur;
    FramePtr f_ref;
    std::vector<int> idxs_cur;
    std::vector<int> idxs_ref;
    KeyFramePtr kf_ref;
    KeyFramePtr kf_last;
    std::vector<KeyFramePtr> local_keyframes;
    std::vector<MapPointPtr> local_points;
    std::vector<MapPointPtr> vo_points;
    int last_reloc_frame_id = -1;
    TrackingHistory tracking_history;

    Eigen::Matrix3d cur_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d cur_t = Eigen::Vector3d::Zero();
    bool has_current_pose = false;
    double time_track = 0.0;

  private:
    mutable std::mutex snapshot_mutex_;
    Snapshot snapshot_;
    static constexpr int kNumMinInliersPoseOptimizationTrackFrame = 10;
    static constexpr int kNumMinInliersPoseOptimizationTrackLocalMap = 20;
    static constexpr int kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = 50;
    static constexpr int kNumMinObsForKeyFrameDefault = 3;

    bool use_motion_model() const;
    bool use_search_frame_by_projection() const;
    void finish_tracked_frame(const FramePtr &f_cur_in, const cv::Mat &img, const cv::Mat &img_right,
                              const cv::Mat &depth);
};

} // namespace pyslam
