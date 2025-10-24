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

#include "camera.h"
#include "camera_pose.h"
#include "config_parameters.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "motion_model.h"
#include "optimizer_g2o.h"
#include "slam_commons.h"
#include "slam_dynamic_config.h"
#include "smart_pointers.h"

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef USE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

namespace pyslam {

// Forward declarations
class Slam;
class Initializer;

// Tracking history structure - matches Python TrackingHistory
struct TrackingHistory {
    std::vector<Eigen::Isometry3d>
        relative_frame_poses;               // relative frame poses w.r.t reference keyframes
    std::vector<KeyFramePtr> kf_references; // list of reference keyframes
    std::vector<double> timestamps;         // list of frame timestamps
    std::vector<int> ids;                   // list of frame ids
    std::vector<SlamState> slam_states;     // list of slam states

    void reset() {
        relative_frame_poses.clear();
        kf_references.clear();
        timestamps.clear();
        ids.clear();
        slam_states.clear();
    }
};

// Main Tracking class - matches Python Tracking class
class Tracking {
  public:
    // Constructor
    explicit Tracking(Slam *slam);

    // Destructor
    ~Tracking() = default;

    // Delete copy constructor and assignment
    Tracking(const Tracking &) = delete;
    Tracking &operator=(const Tracking &) = delete;

    // Core tracking method - matches Python track() method
    void track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
               double timestamp = 0.0);

    // Reset method
    void reset();

    // State management
    SlamState get_state() const { return state_; }
    void set_state(SlamState state) { state_ = state; }

    // Pose optimization
    std::pair<bool, double> pose_optimization(const FramePtr &f_cur, const std::string &name = "");

    // Tracking methods
    void track_previous_frame(const FramePtr &f_ref, const FramePtr &f_cur);
    void track_reference_frame(const FramePtr &f_ref, const FramePtr &f_cur,
                               const std::string &name = "");
    void track_keyframe(const KeyFramePtr &keyframe, const FramePtr &f_cur,
                        const std::string &name = "match-frame-keyframe");
    void track_local_map(const FramePtr &f_cur);

    // Keyframe management
    bool need_new_keyframe(const FramePtr &f_cur);
    void create_new_keyframe(const FramePtr &f_cur, const cv::Mat &img,
                             const cv::Mat &img_right = cv::Mat(),
                             const cv::Mat &depth = cv::Mat());

    // Local map management
    void update_local_map();

    // Essential matrix pose estimation
    std::pair<std::vector<int>, std::vector<int>>
    estimate_pose_by_fitting_ess_mat(const FramePtr &f_ref, const FramePtr &f_cur,
                                     const std::vector<int> &idxs_ref,
                                     const std::vector<int> &idxs_cur);

    // Homography RANSAC
    std::tuple<bool, std::vector<int>, std::vector<int>, int, int>
    find_homography_with_ransac(const FramePtr &f_cur, const FramePtr &f_ref,
                                const std::vector<int> &idxs_cur, const std::vector<int> &idxs_ref,
                                double reproj_threshold = 5.0, int min_num_inliers = 15);

    // VO points management
    void create_vo_points_on_last_frame();
    void clean_vo_points();

    // Stereo map points creation
    void create_and_add_stereo_map_points_on_new_kf(const FramePtr &f, const KeyFramePtr &kf,
                                                    const cv::Mat &img);

    // History management
    void update_tracking_history();
    void update_history();

    // Local mapping synchronization
    void wait_for_local_mapping(double timeout = 5.0);

    // Relocalization
    bool relocalize(const FramePtr &f_cur, const cv::Mat &img);

    // Getters for current state
    FramePtr get_current_frame() const { return f_cur_; }
    FramePtr get_reference_frame() const { return f_ref_; }
    KeyFramePtr get_reference_keyframe() const { return kf_ref_; }
    KeyFramePtr get_last_keyframe() const { return kf_last_; }

    // Statistics
    int get_num_matched_kps() const { return num_matched_kps_; }
    int get_num_inliers() const { return num_inliers_; }
    int get_num_matched_map_points() const { return num_matched_map_points_; }
    bool is_pose_ok() const { return pose_is_ok_; }
    double get_mean_pose_opt_chi2_error() const { return mean_pose_opt_chi2_error_; }

    // Configuration
    void set_far_points_threshold(double threshold) { far_points_threshold_ = threshold; }
    void set_use_fov_centers_based_kf_generation(bool use) {
        use_fov_centers_based_kf_generation_ = use;
    }
    void set_max_fov_centers_distance(double distance) { max_fov_centers_distance_ = distance; }

  private:
    // Core SLAM reference
    Slam *slam_;

    // State management
    SlamState state_;

    // Initializer
    std::unique_ptr<Initializer> initializer_;

    // Motion model
    std::unique_ptr<MotionModel> motion_model_;

    // Dynamic configuration
    std::unique_ptr<SLAMDynamicConfig> dyn_config_;

    // Configuration parameters
    double descriptor_distance_sigma_;
    double reproj_err_frame_map_sigma_;
    int max_frames_between_kfs_;
    int max_frames_between_kfs_after_reloc_;
    int min_frames_between_kfs_;
    double far_points_threshold_;
    bool use_fov_centers_based_kf_generation_;
    double max_fov_centers_distance_;

    // Tracking statistics
    int num_matched_kps_;
    int num_inliers_;
    int num_matched_map_points_;
    int num_matched_map_points_in_last_pose_opt_;
    int num_kf_ref_tracked_points_;
    int last_num_static_stereo_map_points_;
    int total_num_static_stereo_map_points_;
    int last_reloc_frame_id_;

    // Pose estimation
    bool pose_is_ok_;
    double mean_pose_opt_chi2_error_;
    Eigen::Isometry3d predicted_pose_;
    Eigen::Vector3d velocity_;

    // Current and reference frames
    FramePtr f_cur_;
    std::vector<int> idxs_cur_;
    FramePtr f_ref_;
    std::vector<int> idxs_ref_;

    // Keyframes
    KeyFramePtr kf_ref_;
    KeyFramePtr kf_last_;
    int kid_last_BA_;

    // Local map
    std::vector<KeyFramePtr> local_keyframes_;
    std::vector<MapPointPtr> local_points_;
    std::vector<MapPointPtr> vo_points_;

    // Tracking history
    TrackingHistory tracking_history_;

    // History for trajectory
    bool init_history_;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<double> pose_timestamps_;
    Eigen::Vector3d t0_est_;
    Eigen::Vector3d t0_gt_;
    std::vector<Eigen::Vector3d> traj3d_est_;
    std::vector<Eigen::Vector3d> traj3d_gt_;

    // Current pose
    Eigen::Matrix3d cur_R_;
    Eigen::Vector3d cur_t_;
    double gt_x_, gt_y_, gt_z_;

    // Matching mask
    cv::Mat mask_match_;

    // Timing
    std::chrono::high_resolution_clock::time_point time_start_;
    double time_track_;

    // Helper methods
    void update_motion_model();
    void check_for_reset();
    void process_initialization();
    void process_tracking();
    void process_relocalization();
    void finalize_tracking();
};

} // namespace pyslam