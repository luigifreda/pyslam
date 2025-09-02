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

#include <Eigen/Dense>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

#include "camera.h"
#include "camera_pose.h"
#include "feature_shared_info.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pyslam {

// Forward declarations
class Camera;
class CameraPose;
class MapPoint;
class KeyFrame;

// Base object class for frame info management - matches Python FrameBase
// exactly
class FrameBase {
  protected:
    static std::atomic<int> _id; // shared frame counter
    static std::mutex _id_lock;
    mutable std::mutex _lock_pose;

  public:
    // Frame camera info
    Camera *camera;
    std::unique_ptr<CameraPose> _pose; // CameraPose() representing Tcw (pc = Tcw * pw)

    // Frame id management
    int id;
    double timestamp;
    int img_id;

    // Frame statistics
    float median_depth;           // median depth of the frame
    Eigen::Vector3d fov_center_c; // fov center 3D position w.r.t. camera
    Eigen::Vector3d fov_center_w; // fov center 3D position w.r.t world

    // Constructor
    FrameBase(Camera *camera, CameraPose *pose = nullptr, int id = -1, double timestamp = 0.0,
              int img_id = -1);

    // Destructor
    virtual ~FrameBase() = default;

    // Copy constructor and assignment
    FrameBase(const FrameBase &other);
    FrameBase &operator=(const FrameBase &other);

    // Static method
    static int next_id();

    // Properties - matches Python @property methods exactly
    int width() const;
    int height() const;
    const Eigen::Isometry3d &isometry3d() const; // pose as g2o.Isometry3d
    const Eigen::Matrix4d Tcw() const;
    const Eigen::Matrix4d Twc() const;
    const Eigen::Matrix3d Rcw() const;
    const Eigen::Matrix3d Rwc() const;
    const Eigen::Vector3d tcw() const;
    const Eigen::Vector3d Ow() const;
    const Eigen::Matrix4d pose() const;
    const Eigen::Quaterniond quaternion() const;  // g2o.Quaternion(), quaternion_cw
    const Eigen::Quaterniond orientation() const; // g2o.Quaternion(), quaternion_cw
    const Eigen::Vector3d position() const; // 3D vector tcw (world origin w.r.t. camera frame)

    void update_pose(const CameraPose &pose);
    void update_translation(const Eigen::Vector3d &tcw);
    void update_rotation_and_translation(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);

    Eigen::Vector3d transform_point(const Eigen::Vector3d &pw) const;
    MatNx3d transform_points(const MatNx3dRef points) const;

    std::pair<MatNxMd, VecNd> project_points(const MatNx3dRef points,
                                             bool do_stereo_project = false) const;
    std::pair<MatNxMd, VecNd> project_map_points(const std::vector<MapPoint *> &map_points,
                                                 bool do_stereo_project = false) const;
    std::pair<Eigen::Vector2d, double> project_point(const Eigen::Vector3d &pw) const;
    std::pair<Eigen::Vector2d, double> project_map_point(const MapPoint *map_point) const;

    bool is_in_image(const Eigen::Vector2d &uv, float z) const;
    std::vector<bool> are_in_image(const MatNx2dRef uvs, const VecNdRef zs) const;

    std::tuple<bool, Eigen::Vector2d, float> is_visible(const MapPoint *map_point) const;
    std::tuple<std::vector<bool>, MatNx2d, VecNd, VecNd>
    are_visible(const std::vector<MapPoint *> &map_points, bool do_stereo_project = false) const;

    // Comparison operators
    bool operator==(const FrameBase &rhs) const;
    bool operator<(const FrameBase &rhs) const;
    bool operator<=(const FrameBase &rhs) const;

    // Hash function - matches Python __hash__
    size_t hash() const;
};

// A Frame mainly collects keypoints, descriptors and their corresponding 3D
// points - matches Python Frame exactly
class Frame : public FrameBase, public std::enable_shared_from_this<Frame> {
  private:
    mutable std::mutex _lock_features;

  public:
    // Static variables
    static bool is_store_imgs;           // to store images when needed for debugging or
                                         // processing purposes
    static bool is_compute_median_depth; // to compute median depth when needed

    // Frame data
    bool is_keyframe;

    // Feature data arrays  (unpacked from array of
    // cv::KeyPoint())
    MatNx2d kps;                // left keypoint coordinates [Nx2]
    MatNx2d kps_r;              // right keypoint coordinates (extracted on right image) [Nx2]
    MatNx2d kpsu;               // [u]ndistorted keypoint coordinates [Nx2]
    MatNx2d kpsn;               // [n]ormalized keypoint coordinates [Nx2] (Kinv * [kp,1])
    cv::Mat kps_sem;            // [sem]antic keypoint information [NxD] where D is the
                                // semantic information length
    std::vector<int> octaves;   // keypoint octaves [Nx1]
    std::vector<int> octaves_r; // keypoint octaves [Nx1]
    std::vector<float> sizes;   // keypoint sizes [Nx1]
    std::vector<float> angles;  // keypoint angles [Nx1]
    cv::Mat des;                // keypoint descriptors [NxD] where D is the descriptor length
    cv::Mat des_r;              // right keypoint descriptors [NxD] where D is the descriptor
                                // length
    std::vector<float> depths;  // keypoint depths [Nx1]
    std::vector<float> kps_ur;  // corresponding right u-coordinates for left
                                // keypoints [Nx1] (computed with stereo matching
                                // and assuming rectified stereo images)

    // Map points information arrays
    std::vector<MapPoint *> points; // map points => self.points[idx] (if is not None) is the map
                                    // point matched with self.kps[idx]
    std::vector<bool> outliers;     // outliers flags for map points (reset and set by
                                    // pose_optimization())

    // Reference
    KeyFrame *kf_ref; // reference keyframe

    // Image data
    cv::Mat img;          // image (copy of img if available)
    cv::Mat img_right;    // right image (copy of img_right if available)
    cv::Mat depth_img;    // depth (copy of depth if available)
    cv::Mat semantic_img; // semantics (copy of semantic_img if available)

    // Statistics
    bool is_blurry;
    float laplacian_var;

    // KDTree
    std::unique_ptr<cv::flann::Index> _kd; // kdtree for fast-search of keypoints

    // Constructor
    Frame(Camera *camera, const cv::Mat &img = cv::Mat(), const cv::Mat &img_right = cv::Mat(),
          const cv::Mat &depth = cv::Mat(), CameraPose *pose = nullptr, int id = -1,
          double timestamp = 0.0, int img_id = -1, const cv::Mat &semantic_img = cv::Mat(),
          const std::map<std::string, void *> &frame_data_dict = {});

    // Destructor
    ~Frame() = default;

    // Copy constructor and assignment
    Frame(const Frame &other);
    Frame &operator=(const Frame &other);

    cv::flann::Index *kd() const;

    MapPoint *get_point_match(int idx) const;
    void set_point_match(MapPoint *p, int idx);
    void remove_point_match(int idx);
    void replace_point_match(MapPoint *p, int idx);
    void remove_point(MapPoint *p);
    void remove_frame_views(const std::vector<int> &idxs);
    void reset_points();
    std::vector<MapPoint *> get_points() const;
    std::vector<MapPoint *> get_matched_points() const;
    std::vector<int> get_unmatched_points_idxs() const;
    std::vector<MapPoint *> get_matched_inlier_points() const;
    std::vector<MapPoint *> get_matched_good_points() const;
    std::pair<std::vector<int>, std::vector<MapPoint *>> get_matched_good_points_with_idxs() const;

    int num_tracked_points(int minObs = 1) const;
    int num_matched_inlier_map_points() const;
    void update_map_points_statistics(int sensor_type = 0); // SensorType.MONOCULAR = 0
    int clean_outlier_map_points();
    void clean_bad_map_points();
    void clean_vo_matches();
    void check_replaced_map_points();

    void compute_stereo_from_rgbd(const cv::Mat &depth);
    void compute_stereo_matches(const cv::Mat &img, const cv::Mat &img_right);

    inline Eigen::Vector2d keypoint_undistorted(int idx) { return {kpsu(idx, 0), kpsu(idx, 1)}; }

    std::pair<Eigen::Vector3d, bool> unproject_point_3d(int idx, bool transform_in_world) const;
    std::pair<std::vector<Eigen::Vector3d>, std::vector<bool>>
    unproject_points_3d(const std::vector<int> &idxs, bool transform_in_world = false) const;

    float compute_points_median_depth(const std::vector<Eigen::Vector3d> &points3d = {},
                                      float percentile = 0.5f) const;

    void set_img_right(const cv::Mat &img_right);
    void set_depth_img(const cv::Mat &depth_img);
    void set_semantics(const cv::Mat &semantic_img);

    void ensure_contiguous_arrays();

  public:
    // Drawing methods
    cv::Mat draw_feature_trails(const cv::Mat &img, const std::vector<int> &kps_idxs,
                                int trail_max_length = 9) const;
    cv::Mat draw_all_feature_trails(const cv::Mat &img) const;

  public:
    // Serialization
    std::string to_json() const;
    static Frame from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                  const std::vector<Frame *> &frames,
                                  const std::vector<KeyFrame *> &keyframes);

    // Numpy serialization
    pybind11::tuple state_tuple() const;              // builds the versioned tuple
    void restore_from_state(const pybind11::tuple &); // fills this object from the tuple

  public:
    // Methods to perform feature detection
    void manage_features(const cv::Mat &img, const cv::Mat &img_right);

  protected:
    std::pair<FeatureDetectAndComputeOutput, FeatureDetectAndComputeOutput>
    detect_and_compute_features_parallel(const cv::Mat &img, const cv::Mat &img_right);

    template <typename T>
    void extract_depth_values(const cv::Mat_<T> &depth, std::vector<bool> &valid_depth_mask,
                              std::vector<float> &valid_depths);
};

} // namespace pyslam
