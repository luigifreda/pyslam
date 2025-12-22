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
#include "feature_shared_resources.h"

#include "ckdtree_eigen.h"
#include "smart_pointers.h"
#include "utils/inheritable_shared_from_this.h"

#include <Eigen/Dense>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef USE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

namespace pyslam {

// Forward declarations
class Camera;
class CameraPose;
class MapPoint;
class KeyFrame;
class TrackingCore;
class PointsProxy;

// Base object class for frame info management - matches Python FrameBase
// exactly
class FrameBase {

    friend class TrackingCore;

  protected:
    static std::atomic<int> _id; // shared frame counter
    static std::mutex _id_lock;

    static const int _invalid_size; // -1

  protected:
    using pose_mutex_type = std::mutex;
    using pose_lock_guard_type = std::lock_guard<pose_mutex_type>;
    mutable pose_mutex_type _lock_pose;

  public:
    // Frame camera info
    CameraPtr camera = nullptr;
    CameraPose _pose; // CameraPose() representing Tcw (pc = Tcw * pw)

    // Frame id management
    int id = -1;
    double timestamp = std::numeric_limits<double>::quiet_NaN();
    int img_id = -1;

    // Frame statistics
    float median_depth = -1.0f;                             // median depth of the frame
    Eigen::Vector3d fov_center_c = Eigen::Vector3d::Zero(); // fov center 3D position w.r.t. camera
    Eigen::Vector3d fov_center_w = Eigen::Vector3d::Zero(); // fov center 3D position w.r.t world

  public:
    // Constructor
    FrameBase(const CameraPtr &camera, const CameraPose &pose = CameraPose(), int id = -1,
              double timestamp = 0.0, int img_id = -1);

    explicit FrameBase(int id) : id(id) {}

    // Destructor
    virtual ~FrameBase() { camera = nullptr; }

    // Delete copy constructor, assignment, move constructor, assignment
    FrameBase(const FrameBase &other) = delete;
    FrameBase &operator=(const FrameBase &other) = delete;
    FrameBase(FrameBase &&other) = delete;
    FrameBase &operator=(FrameBase &&other) = delete;

    void copy_from(const FrameBase &other);
    void reset();

    CameraPtr get_camera() const;
    void set_camera(CameraPtr &camera_ptr);
    void reset_camera() { camera = nullptr; }
    friend void bind_frame(pybind11::module &m);

    // Static methods
    static int next_id();
    static void set_id(int id);

    // Properties - matches Python @property methods exactly
    const int &width() const { return camera ? camera->width : _invalid_size; }
    const int &height() const { return camera ? camera->height : _invalid_size; }

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
    void update_pose(const Eigen::Isometry3d &isometry3d);
    void update_pose(const Eigen::Matrix4d &Tcw);
    void update_translation(const Eigen::Vector3d &tcw);
    void update_rotation_and_translation(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);
    void update_rotation_and_translation_no_lock_(const Eigen::Matrix3d &Rcw,
                                                  const Eigen::Vector3d &tcw);

    template <typename Scalar> Vec3<Scalar> transform_point(Vec3Ref<Scalar> pw) const;
    template <typename Scalar> MatNx3<Scalar> transform_points(MatNx3Ref<Scalar> points) const;

    template <typename Scalar>
    std::pair<MatNxM<Scalar>, VecN<Scalar>> project_points(MatNx3Ref<Scalar> points,
                                                           bool do_stereo_project = false) const;

    template <typename Scalar>
    std::pair<MatNxM<Scalar>, VecN<Scalar>>
    project_map_points(const std::vector<MapPointPtr> &map_points,
                       bool do_stereo_project = false) const;

    template <typename Scalar>
    std::pair<VecN<Scalar>, Scalar> project_point(Vec3Ref<Scalar> pw,
                                                  bool do_stereo_project = false) const;
    template <typename Scalar>
    std::pair<VecN<Scalar>, Scalar> project_map_point(const MapPointPtr &map_point,
                                                      bool do_stereo_project = false) const;

    template <typename Scalar> bool is_in_image(Vec2Ref<Scalar> uv, Scalar z) const {
        return camera->is_in_image(uv, z);
    }

    template <typename Scalar>
    std::vector<bool> are_in_image(MatNxMRef<Scalar> uvs, VecNRef<Scalar> zs) const {
        return camera->are_in_image(uvs, zs);
    }

    template <typename Scalar>
    std::tuple<bool, Vec2<Scalar>, Scalar> is_visible(const MapPointPtr &map_point) const;

    // clang-format off
    // input: a list of map points
    // output: [Nx1] array of visibility flags,
    //         [Nx2] array of projections (u,v) or [Nx3] array of stereo image points (u,v,ur) in case do_stereo_projet=True,
    //         [Nx1] array of depths,
    //         [Nx1] array of distances PO
    // check a) points are in image b) good view angle c) good distance range
    // clang-format on
    template <typename Scalar>
    std::tuple<std::vector<bool>, MatNxM<Scalar>, VecN<Scalar>, VecN<Scalar>>
    are_visible(const std::vector<MapPointPtr> &map_points, bool do_stereo_project = false) const;

    // Comparison operators
    bool operator==(const FrameBase &rhs) const;
    bool operator<(const FrameBase &rhs) const;
    bool operator<=(const FrameBase &rhs) const;

    // Hash function - matches Python __hash__
    size_t hash() const;
};

// A Frame mainly collects keypoints, descriptors and their corresponding 3D
// points - matches Python Frame exactly
class Frame : public FrameBase, public inheritable_enable_shared_from_this<Frame> {
  protected:
    mutable std::mutex _lock_features;
    mutable std::mutex _lock_kd;
    mutable std::mutex _lock_semantics;

  protected:
    friend class TrackingCore;
    friend class PointsProxy;
    friend void bind_frame(pybind11::module &m);

  public:
    // Static variables
    static bool is_store_imgs;           // to store images when needed for debugging or
                                         // processing purposes
    static bool is_compute_median_depth; // to compute median depth when needed

  public:
    // Frame data
    bool is_keyframe = false;

    // Feature data arrays
    MatNx2f kps;   // left keypoint coordinates [Nx2]
    MatNx2f kps_r; // right keypoint coordinates (extracted on right image) [Nx2]
    MatNx2f kpsu;  // [u]ndistorted keypoint coordinates [Nx2]
    MatNx2f kpsn;  // [n]ormalized keypoint coordinates [Nx2] (Kinv * [kp,1])

    std::vector<int> octaves;   // keypoint octaves [Nx1]
    std::vector<int> octaves_r; // keypoint octaves [Nx1]
    std::vector<float> sizes;   // keypoint sizes [Nx1]
    std::vector<float> angles;  // keypoint angles [Nx1]

    cv::Mat des;   // keypoint descriptors [NxD] where D is the descriptor length
    cv::Mat des_r; // right keypoint descriptors [NxD] where D is the descriptor
    // length
    cv::Mat
        kps_sem; // [sem]antic keypoint information [NxD] where D  the semantic information length

    std::vector<float> depths; // keypoint depths [Nx1]
    std::vector<float> kps_ur; // corresponding right u-coordinates for left
                               // keypoints [Nx1] (computed with stereo matching
                               // and assuming rectified stereo images)

    // Map points information arrays
    std::vector<MapPointPtr> points; // map points => self.points[idx] (if is not None) is the map
                                     // point matched with self.kps[idx]
    std::vector<bool> outliers;      // outliers flags for map points (reset and set by
                                     // pose_optimization())

    // Reference
    KeyFramePtr kf_ref; // reference keyframe

    // Image data
    cv::Mat img;                    // image (copy of img if available)
    cv::Mat img_right;              // right image (copy of img_right if available)
    cv::Mat depth_img;              // depth (copy of depth if available)
    cv::Mat semantic_img;           // semantics (copy of semantic_img if available)
    cv::Mat semantic_instances_img; // semantic instances, type CV_32S (int32)

    // Statistics
    bool is_blurry = false;
    float laplacian_var = 0.0f;

    // KDTree
    std::shared_ptr<cKDTree2f> _kd;

  protected:
    // Temporary storage for ID-based data during deserialization
    std::vector<int> _points_id_data;
    int _kf_ref_id = -1;

  public:
    // Constructor
    Frame(const CameraPtr &camera, const cv::Mat &img = cv::Mat(),
          const cv::Mat &img_right = cv::Mat(), const cv::Mat &depth = cv::Mat(),
          const CameraPose &pose = CameraPose(), int id = -1, double timestamp = 0.0,
          int img_id = -1, const cv::Mat &semantic_img = cv::Mat(),
          const pyslam::FrameDataDict &frame_data_dict = {});

    explicit Frame(int id) : FrameBase(id) {}

    // Destructor
    virtual ~Frame() { reset(); }

    // Delete copy constructor, assignment, move constructor, assignment
    Frame(const Frame &other) = delete;
    Frame &operator=(const Frame &other) = delete;
    Frame(Frame &&other) = delete;
    Frame &operator=(Frame &&other) = delete;

    void copy_from(const Frame &other);
    void reset();

    const std::shared_ptr<cKDTree2f> &kd();

    MapPointPtr get_point_match(int idx) const;
    void set_point_match(MapPointPtr p, int idx); // no reference passing here!
    void remove_point_match(int idx);
    void replace_point_match(MapPointPtr &p, int idx);
    void remove_point(MapPointPtr p);

    bool is_stereo_observation(int idx) const { return kps_ur.size() > 0 && kps_ur[idx] >= 0; }

    void remove_frame_views(const std::vector<int> &idxs);
    void reset_points();
    std::vector<MapPointPtr> get_points() const;
    std::vector<MapPointPtr> get_matched_points() const;
    std::vector<int> get_matched_points_idxs() const;
    std::vector<int> get_unmatched_points_idxs() const;
    std::pair<std::vector<MapPointPtr>, std::vector<int>> get_matched_inlier_points() const;
    std::vector<MapPointPtr> get_matched_good_points() const;
    std::vector<int> get_matched_good_points_idxs() const;
    std::vector<std::pair<MapPointPtr, int>> get_matched_good_points_and_idxs() const;

    int num_tracked_points(int minObs = 1) const;
    int num_matched_inlier_map_points() const;
    std::vector<bool> get_tracked_mask() const;
    int update_map_points_statistics(const SensorType &sensor_type = SensorType::MONOCULAR);
    int clean_outlier_map_points();
    void clean_bad_map_points();
    void clean_vo_matches();
    void check_replaced_map_points();

    void compute_stereo_from_rgbd(const cv::Mat &depth);
    void compute_stereo_matches(const cv::Mat &img, const cv::Mat &img_right);

    inline Eigen::Vector2d keypoint_undistorted(int idx) { return {kpsu(idx, 0), kpsu(idx, 1)}; }

    template <typename Scalar>
    std::pair<Vec3<Scalar>, bool> unproject_point_3d(int idx, bool transform_in_world) const;

    template <typename Scalar>
    std::pair<std::vector<Vec3<Scalar>>, std::vector<bool>>
    unproject_points_3d(const std::vector<int> &idxs, bool transform_in_world = false) const;

    template <typename Scalar>
    Scalar compute_points_median_depth(MatNx3Ref<Scalar> points3d = MatNx3<Scalar>(),
                                       const Scalar percentile = 0.5f) const;

    void set_img_right(const cv::Mat &img_right);
    void set_depth_img(const cv::Mat &depth_img);
    void set_semantics(const cv::Mat &semantic_img);
    void set_semantic_instances(const cv::Mat &semantic_instances_img);
    bool is_semantics_available() const;
    void update_points_semantics(void *semantic_fusion_method = nullptr);

    void ensure_contiguous_arrays();

    // Cleanup method for proper shutdown
    void clear_references() {
        // points.clear();
        // kf_ref = nullptr;
        reset();
    }

  public:
    // Drawing methods
    template <bool with_level_radius>
    cv::Mat draw_feature_trails_(const cv::Mat &img, const std::vector<int> &kps_idxs,
                                 int trail_max_length) const;
    cv::Mat draw_feature_trails(const cv::Mat &img, const std::vector<int> &kps_idxs,
                                const bool with_level_radius, int trail_max_length = 16) const;
    cv::Mat draw_all_feature_trails(const cv::Mat &img, const bool with_level_radius,
                                    int trail_max_length = 16) const;

  public:
    // Serialization
    std::string to_json() const;
    static FramePtr from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                  const std::vector<FramePtr> &frames,
                                  const std::vector<KeyFramePtr> &keyframes);

#ifdef USE_PYTHON
    // Numpy serialization
    pybind11::tuple state_tuple(bool need_lock = true) const; // builds the versioned tuple
    void restore_from_state(const pybind11::tuple &,
                            bool need_lock = true); // fills this object from the tuple
#endif

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

struct FrameIdCompare {
    bool operator()(const Frame *a, const Frame *b) const noexcept { return a->id < b->id; }
    bool operator()(const FramePtr &a, const FramePtr &b) const noexcept { return a->id < b->id; }
};

} // namespace pyslam
