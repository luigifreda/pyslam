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

#include "frame.h"
#include "camera.h"
#include "camera_pose.h"
#include "config_parameters.h"
#include "feature_shared_resources.h"
#include "map_point.h"
#include "utils/color_helpers.h"
#include "utils/drawing.h"
#include "utils/features.h"
#include "utils/messages.h"

#include <algorithm>
#include <cmath>
#include <future>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h> // for gil_scoped_acquire and gil_scoped_release
#endif

namespace pyslam {

// NOTE: This is just a convenience representation of the feature radius for drawing purposes.
inline constexpr auto kDrawFeatureRadius = []() {
    std::array<int, 100> a{};
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<int>(i) * 5;
    return a;
}();

// Static member definitions for FrameBase
std::atomic<int> FrameBase::_id{0};
std::mutex FrameBase::_id_lock;
const int FrameBase::_invalid_size = -1;

// FrameBase Implementation
FrameBase::FrameBase(const CameraPtr &camera, const CameraPose &pose, int id, double timestamp, int img_id)
    : camera(camera)
    , _pose(pose)
    , id(id == -1 ? []() { std::lock_guard<std::mutex> lock(_id_lock); return _id++; }() : id)
    , timestamp(timestamp)
    , img_id(img_id)
    , median_depth(std::numeric_limits<float>::quiet_NaN())
    , fov_center_c(Eigen::Vector3d::Zero())
    , fov_center_w(Eigen::Vector3d::Zero()) {
}

// Copy constructor and assignment operator deleted due to mutex

void FrameBase::copy_from(const FrameBase &other) {
    if (this == &other) {
        return;
    }
    camera = other.camera; // shallow copy

    {
        CameraPose other_pose;
        {
            pose_lock_guard_type other_lock(other._lock_pose);
            other_pose = other._pose;
        }
        {
            pose_lock_guard_type lock(_lock_pose);
            _pose = other_pose; // deep copy
        }
    }

    id = other.id;
    timestamp = other.timestamp;
    img_id = other.img_id;
    median_depth = other.median_depth;
    fov_center_c = other.fov_center_c;
    fov_center_w = other.fov_center_w;
}

void FrameBase::reset() {
    camera = nullptr;
    {
        pose_lock_guard_type lock(_lock_pose);
        _pose = CameraPose(); // deep copy
    }
    id = -1;
    timestamp = std::numeric_limits<double>::quiet_NaN();
    img_id = -1;
    median_depth = std::numeric_limits<float>::quiet_NaN();
    fov_center_c = Eigen::Vector3d::Zero();
    fov_center_w = Eigen::Vector3d::Zero();
}

CameraPtr FrameBase::get_camera() const { return camera; }

void FrameBase::set_camera(CameraPtr &camera_ptr) { camera = camera_ptr; }

int FrameBase::next_id() {
    std::lock_guard<std::mutex> lock(_id_lock);
    return _id;
}

void FrameBase::set_id(int id) {
    std::lock_guard<std::mutex> lock(_id_lock);
    _id = id;
}

const Eigen::Isometry3d &FrameBase::isometry3d() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.isometry3d();
}

const Eigen::Matrix4d FrameBase::Tcw() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.get_matrix();
}

const Eigen::Matrix4d FrameBase::Twc() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.get_inverse_matrix();
}

const Eigen::Matrix3d FrameBase::Rcw() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.get_rotation_matrix();
}

const Eigen::Matrix3d FrameBase::Rwc() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.get_inverse_rotation_matrix();
}

const Eigen::Vector3d FrameBase::tcw() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.position();
}

const Eigen::Vector3d FrameBase::Ow() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.Ow();
}

const Eigen::Matrix4d FrameBase::pose() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.get_matrix(); // Tcw
}

const Eigen::Quaterniond FrameBase::quaternion() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.quaternion();
}

const Eigen::Quaterniond FrameBase::orientation() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.orientation();
}

const Eigen::Vector3d FrameBase::position() const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.position();
}

void FrameBase::update_pose(const CameraPose &pose) {
    pose_lock_guard_type lock(_lock_pose);
    _pose.set(pose.isometry3d());
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        fov_center_w = _pose.Rwc() * fov_center_c + _pose.Ow();
    }
}

void FrameBase::update_pose(const Eigen::Isometry3d &isometry3d) {
    pose_lock_guard_type lock(_lock_pose);
    _pose.set(isometry3d);
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        fov_center_w = _pose.Rwc() * fov_center_c + _pose.Ow();
    }
}

void FrameBase::update_pose(const Eigen::Matrix4d &Tcw) {
    pose_lock_guard_type lock(_lock_pose);
    _pose.set_from_matrix(Tcw);
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        fov_center_w = _pose.Rwc() * fov_center_c + _pose.Ow();
    }
}

void FrameBase::update_translation(const Eigen::Vector3d &tcw) {
    pose_lock_guard_type lock(_lock_pose);
    _pose.set_translation(tcw);
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        fov_center_w = _pose.Rwc() * fov_center_c + _pose.Ow();
    }
}

void FrameBase::update_rotation_and_translation_no_lock_(const Eigen::Matrix3d &Rcw,
                                                         const Eigen::Vector3d &tcw) {
    _pose.set_from_rotation_and_translation(Rcw, tcw);
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        fov_center_w = _pose.Rwc() * fov_center_c + _pose.Ow();
    }
}

void FrameBase::update_rotation_and_translation(const Eigen::Matrix3d &Rcw,
                                                const Eigen::Vector3d &tcw) {
    pose_lock_guard_type lock(_lock_pose);
    update_rotation_and_translation_no_lock_(Rcw, tcw);
}

template <typename Scalar> Vec3<Scalar> FrameBase::transform_point(Vec3Ref<Scalar> pw) const {
    pose_lock_guard_type lock(_lock_pose);
    return _pose.Rcw().template cast<Scalar>() * pw +
           _pose.tcw().template cast<Scalar>(); // p w.r.t. camera
}

template <typename Scalar>
MatNx3<Scalar> FrameBase::transform_points(MatNx3Ref<Scalar> points) const {
    pose_lock_guard_type lock(_lock_pose);
    const Mat3<Scalar> Rcw_mat = _pose.Rcw().template cast<Scalar>();
    const Vec3<Scalar> tcw_vec = _pose.tcw().template cast<Scalar>();

    // Transform: (Rcw @ points.T + tcw).T
    MatNx3<Scalar> result = (Rcw_mat * points.transpose()).transpose();
    result.rowwise() += tcw_vec.transpose();
    return result;
}

// clang-format off
// project an [Nx3] array of map point vectors on this frame
// out: [Nx2] image projections (u,v) or [Nx3] array of stereo projections (u,v,ur) in case do_stereo_projet=True,
//      [Nx1] array of map point depths
// clang-format on
template <typename Scalar>
std::pair<MatNxM<Scalar>, VecN<Scalar>> FrameBase::project_points(MatNx3Ref<Scalar> points,
                                                                  bool do_stereo_project) const {
    auto pcs = transform_points(points);
    if (do_stereo_project) {
        return camera->project_stereo(pcs);
    } else {
        return camera->project(pcs);
    }
}

template <typename Scalar>
std::pair<MatNxM<Scalar>, VecN<Scalar>>
FrameBase::project_map_points(const std::vector<MapPointPtr> &map_points,
                              bool do_stereo_project) const {
    MatNx3<Scalar> points_in;
    points_in.resize(map_points.size(), 3);
    int r = 0;
    for (int i = 0; i < map_points.size(); ++i) {
        const auto &mp = map_points[i];
        if (mp) {
            points_in.row(r) = mp->pt().template cast<Scalar>();
            r++;
        }
    }
    points_in.conservativeResize(r, 3);
    return project_points<Scalar>(points_in, do_stereo_project);
}

template <typename Scalar>
std::pair<VecN<Scalar>, Scalar> FrameBase::project_point(Vec3Ref<Scalar> pw,
                                                         bool do_stereo_project) const {
    const Vec3<Scalar> pc = transform_point<Scalar>(pw); // p w.r.t. camera
    if (do_stereo_project) {
        auto [proj, z] = camera->project_point_stereo(pc);
        VecN<Scalar> vec_result(3);
        vec_result(0) = proj(0); // u
        vec_result(1) = proj(1); // v
        vec_result(2) = proj(2); // ur
        return std::make_pair(std::move(vec_result), z);
    } else {
        auto [proj, z] = camera->project_point(pc);
        VecN<Scalar> vec_result(2);
        vec_result(0) = proj(0); // u
        vec_result(1) = proj(1); // v
        return std::make_pair(std::move(vec_result), z);
    }
}

template <typename Scalar>
std::pair<VecN<Scalar>, Scalar> FrameBase::project_map_point(const MapPointPtr &map_point,
                                                             bool do_stereo_project) const {
    return project_point<Scalar>(map_point->pt().template cast<Scalar>(), do_stereo_project);
}

template <typename Scalar>
std::tuple<bool, Vec2<Scalar>, Scalar> FrameBase::is_visible(const MapPointPtr &map_point) const {
    const auto [uv, z] = project_map_point<Scalar>(map_point);
    const Vec3<Scalar> PO = map_point->pt().template cast<Scalar>() - Ow().template cast<Scalar>();

    if (!is_in_image<Scalar>(uv, z)) {
        return std::make_tuple(false, uv.template cast<Scalar>(), z);
    }

    const Scalar dist3D = PO.norm();
    // point depth must be inside the scale pyramid of the image
    if (dist3D < map_point->min_distance() || dist3D > map_point->max_distance()) {
        return std::make_tuple(false, uv.template cast<Scalar>(), z);
    }
    // viewing angle must be less than 60 deg
    if (PO.dot(map_point->get_normal().template cast<Scalar>()) <
        Parameters::kViewingCosLimitForPoint * dist3D) {
        return std::make_tuple(false, uv.template cast<Scalar>(), z);
    }
    return std::make_tuple(true, uv.template cast<Scalar>(), z);
}

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
FrameBase::are_visible(const std::vector<MapPointPtr> &map_points, bool do_stereo_project) const {
    // Handle empty input
    if (map_points.empty()) {
        return std::make_tuple(std::vector<bool>(), MatNx2<Scalar>(), VecN<Scalar>(),
                               VecN<Scalar>());
    }

    // Let's make the computations in double precision and then cast to Scalar at the end
    MatNx3<double> points_in;
    MatNx3<double> normals_in;
    VecN<double> min_dists_in;
    VecN<double> max_dists_in;

    const int num_points = map_points.size();
    points_in.resize(num_points, 3);
    normals_in.resize(num_points, 3);
    min_dists_in.resize(num_points);
    max_dists_in.resize(num_points);

    int num_valid_points = 0;
    for (size_t i = 0; i < num_points; ++i) {
        const auto &mp = map_points[i];
        if (mp) {
            points_in.row(i) = mp->pt();
            normals_in.row(i) = mp->get_normal();
            const double min_dist = mp->min_distance();
            const double max_dist = mp->max_distance();
            min_dists_in(i) = min_dist;
            max_dists_in(i) = max_dist;
            num_valid_points++;
        } else {

            points_in.row(i) = Eigen::RowVector3d::Zero();
            normals_in.row(i) = Eigen::RowVector3d::Zero();
            min_dists_in(i) = 0.0;
            max_dists_in(i) = 0.0;
        }
    }

    if (num_valid_points != num_points) {
        MSG_RED_WARN("Number of valid map points mismatch in are_visible");
        return std::make_tuple(std::vector<bool>(), MatNx2<Scalar>(), VecN<Scalar>(),
                               VecN<Scalar>());
    }

    // Handle case where no valid map points were found
    if (num_valid_points == 0) {
        return std::make_tuple(std::vector<bool>(), MatNx2<Scalar>(), VecN<Scalar>(),
                               VecN<Scalar>());
    }

    auto projection_result = project_points<double>(points_in, do_stereo_project);
    const auto &uvs = projection_result.first;
    const auto &zs = projection_result.second;

    // Validate that projection returned expected number of points
    if (uvs.rows() != num_valid_points || zs.size() != num_valid_points) {
        MSG_ERROR("Projection result size mismatch in are_visible");
        return std::make_tuple(std::vector<bool>(), MatNx2<Scalar>(), VecN<Scalar>(),
                               VecN<Scalar>());
    }

    const Eigen::RowVector3d Ow_row = Ow().transpose().template cast<double>();
    MatNx3<double> POs;
    POs.resize(points_in.rows(), 3);
    VecN<double> dists_in;
    dists_in.resize(points_in.rows());

    for (size_t i = 0; i < points_in.rows(); ++i) {
        const auto &point = points_in.row(i);
        const Eigen::RowVector3d PO = point - Ow_row;
        dists_in[i] = PO.norm();

        // Safety check for division by zero
        if (dists_in[i] < static_cast<double>(kMinZ)) {
            // Point is too close to camera center, mark as not visible
            POs.row(i) = Eigen::RowVector3d::Zero();
        } else {
            POs.row(i) = PO / dists_in[i];
        }
    }

    VecN<double> cos_view;
    cos_view.resize(normals_in.rows());
    for (size_t i = 0; i < normals_in.rows(); ++i) {
        cos_view[i] = normals_in.row(i).dot(POs.row(i));
    }

    auto are_in_image_flags = are_in_image<double>(uvs, zs.template cast<double>());

    // Validate array sizes are consistent
    if (are_in_image_flags.size() != num_valid_points || cos_view.size() != num_valid_points ||
        dists_in.size() != num_valid_points) {
        MSG_ERROR("Array size mismatch in are_visible");
        return std::make_tuple(std::vector<bool>(), MatNx2<Scalar>(), VecN<Scalar>(),
                               VecN<Scalar>());
    }

    std::vector<bool> are_in_good_view_angle;
    are_in_good_view_angle.resize(cos_view.size());
    for (size_t i = 0; i < cos_view.size(); ++i) {
        are_in_good_view_angle[i] = cos_view[i] > Parameters::kViewingCosLimitForPoint;
    }

    std::vector<bool> are_in_good_distance;
    are_in_good_distance.resize(dists_in.size());
    for (size_t i = 0; i < dists_in.size(); ++i) {
        are_in_good_distance[i] = dists_in[i] >= min_dists_in[i] && dists_in[i] <= max_dists_in[i];
    }

    std::vector<bool> out_flags;
    out_flags.resize(are_in_image_flags.size());
    for (size_t i = 0; i < are_in_image_flags.size(); ++i) {
        out_flags[i] =
            are_in_image_flags[i] && are_in_good_view_angle[i] && are_in_good_distance[i];
    }

    return std::make_tuple(out_flags, uvs.template cast<Scalar>(), zs.template cast<Scalar>(),
                           dists_in.template cast<Scalar>());
}

bool FrameBase::operator==(const FrameBase &rhs) const { return id == rhs.id; }

bool FrameBase::operator<(const FrameBase &rhs) const { return id < rhs.id; }

bool FrameBase::operator<=(const FrameBase &rhs) const { return id <= rhs.id; }

size_t FrameBase::hash() const { return std::hash<int>{}(id); }

// ------------------------------------------------------------
// Frame
// ------------------------------------------------------------

// Static member definitions for Frame
bool Frame::is_store_imgs = false;
bool Frame::is_compute_median_depth = false;

void Frame::copy_from(const Frame &other) {

    FrameBase::copy_from(other);

    is_keyframe = other.is_keyframe;

    // shallow copy
    img = other.img;
    img_right = other.img_right;
    depth_img = other.depth_img;
    semantic_img = other.semantic_img;
    semantic_instances_img = other.semantic_instances_img;

    kf_ref = other.kf_ref;

    kps = other.kps;
    kps_r = other.kps_r;
    kpsu = other.kpsu;
    kpsn = other.kpsn;

    octaves = other.octaves;
    octaves_r = other.octaves_r;
    sizes = other.sizes;
    angles = other.angles;

    des = other.des;
    des_r = other.des_r;
    kps_sem = other.kps_sem;

    depths = other.depths;
    kps_ur = other.kps_ur;

    points = other.points;
    // outliers = other.outliers;  // We don't preserve Frame outliers

    is_blurry = other.is_blurry;
    laplacian_var = other.laplacian_var;

    kf_ref = other.kf_ref;

    _kd = other._kd;

    _points_id_data = other._points_id_data;
    _kf_ref_id = other._kf_ref_id;
}

void Frame::reset() {
    FrameBase::reset();
    is_keyframe = false;

    _kd = nullptr; // first reset the KD-tree pointer to avoid memory corruption with kpsu data

    img = cv::Mat();
    img_right = cv::Mat();
    depth_img = cv::Mat();
    semantic_img = cv::Mat();
    semantic_instances_img = cv::Mat();

    kps.resize(0, 2);
    kps_r.resize(0, 2);
    kpsu.resize(0, 2);
    kpsn.resize(0, 2);

    octaves.clear();
    octaves_r.clear();
    sizes.clear();
    angles.clear();

    des = cv::Mat();
    des_r = cv::Mat();

    depths.clear();
    kps_ur.clear();

    points.clear();
    outliers.clear();

    kf_ref = nullptr;

    is_blurry = false;
    laplacian_var = 0.0f;
}

// Frame Implementation
Frame::Frame(const CameraPtr &camera, const cv::Mat &img, const cv::Mat &img_right,
             const cv::Mat &depth, const CameraPose &pose, int id, double timestamp, int img_id,
             const cv::Mat &semantic_img, const pyslam::FrameDataDict &frame_data_dict)
    : FrameBase(camera, pose, id, timestamp, img_id) {

    // Initialize other members
    is_keyframe = false;
    is_blurry = false;
    laplacian_var = 0.0f;
    kf_ref = nullptr;

    // Initialize feature data arrays
    kps.resize(0, 2);
    kps_r.resize(0, 2);
    kpsu.resize(0, 2);
    kpsn.resize(0, 2);

    octaves.clear();
    octaves_r.clear();
    sizes.clear();
    angles.clear();

    des = cv::Mat();
    des_r = cv::Mat();

    depths.clear();
    kps_ur.clear();
    points.clear();
    outliers.clear();

    // Initialize image data
    if (!img.empty() && Frame::is_store_imgs) {
        this->img = img.clone();
    }
    if (!img_right.empty() && Frame::is_store_imgs) {
        this->img_right = img_right.clone();
    }
    cv::Mat depth_ = depth;
    if (!depth.empty()) {
        if (camera) {
            if (fabs(camera->depth_factor - 1.0) > 1e-5 && depth.type() != CV_32F) {
                depth_.convertTo(depth_, CV_32F, camera->depth_factor);
            }
        } else {
            MSG_ERROR("Frame::Frame() - camera is nullptr");
        }

        if (Frame::is_store_imgs) {
            this->depth_img = depth_.clone();
        }
    }
    if (!semantic_img.empty() && Frame::is_store_imgs) {
        this->semantic_img = semantic_img.clone();
    }

    if (!img.empty()) {
        this->manage_features(img, img_right);
    }

    const int N = kps.rows();
    if (N > 0) {
        if (!depth_.empty()) {
            compute_stereo_from_rgbd(depth_);
        } else if (!img_right.empty()) {
            depths.resize(N, -1.0);
            kps_ur.resize(N, -1.0);
            compute_stereo_matches(img, img_right);
        }
    }
}

// Frame copy constructor and assignment operator deleted due to mutexes

const std::shared_ptr<cKDTree2f> &Frame::kd() {
    std::lock_guard<std::mutex> lock(_lock_kd);
    if (!_kd) {
        if (kpsu.rows() > 0) {
#if 0
            // Use the copying constructor to create an owning KD-tree
            // This prevents memory corruption when kpsu data changes
            MatNx2f kpsu_copy = kpsu;
            _kd = std::make_shared<cKDTree2f>(kpsu_copy);
#else
            _kd = std::make_shared<cKDTree2f>(kpsu);
#endif
        }
    }
    return _kd;
}

MapPointPtr Frame::get_point_match(int idx) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    if (idx >= 0 && idx < static_cast<int>(points.size())) {
        return points[idx];
    }
    return nullptr;
}

void Frame::set_point_match(MapPointPtr p, int idx) {
    std::lock_guard<std::mutex> lock(_lock_features);
    if (idx >= 0 && idx < static_cast<int>(points.size())) {
        points[idx] = p;
    }
}

void Frame::remove_point_match(int idx) {
    std::lock_guard<std::mutex> lock(_lock_features);
    if (idx >= 0 && idx < static_cast<int>(points.size())) {
        points[idx] = nullptr;
    }
}

void Frame::replace_point_match(MapPointPtr &p, int idx) {
    if (idx >= 0 && idx < static_cast<int>(points.size())) {
        points[idx] = p; // replacing is not critical (it does not create a 'None jump')
    }
}

void Frame::remove_point(MapPointPtr p) {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] == p) {
            points[i] = nullptr;
        }
    }
}

void Frame::remove_frame_views(const std::vector<int> &idxs) {
    if (idxs.empty()) {
        return;
    }

    std::vector<std::pair<MapPointPtr, int>> frame_views_to_remove;
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        for (int idx : idxs) {
            if (idx >= 0 && idx < static_cast<int>(points.size()) && points[idx]) {
                frame_views_to_remove.emplace_back(points[idx], idx);
            }
        }
    }
    auto self = shared_from_this();
    for (const auto &[p, idx] : frame_views_to_remove) {
        p->remove_frame_view(self, idx);
    }
}

void Frame::reset_points() {
    std::lock_guard<std::mutex> lock(_lock_features);
    int num_keypoints = kps.rows();
    points.assign(num_keypoints, nullptr);
    outliers.assign(num_keypoints, false);
}

std::vector<MapPointPtr> Frame::get_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return points;
}

std::vector<MapPointPtr> Frame::get_matched_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<MapPointPtr> matched_points;
    matched_points.reserve(points.size());
    for (const auto &p : points) {
        if (p) {
            matched_points.push_back(p);
        }
    }
    return matched_points;
}

std::vector<int> Frame::get_matched_points_idxs() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<int> matched_idxs;
    matched_idxs.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i]) {
            matched_idxs.push_back(static_cast<int>(i));
        }
    }
    return matched_idxs;
}

std::vector<int> Frame::get_unmatched_points_idxs() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<int> unmatched_idxs;
    for (size_t i = 0; i < points.size(); ++i) {
        if (!points[i]) {
            unmatched_idxs.push_back(i);
        }
    }
    return unmatched_idxs;
}

std::pair<std::vector<MapPointPtr>, std::vector<int>> Frame::get_matched_inlier_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<MapPointPtr> matched_points;
    std::vector<int> matched_idxs;
    matched_points.reserve(points.size());
    matched_idxs.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && !outliers[i]) {
            matched_points.push_back(points[i]);
            matched_idxs.push_back(i);
        }
    }
    return std::make_pair(matched_points, matched_idxs);
}

std::vector<MapPointPtr> Frame::get_matched_good_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<MapPointPtr> good_points;
    good_points.reserve(points.size());
    for (const auto &p : points) {
        if (p && !p->is_bad()) {
            good_points.push_back(p);
        }
    }
    return good_points;
}

std::vector<int> Frame::get_matched_good_points_idxs() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<int> idxs;
    idxs.reserve(points.size());
    const size_t num_points = points.size();
    for (size_t i = 0; i < num_points; ++i) {
        const auto &p = points[i];
        if (p && !p->is_bad()) {
            idxs.push_back(i);
        }
    }
    return idxs;
}

std::vector<std::pair<MapPointPtr, int>> Frame::get_matched_good_points_and_idxs() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    const size_t num_points = points.size();
    std::vector<std::pair<MapPointPtr, int>> good_points_and_idxs;
    good_points_and_idxs.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        const auto &p = points[i];
        if (p && !p->is_bad()) {
            good_points_and_idxs.emplace_back(p, i);
        }
    }
    return good_points_and_idxs;
}

int Frame::num_tracked_points(int minObs) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    int count = 0;
    for (const auto &p : points) {
        if (p && p->is_good_with_min_obs(minObs)) {
            count++;
        }
    }
    return count;
}

int Frame::num_matched_inlier_map_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    int count = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &p = points[i];
        if (p && !outliers[i] && p->num_observations() > 0) {
            count++;
        }
    }
    return count;
}

// without locking since used in the tracking thread
std::vector<bool> Frame::get_tracked_mask() const {
    const int num_points = static_cast<int>(points.size());
    std::vector<bool> tracked_mask(num_points, false);
    for (int i = 0; i < num_points; ++i) {
        if (points[i] && !outliers[i]) {
            tracked_mask[i] = true; // point is tracked
        }
    }
    return std::move(tracked_mask);
}

int Frame::update_map_points_statistics(const SensorType &sensor_type) {
    std::lock_guard<std::mutex> lock(_lock_features);
    int num_matched_inlier_points = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i]) {
            if (!outliers[i]) {
                points[i]->increase_found();
                if (points[i]->num_observations() > 0) {
                    num_matched_inlier_points++;
                }
            } else if (sensor_type == SensorType::STEREO) {
                points[i] = nullptr;
            }
        }
    }
    return num_matched_inlier_points;
}

int Frame::clean_outlier_map_points() {
    int num_matched_points = 0;
    std::vector<std::pair<MapPointPtr, int>> frame_views_to_remove;

    {
        std::lock_guard<std::mutex> lock(_lock_features);
        for (size_t i = 0; i < points.size(); ++i) {
            const auto &p = points[i];
            if (p) {
                if (outliers[i]) {
                    frame_views_to_remove.emplace_back(p, static_cast<int>(i));
                    p->last_frame_id_seen = this->id;
                    points[i] = nullptr;
                    outliers[i] = false;
                } else if (p->num_observations() > 0) {
                    num_matched_points++;
                }
            }
        }
    }

    auto self = shared_from_this();
    for (const auto &[p, idx] : frame_views_to_remove) {
        p->remove_frame_view(self, idx);
    }
    return num_matched_points;
}

void Frame::clean_bad_map_points() {
    std::vector<std::pair<MapPointPtr, int>> frame_views_to_remove;

    {
        std::lock_guard<std::mutex> lock(_lock_features);
        for (size_t i = 0; i < points.size(); ++i) {
            const auto &p = points[i];
            if (!p)
                continue;
            if (p->is_bad()) {
                frame_views_to_remove.emplace_back(p, static_cast<int>(i));
                points[i] = nullptr;
                outliers[i] = false;
            } else {
                p->last_frame_id_seen = this->id;
                p->increase_visible();
            }
        }
    }

    auto self = shared_from_this();
    for (const auto &[p, idx] : frame_views_to_remove) {
        p->remove_frame_view(self, idx);
    }
}

void Frame::clean_vo_matches() {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &p = points[i];
        if (p && p->num_observations() < 1) {
            points[i] = nullptr;
            outliers[i] = false;
        }
    }
}

void Frame::check_replaced_map_points() {
    std::lock_guard<std::mutex> lock(_lock_features);
    int num_replaced_points = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &p = points[i];
        if (p) {
            const MapPointPtr replacement = p->get_replacement();
            if (replacement) {
                points[i] = replacement;
                num_replaced_points++;
            }
        }
    }
    // std::cout << "#replaced points: " << num_replaced_points << std::endl;
}

template <typename T>
void Frame::extract_depth_values(const cv::Mat_<T> &depth, std::vector<bool> &valid_depth_mask,
                                 std::vector<float> &valid_depths) {
    const int N = kps.rows();
    valid_depth_mask.resize(N, false);
    valid_depths.reserve(N);

    for (int i = 0; i < N; i++) {
        // Convert keypoint coordinates to integers (u, v)
        const int u = static_cast<int>(std::floor(kps(i, 0)));
        const int v = static_cast<int>(std::floor(kps(i, 1)));

        // Ensure coordinates are within image bounds
        if (u >= 0 && u < depth.cols && v >= 0 && v < depth.rows) {
            // Get depth value at keypoint location (v, u) - note the order
            const float depth_val = static_cast<float>(depth.template at<T>(v, u));
            valid_depth_mask[i] = (depth_val > Parameters::kMinDepth) && std::isfinite(depth_val);
            if (valid_depth_mask[i]) {
                depths[i] = static_cast<float>(depth_val);
                kps_ur[i] = static_cast<float>(kpsu(i, 0) - camera->bf / depth_val);
                valid_depths.push_back(depths[i]);
            }
        }
    }
}

void Frame::compute_stereo_from_rgbd(const cv::Mat &depth) {

    // Get number of keypoints
    const int N = kps.rows();

    // Initialize output arrays
    depths.resize(N, -1.0);
    kps_ur.resize(N, -1.0);

    // Extract depth values at keypoint locations
    std::vector<bool> valid_depth_mask(N, false);

    // Collect valid depth values for median computation
    std::vector<float> valid_depths;
    valid_depths.reserve(N);

    if (depth.type() == CV_32F) {
        extract_depth_values<float>(depth, valid_depth_mask, valid_depths);
    }
#if 0
    else if (depth.type() == CV_64F) {
        extract_depth_values<double>(depth, valid_depth_mask, valid_depths);
    } else if (depth.type() == CV_16U) { // <- use 16U for unsigned short
        extract_depth_values<uint16_t>(depth, valid_depth_mask, valid_depths);
    } else if (depth.type() == CV_16F) { // OpenCV half float
        // Read as float16 and cast to float; OpenCV exposes cv::float16_t / _Float16 (__fp16)
        const cv::Mat depth32f = cv::Mat(depth.size(), CV_32F);
        depth.convertTo(depth32f, CV_32F); // safe conversion
        extract_depth_values<float>(depth32f, valid_depth_mask, valid_depths);
    } else {
        MSG_ERROR("Frame::compute_stereo_from_rgbd() - unknown depth type");
    }
#else
    else {
        MSG_ERROR("Frame::compute_stereo_from_rgbd() - unknown depth type");
    }
#endif

    // Compute median depth if enabled
    if (Frame::is_compute_median_depth) {
        if (!valid_depths.empty()) {
            // Compute median
            std::sort(valid_depths.begin(), valid_depths.end());
            size_t mid = valid_depths.size() / 2;
            if (valid_depths.size() % 2 == 0) {
                median_depth =
                    static_cast<float>((valid_depths[mid - 1] + valid_depths[mid]) / 2.0);
            } else {
                median_depth = static_cast<float>(valid_depths[mid]);
            }

            // Compute FOV center in camera coordinates
            fov_center_c = camera->unproject_point_3d<double>(camera->cx, camera->cy,
                                                              static_cast<double>(median_depth));

            // Transform to world coordinates: fov_center_w = Rwc @ fov_center_c + Ow
            {
                pose_lock_guard_type lock(_lock_pose);
                fov_center_w =
                    _pose.get_inverse_rotation_matrix() * fov_center_c + _pose.position();
            }
        } else {
            median_depth = 0.0f;
            fov_center_c = Eigen::Vector3d::Zero();
            fov_center_w = Eigen::Vector3d::Zero();
        }
    }
}

void Frame::compute_stereo_matches(const cv::Mat &img, const cv::Mat &img_right) {
    constexpr bool kRowMatching = true;
    const float kRatioTest = Parameters::kFeatureMatchDefaultRatioTest;
    const float min_z = camera->b;
    const float min_disparity = 0;
    const float max_disparity = camera->bf / min_z;

    if (!FeatureSharedResources::stereo_matching_callback) {
        MSG_ERROR("Frame::compute_stereo_matches() - stereo_matching_callback not set");
        return;
    }

    const auto &[idxs1, idxs2] = FeatureSharedResources::stereo_matching_callback(
        img, img_right, des, des_r, kps, kps_r, kRatioTest, kRowMatching, max_disparity);
    if (idxs1.size() == 0 || idxs2.size() == 0) {
        MSG_WARN("Frame::compute_stereo_matches() - no stereo matches found");
        return;
    }

    // Filter by good disparities
    std::vector<int> good_matched_idxs1, good_matched_idxs2;
    std::vector<float> good_disparities;
    good_matched_idxs1.reserve(idxs1.size());
    good_matched_idxs2.reserve(idxs2.size());
    good_disparities.reserve(idxs1.size());

    for (size_t i = 0; i < idxs1.size(); ++i) {
        float disparity = kps(idxs1[i], 0) - kps_r(idxs2[i], 0);
        if (disparity > min_disparity && disparity < max_disparity) {
            good_disparities.push_back(disparity);
            good_matched_idxs1.push_back(idxs1[i]);
            good_matched_idxs2.push_back(idxs2[i]);
        }
    }

    if (good_matched_idxs1.empty()) {
        MSG_WARN("Frame::compute_stereo_matches() - no good disparities found");
        return;
    }

    // Optional: Check with fundamental matrix (disabled by default as in Python)
    constexpr bool do_check_with_fundamental_mat = false;
    if (do_check_with_fundamental_mat) {
        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(good_matched_idxs1.size());
        pts2.reserve(good_matched_idxs2.size());

        for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
            pts1.emplace_back(kps(good_matched_idxs1[i], 0), kps(good_matched_idxs1[i], 1));
            pts2.emplace_back(kps_r(good_matched_idxs2[i], 0), kps_r(good_matched_idxs2[i], 1));
        }

        cv::Mat mask_inliers;
        constexpr float fmat_err_thld = 3.0;
        constexpr float confidence = 0.999;
#if OPENCV_VERSION_MAJOR >= 4
        constexpr int ransac_method = cv::USAC_MAGSAC;
#else
        constexpr int ransac_method = cv::RANSAC;
#endif
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, ransac_method, fmat_err_thld, confidence,
                                           mask_inliers);

        std::vector<int> filtered_idxs1, filtered_idxs2;
        std::vector<float> filtered_disparities;

        for (int i = 0; i < mask_inliers.rows; ++i) {
            if (mask_inliers.at<uchar>(i) == 1) {
                filtered_idxs1.push_back(good_matched_idxs1[i]);
                filtered_idxs2.push_back(good_matched_idxs2[i]);
                filtered_disparities.push_back(good_disparities[i]);
            }
        }

        good_matched_idxs1 = std::move(filtered_idxs1);
        good_matched_idxs2 = std::move(filtered_idxs2);
        good_disparities = std::move(filtered_disparities);

        std::cout << "Frame::compute_stereo_matches() - fundamental matrix filtering: "
                  << good_matched_idxs1.size() << " matches remaining" << std::endl;
    }

    // Subpixel stereo matching
    constexpr bool do_subpixel_stereo_matching = true;
    if (do_subpixel_stereo_matching) {
        cv::Mat img_bw, img_right_bw;

        // Convert to grayscale if needed
        if (img.channels() > 1) {
            cv::cvtColor(img, img_bw, cv::COLOR_RGB2GRAY);
        } else {
            img_bw = img;
        }

        if (img_right.channels() > 1) {
            cv::cvtColor(img_right, img_right_bw, cv::COLOR_RGB2GRAY);
        } else {
            img_right_bw = img_right;
        }

        // Perform subpixel stereo matching
        auto [refined_disparities, refined_us_right, valid_idxs] =
            stereo_match_subpixel_correlation<float>(good_matched_idxs1, good_matched_idxs2, kps,
                                                     kps_r, min_disparity, max_disparity, img_bw,
                                                     img_right_bw);

        // Update arrays with refined results
        std::vector<int> final_idxs1, final_idxs2;
        std::vector<float> final_disparities;
        final_idxs1.reserve(valid_idxs.size());
        final_idxs2.reserve(valid_idxs.size());
        final_disparities.reserve(valid_idxs.size());

        for (size_t i = 0; i < valid_idxs.size(); ++i) {
            const int idx = valid_idxs[i];
            final_idxs1.push_back(good_matched_idxs1[idx]);
            final_idxs2.push_back(good_matched_idxs2[idx]);
            final_disparities.push_back(refined_disparities[idx]);

            // Update kps_ur with refined right u-coordinates
            kps_ur[good_matched_idxs1[idx]] = refined_us_right[idx];
        }

        good_matched_idxs1 = std::move(final_idxs1);
        good_matched_idxs2 = std::move(final_idxs2);
        good_disparities = std::move(final_disparities);
    }

    // Optional: Check chi-squared of reprojection errors, just for the hell of it (debugging)
    // (disabled by default)
    constexpr bool do_chi2_check = false;
    if (do_chi2_check && !good_matched_idxs1.empty()) {
        // Triangulate points
        Eigen::Matrix4f pose_l = Eigen::Matrix4f::Identity();
        Eigen::Vector3f t_rl(-camera->b, 0, 0);
        Eigen::Matrix4f pose_rl = Eigen::Matrix4f::Identity();
        pose_rl.block<3, 1>(0, 3) = t_rl;

        // Convert keypoints to normalized coordinates
        std::vector<Eigen::Vector2f> kpsn_l, kpsn_r;
        kpsn_l.reserve(good_matched_idxs1.size());
        kpsn_r.reserve(good_matched_idxs2.size());

        Eigen::Matrix3f Kinv = camera->Kinv.cast<float>();
        for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
            const Eigen::Vector3f kp_l(kps(good_matched_idxs1[i], 0), kps(good_matched_idxs1[i], 1),
                                       1.0);
            const Eigen::Vector3f kp_r(kps_r(good_matched_idxs2[i], 0),
                                       kps_r(good_matched_idxs2[i], 1), 1.0);
            kpsn_l.emplace_back((Kinv * kp_l).head<2>());
            kpsn_r.emplace_back((Kinv * kp_r).head<2>());
        }

        // Triangulate points:
        // Using the depth from the disparity to back-project to 3D points from left image
        // Then project the 3D points to right image to check the reprojection errors
        std::vector<Eigen::Vector3f> pts3d;
        pts3d.reserve(good_matched_idxs1.size());

        for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
            const float disparity = good_disparities[i];
            if (disparity > 0) {
                const float depth = camera->bf / disparity;
                Eigen::Vector3f pt3d(kpsn_l[i].x() * depth, kpsn_l[i].y() * depth, depth);
                pts3d.push_back(pt3d);
            } else {
                pts3d.emplace_back(0, 0, 0); // Invalid point
            }
        }

        // Check reprojection errors
        std::vector<bool> good_chi2_mask;
        good_chi2_mask.reserve(pts3d.size());

        for (size_t i = 0; i < pts3d.size(); ++i) {
            if (pts3d[i].z() > 0) {
                // Project back to right image
                Eigen::Vector3f pt3d_r = pts3d[i] + t_rl;

                // auto [uv_l, depth_l] = camera->project_point(pts3d[i]);
                auto [uv_r, depth_r] = camera->project_point(pt3d_r);

                // if (depth_l > 0 && depth_r > 0) {
                if (depth_r > 0) {
                    // Compute reprojection errors
                    // Eigen::Vector2d err_l = uv_l - Eigen::Vector2d(kps(good_matched_idxs1[i],
                    // 0), kps(good_matched_idxs1[i], 1));
                    Eigen::Vector2f err_r = uv_r - Eigen::Vector2f(kps_r(good_matched_idxs2[i], 0),
                                                                   kps_r(good_matched_idxs2[i], 1));

                    // float err_l_sqr = err_l.squaredNorm();
                    float err_r_sqr = err_r.squaredNorm();

                    // Get sigma from octave levels
                    // const float inv_sigma2_l =
                    // FeatureSharedResources::inv_level_sigmas2[octaves[good_matched_idxs1[i]]];
                    const float inv_sigma2_r =
                        FeatureSharedResources::inv_level_sigmas2[octaves_r[good_matched_idxs2[i]]];

                    // float chi2_l = err_l_sqr / (sigma_l * sigma_l);
                    float chi2_r = err_r_sqr * inv_sigma2_r;

                    // good_chi2_mask.push_back(chi2_l < Parameters::kChi2Mono && chi2_r <
                    // Parameters::kChi2Mono);
                    good_chi2_mask.push_back(chi2_r < Parameters::kChi2Mono);
                } else {
                    good_chi2_mask.push_back(false);
                }
            } else {
                good_chi2_mask.push_back(false);
            }
        }

        // Filter by chi-squared results
        std::vector<int> filtered_idxs1, filtered_idxs2;
        std::vector<float> filtered_disparities;

        for (size_t i = 0; i < good_chi2_mask.size(); ++i) {
            if (good_chi2_mask[i]) {
                filtered_idxs1.push_back(good_matched_idxs1[i]);
                filtered_idxs2.push_back(good_matched_idxs2[i]);
                filtered_disparities.push_back(good_disparities[i]);
            }
        }

        good_matched_idxs1 = std::move(filtered_idxs1);
        good_matched_idxs2 = std::move(filtered_idxs2);
        good_disparities = std::move(filtered_disparities);

        std::cout << "Frame::compute_stereo_matches() - chi-squared filtering: "
                  << good_matched_idxs1.size() << " matches remaining" << std::endl;
    }

    // Update depths and kps_ur arrays
    for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
        const int idx1 = good_matched_idxs1[i];
        const int idx2 = good_matched_idxs2[i];
        depths[idx1] = camera->bf / good_disparities[i];
        if (!do_subpixel_stereo_matching) {
            kps_ur[idx1] = kps_r(idx2, 0);
        }
    }

    std::cout << "Frame::compute_stereo_matches() - found final " << good_matched_idxs1.size()
              << " stereo matches" << std::endl;

    // Compute median depth if enabled
    if (is_compute_median_depth) {
        std::vector<float> valid_depths;
        valid_depths.reserve(depths.size());

        for (float depth : depths) {
            if (depth > Parameters::kMinDepth) {
                valid_depths.push_back(depth);
            }
        }

        if (!valid_depths.empty()) {
            std::sort(valid_depths.begin(), valid_depths.end());
            size_t mid = valid_depths.size() / 2;
            if (valid_depths.size() % 2 == 0) {
                median_depth = (valid_depths[mid - 1] + valid_depths[mid]) / 2.0f;
            } else {
                median_depth = valid_depths[mid];
            }

            // Compute FOV center
            fov_center_c = camera->unproject_point_3d<double>(camera->cx, camera->cy,
                                                              static_cast<double>(median_depth));
            fov_center_w = _pose.Rwc() * fov_center_c + _pose.Ow();

            std::cout << "Frame::compute_stereo_matches() - median depth: " << median_depth
                      << std::endl;
        }
    }

    // Show stereo matches if enabled
    if (Parameters::kStereoMatchingShowMatchedPoints) {
        MatNx2f matched_kps1(static_cast<int>(good_matched_idxs1.size()), 2);
        MatNx2f matched_kps2(static_cast<int>(good_matched_idxs2.size()), 2);
        for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
            matched_kps1(i, 0) = kps(good_matched_idxs1[i], 0);
            matched_kps1(i, 1) = kps(good_matched_idxs1[i], 1);
            matched_kps2(i, 0) = kps_r(good_matched_idxs2[i], 0);
            matched_kps2(i, 1) = kps_r(good_matched_idxs2[i], 1);
        }
        cv::Mat stereo_img_matches = pyslam::draw_feature_matches(img, img_right, matched_kps1,
                                                                  matched_kps2, {}, {}, /**/ false);
        cv::imshow("stereo_img_matches", stereo_img_matches);
        cv::waitKey(1);
    }
}

// NOTE: Not-thread safe, not to be used outside of the tracking thread
template <typename Scalar>
std::pair<Vec3<Scalar>, bool> Frame::unproject_point_3d(int idx, bool transform_in_world) const {
    const Scalar depth = depths[idx];
    if (depth > Parameters::kMinDepth) {
        Vec3<Scalar> pt3d(depth * kpsn(idx, 0), depth * kpsn(idx, 1), depth);
        if (transform_in_world) {
            pt3d = _pose.Rwc().template cast<Scalar>() * pt3d + _pose.Ow().template cast<Scalar>();
        }
        return std::make_pair(pt3d, true);
    } else {
        return std::make_pair(Vec3<Scalar>::Zero(), false);
    }
}

template <typename Scalar>
std::pair<std::vector<Vec3<Scalar>>, std::vector<bool>>
Frame::unproject_points_3d(const std::vector<int> &idxs, bool transform_in_world) const {
    std::vector<Vec3<Scalar>> pts3d(idxs.size(), Vec3<Scalar>());
    std::vector<bool> valid_mask(idxs.size(), false);

    for (size_t i = 0; i < idxs.size(); ++i) {
        const int idx = idxs[i];
        const Scalar depth = depths[idx];
        if (depth > Parameters::kMinDepth) {
            Vec3<Scalar> pt3d(depth * kpsn(idx, 0), depth * kpsn(idx, 1), depth);
            if (transform_in_world) {
                pts3d[i] =
                    _pose.Rwc().template cast<Scalar>() * pt3d + _pose.Ow().template cast<Scalar>();
            } else {
                pts3d[i] = pt3d;
            }
            valid_mask[i] = true;
        }
    }
    return std::make_pair(pts3d, valid_mask);
}

template <typename Scalar>
Scalar Frame::compute_points_median_depth(MatNx3Ref<Scalar> points3d,
                                          const Scalar percentile) const {
    Vec3<Scalar> Rcw2;
    Scalar tcw2;
    {
        pose_lock_guard_type lock(_lock_pose);
        Rcw2 = _pose.Rcw().row(2).template cast<Scalar>(); // just 2-nd row
        tcw2 = static_cast<Scalar>(_pose.tcw()[2]);        // just 2-nd element (z-axis)
    }
    MatNx3<Scalar> points3d_frame;
    const bool use_input_points3d = points3d.rows() > 0;
    Eigen::Index n = points3d.rows();
    if (!use_input_points3d) {
        pose_lock_guard_type lock(_lock_features);
        // The input points3d is empty, so we use the points array (from this Frame)
        points3d_frame.resize(points.size(), 3);
        Eigen::Index r = 0;
        for (const auto &mp : points) {
            if (mp) {
                points3d_frame.row(r++) = mp->pt().template cast<Scalar>();
            }
        }
        points3d_frame.conservativeResize(r, 3);
        n = r;
    }

    if (n == 0) {
        return static_cast<Scalar>(-1.0);
    }

    const MatNx3Ref<Scalar> points3d_to_use =
        (use_input_points3d) ? points3d : Eigen::Ref<const MatNx3<Scalar>>(points3d_frame);

    std::vector<Scalar> depths;
    depths.reserve(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar zc = Rcw2.dot(points3d_to_use.row(i)) + tcw2;
        depths.push_back(zc);
    }
#if 0
    std::sort(depths.begin(), depths.end());
    const size_t idx = std::min(static_cast<size_t>(depths.size() * percentile), depths.size() - 1);
#else
    // Clamp percentile to [0,1], then select in linear time
    const Scalar p = std::max<Scalar>(0, std::min<Scalar>(1, percentile));
    const size_t idx = std::min(static_cast<size_t>(depths.size() * p), depths.size() - 1);
    std::nth_element(depths.begin(), depths.begin() + idx, depths.end());
#endif
    return static_cast<Scalar>(depths[idx]);
}

void Frame::set_img_right(const cv::Mat &img_right) { this->img_right = img_right.clone(); }

void Frame::set_depth_img(const cv::Mat &depth_img) {
    if (camera && camera->depth_factor != 1.0f) {
        cv::Mat scaled_depth;
        depth_img.convertTo(scaled_depth, depth_img.type(), camera->depth_factor);
        this->depth_img = scaled_depth.clone();
    } else {
        this->depth_img = depth_img.clone();
    }
}

template <bool is_single_channel, typename T>
void extract_semantic_at_keypoints_impl(const cv::Mat &semantic_img, MatNx2fRef kps,
                                        cv::Mat &kps_sem) {
    const int num_kps = static_cast<int>(kps.rows());
    const int channels = semantic_img.channels();
    const int depth = semantic_img.depth();
    kps_sem.create(num_kps, channels, CV_MAKETYPE(depth, 1));

    const int cols = semantic_img.cols;
    const int rows = semantic_img.rows;

    for (int i = 0; i < num_kps; ++i) {
        const int u = static_cast<int>(std::floor(kps(i, 0)));
        const int v = static_cast<int>(std::floor(kps(i, 1)));

        const bool in_bounds = (u >= 0 && u < cols && v >= 0 && v < rows);
        T *dst = kps_sem.ptr<T>(i);

        if (in_bounds) {
            if constexpr (is_single_channel) {
                // source is single channel
                dst[0] = semantic_img.at<T>(v, u);
                // any extra columns (shouldn’t exist) are left as-is
            } else {
                // source is interleaved multi-channel
                const T *src = semantic_img.ptr<T>(v) + u * channels;
                std::memcpy(dst, src, sizeof(T) * channels);
            }
        } else {
            // out of bounds => fill with zeros
            std::fill(dst, dst + channels, T(0));
        }
    }
}

void extract_semantic_at_keypoints(const cv::Mat &semantic_img, MatNx2fRef kps, cv::Mat &kps_sem) {
    const bool single = (semantic_img.channels() == 1);

    switch (semantic_img.depth()) {
    case CV_8U:
        single ? extract_semantic_at_keypoints_impl<true, uchar>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, uchar>(semantic_img, kps, kps_sem);
        break;
    case CV_16U:
        single ? extract_semantic_at_keypoints_impl<true, uint16_t>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, uint16_t>(semantic_img, kps, kps_sem);
        break;
    case CV_32S:
        single ? extract_semantic_at_keypoints_impl<true, int32_t>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, int32_t>(semantic_img, kps, kps_sem);
        break;
    case CV_32F:
        single ? extract_semantic_at_keypoints_impl<true, float>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, float>(semantic_img, kps, kps_sem);
        break;
    case CV_64F:
        single ? extract_semantic_at_keypoints_impl<true, double>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, double>(semantic_img, kps, kps_sem);
        break;
#ifdef CV_16F
    case CV_16F:
#ifdef __APPLE__
        // macOS: cv::float16_t is not available, convert to CV_32F first
        {
            cv::Mat semantic_img_32f;
            semantic_img.convertTo(semantic_img_32f, CV_32F);
            single
                ? extract_semantic_at_keypoints_impl<true, float>(semantic_img_32f, kps, kps_sem)
                : extract_semantic_at_keypoints_impl<false, float>(semantic_img_32f, kps, kps_sem);
        }
#else
        // Linux and other platforms: use cv::float16_t directly if available
        single
            ? extract_semantic_at_keypoints_impl<true, cv::float16_t>(semantic_img, kps, kps_sem)
            : extract_semantic_at_keypoints_impl<false, cv::float16_t>(semantic_img, kps, kps_sem);
#endif
        break;
#endif
    default:
        MSG_ERROR_STREAM(
            "Frame::extract_semantic_at_keypoints() - unsupported image depth for semantics: "
            << semantic_img.depth());
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported image depth for semantics");
    }
}

void Frame::set_semantics(const cv::Mat &semantic_img) {
    {
        std::lock_guard<std::mutex> lock(_lock_semantics);
        this->semantic_img = semantic_img.clone();
    }

    if (kps.rows() > 0) {
        std::lock_guard<std::mutex> lock(_lock_features);

        // Extract semantic information at keypoint locations
        extract_semantic_at_keypoints(this->semantic_img, kps, kps_sem);

        // Normalize kps_sem type to avoid mixed-type issues downstream (e.g., push_back)
        // Policy: LABEL -> CV_32S; PROBABILITY_VECTOR/FEATURE_VECTOR -> CV_32F
        const auto &semantic_feature_type = FeatureSharedResources::semantic_feature_type;
        const int target_depth = get_cv_depth_for_semantic_feature_type(semantic_feature_type);
        const int channels = kps_sem.channels();
        const int target_type = CV_MAKETYPE(target_depth, channels);
        if (kps_sem.type() != target_type) {
            cv::Mat converted;
            kps_sem.convertTo(converted, target_type);
            kps_sem = std::move(converted);
        }

        // Ensure contiguous memory layout (equivalent to np.ascontiguousarray)
        if (!kps_sem.isContinuous()) {
            kps_sem = kps_sem.clone();
        }
    }
}

void Frame::set_semantic_instances(const cv::Mat &semantic_instances_img) {
    {
        std::lock_guard<std::mutex> lock(_lock_semantics);
        this->semantic_instances_img = semantic_instances_img.clone();
        // convert to 32S if not already (int32 supports large panoptic segment IDs)
        if (this->semantic_instances_img.depth() != CV_32S) {
            this->semantic_instances_img.convertTo(this->semantic_instances_img, CV_32S);
        }
    }
}

bool Frame::is_semantics_available() const {
    std::lock_guard<std::mutex> lock(_lock_semantics);
    return !this->semantic_img.empty();
}

void Frame::update_points_semantics(void *semantic_fusion_method) {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (const auto &mp : points) {
        if (mp) {
            mp->update_semantics(semantic_fusion_method);
        }
    }
}

void Frame::ensure_contiguous_arrays() {
    // Ensure all OpenCV matrices are contiguous in memory
    if (!des.empty() && !des.isContinuous()) {
        des = des.clone();
    }
    if (!des_r.empty() && !des_r.isContinuous()) {
        des_r = des_r.clone();
    }
    if (!img.empty() && !img.isContinuous()) {
        img = img.clone();
    }
    if (!img_right.empty() && !img_right.isContinuous()) {
        img_right = img_right.clone();
    }
    if (!depth_img.empty() && !depth_img.isContinuous()) {
        depth_img = depth_img.clone();
    }
    if (!semantic_img.empty() && !semantic_img.isContinuous()) {
        semantic_img = semantic_img.clone();
    }
    if (!semantic_instances_img.empty() && !semantic_instances_img.isContinuous()) {
        semantic_instances_img = semantic_instances_img.clone();
    }
}

template <bool with_level_radius>
cv::Mat Frame::draw_feature_trails_(const cv::Mat &img, const std::vector<int> &kps_idxs,
                                    int trail_max_length) const {
    cv::Mat img_out = img.clone();
    std::lock_guard<std::mutex> lock(_lock_features);

    // Pre-allocate reusable containers
    std::vector<cv::Point> pts;
    pts.reserve(trail_max_length);

    // Pre-compute common colors
    const cv::Scalar color_green(0, 255, 0);
    const cv::Scalar color_red(255, 0, 0);
    const cv::Scalar color_black(0, 0, 0);

    // use distorted coordinates when drawing on distorted original image
    for (const int idx : kps_idxs) {
        const auto &kp = kps.row(idx);
        const auto uv = Eigen::Vector2i(std::floor(kp[0]), std::floor(kp[1]));

        const auto &mp = points[idx];
        if (mp && !mp->is_bad()) {
            // there is a corresponding 3D map point

            const int radius = kDrawFeatureRadius[octaves[idx]]; // fake size for visualization
            const auto p_frame_views = mp->frame_views();        // list of (Frame, idx)
            if (!p_frame_views.empty()) {
                // draw the trail (for each keypoint, its trail_max_length corresponding points
                // in previous frames)
                pts.clear();
                int count = 0;
                int last_frame_id = -1;

                for (auto it = p_frame_views.rbegin(); it != p_frame_views.rend(); ++it) {
                    const auto &[f, idx] = *it;
                    if (!f)
                        continue;
                    if (last_frame_id != -1 && last_frame_id - 1 != f->id) {
                        // stop when there is a jump in the ids of frame observations
                        break;
                    }
                    const auto &fkp = f->kps.row(idx);
                    pts.emplace_back(std::floor(fkp[0]), std::floor(fkp[1]));
                    last_frame_id = f->id;
                    count++;
                    if (count >= trail_max_length) {
                        break;
                    }
                }

                cv::Scalar cv_point_color;
                if constexpr (with_level_radius) {
                    cv_point_color = (p_frame_views.size() > 2) ? color_green : color_red;
                } else {
                    const auto color =
                        pyslam::ColorTableGenerator::instance().color_from_int(mp->id);
                    cv_point_color = cv::Scalar(color.r, color.g, color.b);
                }

                // Draw the trail
                const size_t num_pts = pts.size();
                if (num_pts > 1) {
                    if constexpr (with_level_radius) {
                        const auto trail_color = pyslam::Colors::myjet_color_x_255(num_pts);
                        const auto cv_trail_color =
                            cv::Scalar(trail_color[0], trail_color[1], trail_color[2]);
                        cv::polylines(img_out, pts, false, cv_trail_color, 1, cv::LINE_AA);
                    } else {
                        cv::polylines(img_out, pts, false, cv_point_color, 1, cv::LINE_AA);
                    }
                }
                if constexpr (with_level_radius) {
                    cv::circle(img_out, cv::Point(uv[0], uv[1]), radius, cv_point_color, 1);
                } else {
                    cv::circle(img_out, cv::Point(uv[0], uv[1]), 4, cv_point_color, -1);
                }
            }
        } else {
            // no corresponding 3D map point
            cv::circle(img_out, cv::Point(uv[0], uv[1]), 2, color_black);
        }
    }
    return img_out;
}

cv::Mat Frame::draw_feature_trails(const cv::Mat &img, const std::vector<int> &kps_idxs,
                                   const bool with_level_radius, int trail_max_length) const {
    if (with_level_radius) {
        return draw_feature_trails_<true>(img, kps_idxs, trail_max_length);
    } else {
        return draw_feature_trails_<false>(img, kps_idxs, trail_max_length);
    }
}

cv::Mat Frame::draw_all_feature_trails(const cv::Mat &img, const bool with_level_radius,
                                       int trail_max_length) const {
    std::vector<int> all_idxs(kps.rows());
    std::iota(all_idxs.begin(), all_idxs.end(), 0);
    return draw_feature_trails(img, all_idxs, with_level_radius, trail_max_length);
}

void Frame::manage_features(const cv::Mat &img, const cv::Mat &img_right) {
    constexpr bool kVerbose = false;
    if (FeatureSharedResources::feature_detect_and_compute_callback) {
        if constexpr (kVerbose) {
            std::cout << "Frame::manage_features() called" << std::endl;
        }

        // Call Python feature detection for left and right images
        const auto &[result_left, result_right] =
            detect_and_compute_features_parallel(img, img_right);

        const auto &[keypoints_, des_] = result_left;

        // With the new opencv caster for cv::Mat, we can avoid copying the descriptor matrix
#define COPY_DESCRIPTOR 0
#if COPY_DESCRIPTOR
        des = des_.empty() ? cv::Mat()
                           : des_.clone(); // copy the descriptors for safety (in this case, we
                                           // cannot safely share memory across Python/C++)
#else
        des = des_;
#endif

        if (des_.empty()) {
            MSG_WARN("Warning: Empty descriptor matrix received from Python callback");
        }

        // Debug: Check the descriptor matrix before cloning
        if constexpr (kVerbose) {
            std::cout << "Raw descriptor matrix size: " << des_.rows << "x" << des_.cols
                      << ", type: " << des_.type() << std::endl;
            if (des_.rows > 0 && des_.cols > 0) {
                std::cout << "Raw first descriptor: ";
                for (int i = 0; i < std::min(5, des_.cols); ++i) {
                    std::cout << (int)des_.at<uchar>(0, i) << " ";
                }
                std::cout << std::endl;
            }

            // Ensure the descriptor matrix is properly allocated and copied
            if (!des_.empty()) {
                std::cout << "Cloned descriptor matrix size: " << des.rows << "x" << des.cols
                          << ", type: " << des.type() << std::endl;
                if (des.rows > 0 && des.cols > 0) {
                    std::cout << "Cloned first descriptor: ";
                    for (int i = 0; i < std::min(5, des.cols); ++i) {
                        std::cout << (int)des.at<uchar>(0, i) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        const int num_kps = keypoints_.size();

        // Convert keypoints to internal format
        kps.resize(num_kps, 2);
        octaves.resize(num_kps);
        sizes.resize(num_kps);
        angles.resize(num_kps);

        // cv::KeyPoint <-> (pt.x, pt.y, size, angle, response, octave)
        for (size_t i = 0; i < num_kps; ++i) {
            const auto &kp = keypoints_[i];
            kps(i, 0) = std::get<0>(kp);
            kps(i, 1) = std::get<1>(kp);
            sizes[i] = std::get<2>(kp);
            angles[i] = std::get<3>(kp);
            // response is not used
            octaves[i] = std::get<5>(kp);
        }

        // Continue with undistortion and normalization
        if (camera && kps.rows() > 0) {
            // Convert to undistorted coordinates
            kpsu = camera->undistort_points(kps);
            kpsn = camera->unproject_points(kpsu);
        }

        // Initialize point arrays
        if (static_cast<int>(points.size()) != kps.rows()) {
            points.assign(kps.rows(), nullptr);
            outliers.assign(kps.rows(), false);
        }

        if constexpr (kVerbose) {
            std::cout << "Frame::manage_features() - kps: " << kps.rows() << "x" << kps.cols()
                      << std::endl;
            std::cout << "Frame::manage_features() - kpsu: " << kpsu.rows() << "x" << kpsu.cols()
                      << std::endl;
            std::cout << "Frame::manage_features() - kpsn: " << kpsn.rows() << "x" << kpsn.cols()
                      << std::endl;
        }

        // Handle stereo if needed
        if (!img_right.empty()) {
            // const auto &[keypoints_r_, des_r_] =
            // feature_detect_and_compute_right_callback(img_right);

            const auto &[keypoints_r_, des_r_] = result_right;
#if COPY_DESCRIPTOR
            des_r = des_r_.empty()
                        ? cv::Mat()
                        : des_r_.clone(); // copy the descriptors for safety (in this
                                          // case, we cannot safely share memory across Python/C++)
#else
            des_r = des_r_;
#endif

            if (des_r_.empty()) {
                MSG_WARN("Warning: Empty right descriptor matrix received from Python callback");
            }

            const int num_kps_r = keypoints_r_.size();

            kps_r.resize(num_kps_r, 2);
            octaves_r.resize(num_kps_r);

            for (size_t i = 0; i < num_kps_r; ++i) {
                const auto &kp = keypoints_r_[i];
                kps_r(i, 0) = std::get<0>(kp);
                kps_r(i, 1) = std::get<1>(kp);
                // sizes_r[i] = std::get<2>(kp);
                // angles_r[i] = std::get<3>(kp);
                // response is not used
                octaves_r[i] = std::get<5>(kp);
            }

            if constexpr (kVerbose) {
                std::cout << "Frame::manage_features() - found " << num_kps_r << " right keypoints"
                          << std::endl;
            }
        }
    }
}

std::pair<FeatureDetectAndComputeOutput, FeatureDetectAndComputeOutput>
Frame::detect_and_compute_features_parallel(const cv::Mat &img, const cv::Mat &img_right) {
    constexpr bool kVerbose = false; // Fix: reduce logging
    const bool both_callbacks_set =
        FeatureSharedResources::feature_detect_and_compute_callback &&
        FeatureSharedResources::feature_detect_and_compute_right_callback;
    const bool is_stereo = !img_right.empty();

    if (both_callbacks_set && is_stereo) {
        // ----------------------------------------
        // Parallel detection with GIL management
        // ----------------------------------------
        if constexpr (kVerbose) {
            std::cout << "Frame::detect_and_compute_features_parallel() - parallel detection with "
                         "GIL management"
                      << std::endl;
        }

        // Run parallel detection
        FeatureDetectAndComputeOutput result_left, result_right;
        {
#ifdef USE_PYTHON
            // Release the GIL in the calling thread so worker threads can acquire it.
            // Only do this if Python is initialized (required for gil_scoped_release).
            std::unique_ptr<pybind11::gil_scoped_release> release_guard;
            if (Py_IsInitialized() && PyGILState_Check()) {
                release_guard = std::make_unique<pybind11::gil_scoped_release>();
            }
#endif

            auto left_future =
                std::async(std::launch::async, [this, &img]() -> FeatureDetectAndComputeOutput {
                    if (img.empty()) {
                        return FeatureDetectAndComputeOutput();
                    }
#ifdef USE_PYTHON
                    pybind11::gil_scoped_acquire acquire; // Acquire GIL just for this callback
#endif
                    return FeatureSharedResources::feature_detect_and_compute_callback(img);
                });

            auto right_future = std::async(
                std::launch::async, [this, &img_right]() -> FeatureDetectAndComputeOutput {
                    if (img_right.empty()) {
                        return FeatureDetectAndComputeOutput();
                    }
#ifdef USE_PYTHON
                    pybind11::gil_scoped_acquire acquire; // Acquire GIL just for this callback
#endif
                    return FeatureSharedResources::feature_detect_and_compute_right_callback(
                        img_right);
                });

            // Wait for both to complete (GIL is still released here)
            result_left = left_future.get();
            result_right = right_future.get();
        }

        return {result_left, result_right};

    } else if (FeatureSharedResources::feature_detect_and_compute_callback) {
        // ----------------------------------------
        // Single left image callback - no need for GIL management
        // ----------------------------------------
        if constexpr (kVerbose) {
            std::cout << "Frame::detect_and_compute_features_parallel() - just detect on left image"
                      << std::endl;
        }
        if (img.empty()) {
            return {FeatureDetectAndComputeOutput(), FeatureDetectAndComputeOutput()};
        } else {
#ifdef USE_PYTHON
            pybind11::gil_scoped_acquire acquire; // Acquire GIL just for this callback
#endif
            return {FeatureSharedResources::feature_detect_and_compute_callback(img),
                    FeatureDetectAndComputeOutput()};
        }
    } else {
        MSG_ERROR("Frame::detect_and_compute_features_parallel() - Feature detection and compute "
                  "callbacks are not set");
        throw std::runtime_error("Feature detection and compute callbacks are not set");
    }
}

//===============================================================================================
// Explicit template instantiations for FrameBase methods used in Python bindings
//===============================================================================================

template Vec3<float> pyslam::FrameBase::transform_point<float>(Vec3Ref<float> pw) const;
template Vec3<double> pyslam::FrameBase::transform_point<double>(Vec3Ref<double> pw) const;
template MatNx3<float> pyslam::FrameBase::transform_points<float>(MatNx3Ref<float> points) const;
template MatNx3<double> pyslam::FrameBase::transform_points<double>(MatNx3Ref<double> points) const;

template std::pair<MatNxM<float>, VecN<float>>
pyslam::FrameBase::project_points<float>(const MatNx3Ref<float> points,
                                         bool do_stereo_project) const;
template std::pair<MatNxM<double>, VecN<double>>
pyslam::FrameBase::project_points<double>(const MatNx3Ref<double> points,
                                          bool do_stereo_project) const;

template std::pair<VecN<float>, float>
pyslam::FrameBase::project_point<float>(Vec3Ref<float> pw, bool do_stereo_project) const;
template std::pair<VecN<double>, double>
pyslam::FrameBase::project_point<double>(Vec3Ref<double> pw, bool do_stereo_project) const;

template std::pair<VecN<float>, float>
pyslam::FrameBase::project_map_point<float>(const MapPointPtr &map_point,
                                            bool do_stereo_project) const;
template std::pair<VecN<double>, double>
pyslam::FrameBase::project_map_point<double>(const MapPointPtr &map_point,
                                             bool do_stereo_project) const;

template std::tuple<bool, Vec2<float>, float>
pyslam::FrameBase::is_visible<float>(const MapPointPtr &map_point) const;
template std::tuple<bool, Vec2<double>, double>
pyslam::FrameBase::is_visible<double>(const MapPointPtr &map_point) const;

template std::tuple<std::vector<bool>, MatNxM<float>, VecN<float>, VecN<float>>
pyslam::FrameBase::are_visible<float>(const std::vector<MapPointPtr> &map_points,
                                      bool do_stereo_project) const;
template std::tuple<std::vector<bool>, MatNxM<double>, VecN<double>, VecN<double>>
pyslam::FrameBase::are_visible<double>(const std::vector<MapPointPtr> &map_points,
                                       bool do_stereo_project) const;

template std::pair<MatNxM<float>, VecN<float>>
pyslam::FrameBase::project_map_points<float>(const std::vector<MapPointPtr> &map_points,
                                             bool do_stereo_project) const;
template std::pair<MatNxM<double>, VecN<double>>
pyslam::FrameBase::project_map_points<double>(const std::vector<MapPointPtr> &map_points,
                                              bool do_stereo_project) const;

// Explicit template instantiations for Frame methods used in Python bindings
template std::pair<std::vector<Vec3<float>>, std::vector<bool>>
pyslam::Frame::unproject_points_3d<float>(const std::vector<int> &idxs,
                                          bool transform_in_world) const;
template std::pair<std::vector<Vec3<double>>, std::vector<bool>>
pyslam::Frame::unproject_points_3d<double>(const std::vector<int> &idxs,
                                           bool transform_in_world) const;

template std::pair<Vec3<float>, bool>
pyslam::Frame::unproject_point_3d<float>(int idx, bool transform_in_world) const;
template std::pair<Vec3<double>, bool>
pyslam::Frame::unproject_point_3d<double>(int idx, bool transform_in_world) const;

template float pyslam::Frame::compute_points_median_depth(MatNx3Ref<float> points3d,
                                                          const float percentile = 0.5f) const;
template double pyslam::Frame::compute_points_median_depth(MatNx3Ref<double> points3d,
                                                           const double percentile = 0.5) const;

} // namespace pyslam
