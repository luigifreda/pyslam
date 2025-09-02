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
#include "map_point.h"
#include "utils/features.h"
#include "utils/messages.h"
#include "utils/serialization.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

#include <pybind11/pybind11.h> // for gil_scoped_acquire and gil_scoped_release

namespace pyslam {

constexpr float kMinDepth = 1e-2f;

inline constexpr auto kDrawFeatureRadius = []() {
    std::array<int, 100> a{};
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<int>(i) * 5;
    return a;
}();

// colors from
// https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
constexpr std::array<std::array<float, 3>, 10> colors_myjet = {{{{0.0, 0.0, 0.5}},
                                                                {{0.0, 0.0, 0.99910873}},
                                                                {{0.0, 0.37843137, 1.0}},
                                                                {{0.0, 0.83333333, 1.0}},
                                                                {{0.30044276, 1.0, 0.66729918}},
                                                                {{0.66729918, 1.0, 0.30044276}},
                                                                {{1.0, 0.90123457, 0.0}},
                                                                {{1.0, 0.48002905, 0.0}},
                                                                {{0.99910873, 0.07334786, 0.0}},
                                                                {{0.5, 0.0, 0.0}}}};

constexpr float kChi2Mono =
    5.991; // chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
constexpr float kChi2Stereo =
    7.815; // chi-square 3 DOFs, used for reprojection error  (Hartley Zisserman pg 119)

// Static member definitions for FrameBase
std::atomic<int> FrameBase::_id{0};
std::mutex FrameBase::_id_lock;

// Static member definitions for Frame
bool Frame::is_store_imgs = false;
bool Frame::is_compute_median_depth = false;

// FrameBase Implementation
FrameBase::FrameBase(Camera* camera, CameraPose* pose, int id, double timestamp, int img_id)
    : camera(camera)
    , _pose(pose ? std::make_unique<CameraPose>(*pose) : std::make_unique<CameraPose>())
    , id(id == -1 ? []() { std::lock_guard<std::mutex> lock(_id_lock); return _id++; }() : id)
    , timestamp(timestamp)
    , img_id(img_id)
    , median_depth(-1.0f)
    , fov_center_c(Eigen::Vector3d::Zero())
    , fov_center_w(Eigen::Vector3d::Zero()) {
}

FrameBase::FrameBase(const FrameBase &other)
    : camera(other.camera), _pose(std::make_unique<CameraPose>(*other._pose)), id(other.id),
      timestamp(other.timestamp), img_id(other.img_id), median_depth(other.median_depth),
      fov_center_c(other.fov_center_c), fov_center_w(other.fov_center_w) {}

FrameBase &FrameBase::operator=(const FrameBase &other) {
    if (this != &other) {
        camera = other.camera;
        _pose = std::make_unique<CameraPose>(*other._pose);
        id = other.id;
        timestamp = other.timestamp;
        img_id = other.img_id;
        median_depth = other.median_depth;
        fov_center_c = other.fov_center_c;
        fov_center_w = other.fov_center_w;
    }
    return *this;
}

int FrameBase::next_id() {
    std::lock_guard<std::mutex> lock(_id_lock);
    return _id;
}

int FrameBase::width() const { return camera ? camera->width : 0; }

int FrameBase::height() const { return camera ? camera->height : 0; }

const Eigen::Isometry3d &FrameBase::isometry3d() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->isometry3d();
}

const Eigen::Matrix4d FrameBase::Tcw() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->get_matrix();
}

const Eigen::Matrix4d FrameBase::Twc() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->get_inverse_matrix();
}

const Eigen::Matrix3d FrameBase::Rcw() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->get_rotation_matrix();
}

const Eigen::Matrix3d FrameBase::Rwc() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->get_inverse_rotation_matrix();
}

const Eigen::Vector3d FrameBase::tcw() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->position();
}

const Eigen::Vector3d FrameBase::Ow() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->position();
}

const Eigen::Matrix4d FrameBase::pose() const { return Tcw(); }

const Eigen::Quaterniond FrameBase::quaternion() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->quaternion();
}

const Eigen::Quaterniond FrameBase::orientation() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->orientation();
}

const Eigen::Vector3d FrameBase::position() const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->position();
}

void FrameBase::update_pose(const CameraPose &pose) {
    std::lock_guard<std::mutex> lock(_lock_pose);
    _pose->set(pose.isometry3d());
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        // Don't call Rwc() and Ow() here - they also try to acquire the same lock
        // Instead, compute directly from the pose
        fov_center_w = _pose->get_inverse_rotation_matrix() * fov_center_c + _pose->position();
    }
}

void FrameBase::update_translation(const Eigen::Vector3d &tcw) {
    std::lock_guard<std::mutex> lock(_lock_pose);
    _pose->set_translation(tcw);
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        // Don't call Rwc() and Ow() here - they also try to acquire the same lock
        // Instead, compute directly from the pose
        fov_center_w = _pose->get_inverse_rotation_matrix() * fov_center_c + _pose->position();
    }
}

void FrameBase::update_rotation_and_translation(const Eigen::Matrix3d &Rcw,
                                                const Eigen::Vector3d &tcw) {
    std::lock_guard<std::mutex> lock(_lock_pose);
    _pose->set_from_rotation_and_translation(Rcw, tcw);
    if (fov_center_c != Eigen::Vector3d::Zero()) {
        // Don't call Rwc() and Ow() here - they also try to acquire the same lock
        // Instead, compute directly from the pose
        fov_center_w = _pose->get_inverse_rotation_matrix() * fov_center_c + _pose->position();
    }
}

Eigen::Vector3d FrameBase::transform_point(const Eigen::Vector3d &pw) const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    return _pose->get_rotation_matrix() * pw + _pose->position(); // p w.r.t. camera
}

MatNx3d FrameBase::transform_points(const MatNx3dRef points) const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    Eigen::Matrix3d Rcw_mat = _pose->get_rotation_matrix();
    Eigen::Vector3d tcw_vec = _pose->position();

    // Transform: (Rcw @ points.T + tcw).T
    MatNx3d result = (Rcw_mat * points.transpose()).transpose();
    result.rowwise() += tcw_vec.transpose();
    return result;
}

std::pair<MatNxMd, VecNd> FrameBase::project_points(const MatNx3dRef points,
                                                    bool do_stereo_project) const {
    auto pcs = transform_points(points);
    if (do_stereo_project) {
        return camera->project_stereo(pcs);
    } else {
        return camera->project(pcs);
    }
}

std::pair<MatNxMd, VecNd> FrameBase::project_map_points(const std::vector<MapPoint *> &map_points,
                                                        bool do_stereo_project) const {
    MatNx3d points;
    points.resize(map_points.size(), 3);
    for (int i = 0; i < map_points.size(); ++i) {
        const auto *mp = map_points[i];
        if (mp)
            points.row(i) = mp->pt();
    }
    return project_points(points, do_stereo_project);
}

std::pair<Eigen::Vector2d, double> FrameBase::project_point(const Eigen::Vector3d &pw) const {
    Eigen::Vector3d pc = transform_point(pw); // p w.r.t. camera
    return camera->project_point(pc);
}

std::pair<Eigen::Vector2d, double> FrameBase::project_map_point(const MapPoint *map_point) const {
    return project_point(map_point->pt());
}

bool FrameBase::is_in_image(const Eigen::Vector2d &uv, float z) const {
    return camera->is_in_image(uv, z);
}

std::vector<bool> FrameBase::are_in_image(const pyslam::MatNx2dRef uvs,
                                          const pyslam::VecNdRef zs) const {
    return camera->are_in_image(uvs, zs);
}

std::tuple<bool, Eigen::Vector2d, float> FrameBase::is_visible(const MapPoint *map_point) const {
    const auto [uv, z] = project_map_point(map_point);
    Eigen::Vector3d PO = map_point->pt() - Ow();

    if (!is_in_image(uv, static_cast<float>(z))) {
        return std::make_tuple(false, uv, static_cast<float>(z));
    }

    float dist3D = PO.norm();
    // point depth must be inside the scale pyramid of the image
    if (dist3D < map_point->min_distance() || dist3D > map_point->max_distance()) {
        return std::make_tuple(false, uv, static_cast<float>(z));
    }
    // viewing angle must be less than 60 deg
    if (PO.dot(map_point->get_normal()) <
        0.5 * dist3D) { // Parameters.kViewingCosLimitForPoint = 0.5
        return std::make_tuple(false, uv, static_cast<float>(z));
    }
    return std::make_tuple(true, uv, static_cast<float>(z));
}

std::tuple<std::vector<bool>, MatNx2d, VecNd, VecNd>
FrameBase::are_visible(const std::vector<MapPoint *> &map_points, bool do_stereo_project) const {
    MatNx3d points;
    MatNx3d normals;
    VecNd min_dists;
    VecNd max_dists;

    points.resize(map_points.size(), 3);
    normals.resize(map_points.size(), 3);
    min_dists.resize(map_points.size());
    max_dists.resize(map_points.size());

    for (size_t i = 0; i < map_points.size(); ++i) {
        const auto *mp = map_points[i];
        if (mp) {
            points.row(i) = mp->pt();
            normals.row(i) = mp->get_normal();
            float min_dist = mp->min_distance();
            float max_dist = mp->max_distance();
            min_dists(i) = min_dist;
            max_dists(i) = max_dist;
        }
    }

    auto projection_result = project_points(points, do_stereo_project);
    const auto &uvs = projection_result.first;
    const auto &zs_double = projection_result.second;

    Eigen::Vector3d Ow_vec = Ow();
    MatNx3d POs;
    VecNd dists;
    POs.resize(points.size(), 3);
    dists.resize(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        const auto &point = points.row(i);
        Eigen::Vector3d PO = point - Ow_vec.transpose();
        dists(i) = PO.norm();
        POs.row(i) = PO / dists(i);
    }

    VecNd cos_view;
    cos_view.resize(normals.size());
    for (size_t i = 0; i < normals.size(); ++i) {
        cos_view(i) = normals.row(i).dot(POs.row(i));
    }

    auto are_in_image_flags = are_in_image(uvs, zs_double);
    std::vector<bool> are_in_good_view_angle;
    are_in_good_view_angle.resize(cos_view.size());
    for (size_t i = 0; i < cos_view.size(); ++i) {
        are_in_good_view_angle[i] = cos_view(i) > 0.5 * dists(i);
    }

    std::vector<bool> are_in_good_distance;
    are_in_good_distance.resize(dists.size());
    for (size_t i = 0; i < dists.size(); ++i) {
        are_in_good_distance[i] = dists(i) >= min_dists(i) && dists(i) <= max_dists(i);
    }

    std::vector<bool> out_flags;
    out_flags.resize(are_in_image_flags.size());
    for (size_t i = 0; i < are_in_image_flags.size(); ++i) {
        out_flags[i] =
            are_in_image_flags[i] && are_in_good_view_angle[i] && are_in_good_distance[i];
    }

    // Convert zs_double to float for return
    VecNd zs_float;
    zs_float.resize(zs_double.size());
    for (size_t i = 0; i < zs_double.size(); ++i) {
        const auto z = zs_double(i);
        zs_float(i) = static_cast<float>(z);
    }

    return std::make_tuple(out_flags, uvs, zs_float, dists);
}

bool FrameBase::operator==(const FrameBase &rhs) const { return id == rhs.id; }

bool FrameBase::operator<(const FrameBase &rhs) const { return id < rhs.id; }

bool FrameBase::operator<=(const FrameBase &rhs) const { return id <= rhs.id; }

size_t FrameBase::hash() const { return std::hash<int>{}(id); }

// Frame Implementation
Frame::Frame(Camera *camera, const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth,
             CameraPose *pose, int id, double timestamp, int img_id, const cv::Mat &semantic_img,
             const std::map<std::string, void *> &frame_data_dict)
    : FrameBase(camera, pose, id, timestamp, img_id), img(img), img_right(img_right),
      depth_img(depth), semantic_img(semantic_img) {

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
    if (is_store_imgs && !img.empty()) {
        this->img = img.clone();
    }
    if (is_store_imgs && !img_right.empty()) {
        this->img_right = img_right.clone();
    }
    if (is_store_imgs && !depth.empty()) {
        if (camera->depth_factor != 1.0) {
            this->depth_img = depth * camera->depth_factor;
        } else {
            this->depth_img = depth.clone();
        }
    }
    if (is_store_imgs && !semantic_img.empty()) {
        this->semantic_img = semantic_img.clone();
    }

    if (!img.empty()) {
        this->manage_features(img, img_right);
    }

    const int N = kps.rows();
    if (N > 0) {
        if (!depth_img.empty()) {
            compute_stereo_from_rgbd(depth_img);
        } else if (!img_right.empty()) {
            depths.resize(N, -1.0);
            kps_ur.resize(N, -1.0);
            compute_stereo_matches(img, img_right);
        }
    }
}

Frame::Frame(const Frame &other)
    : FrameBase(other), kps(other.kps), kps_r(other.kps_r), kpsu(other.kpsu), kpsn(other.kpsn),
      kps_sem(other.kps_sem), octaves(other.octaves), octaves_r(other.octaves_r),
      sizes(other.sizes), angles(other.angles), des(other.des.clone()), des_r(other.des_r.clone()),
      depths(other.depths), kps_ur(other.kps_ur), points(other.points), outliers(other.outliers),
      kf_ref(other.kf_ref), img(other.img.clone()), img_right(other.img_right.clone()),
      depth_img(other.depth_img.clone()), semantic_img(other.semantic_img.clone()),
      is_blurry(other.is_blurry), laplacian_var(other.laplacian_var) {}

Frame &Frame::operator=(const Frame &other) {
    if (this != &other) {
        FrameBase::operator=(other);
        kps = other.kps;
        kps_r = other.kps_r;
        kpsu = other.kpsu;
        kpsn = other.kpsn;
        kps_sem = other.kps_sem;
        octaves = other.octaves;
        octaves_r = other.octaves_r;
        sizes = other.sizes;
        angles = other.angles;
        des = other.des.clone();
        des_r = other.des_r.clone();
        depths = other.depths;
        kps_ur = other.kps_ur;
        points = other.points;
        outliers = other.outliers;
        kf_ref = other.kf_ref;
        img = other.img.clone();
        img_right = other.img_right.clone();
        depth_img = other.depth_img.clone();
        semantic_img = other.semantic_img.clone();
        is_blurry = other.is_blurry;
        laplacian_var = other.laplacian_var;
    }
    return *this;
}

cv::flann::Index *Frame::kd() const {
    if (!_kd) {
        if (kpsu.size() > 0) {
            // Convert kpsu to cv::Mat for FLANN
            cv::Mat kpsu_mat(kpsu.size(), 2, CV_32F);
            for (size_t i = 0; i < kpsu.size(); ++i) {
                kpsu_mat.at<float>(i, 0) = kpsu(i, 0);
                kpsu_mat.at<float>(i, 1) = kpsu(i, 1);
            }
            // Use const_cast to modify the mutable member
            const_cast<Frame *>(this)->_kd =
                std::make_unique<cv::flann::Index>(kpsu_mat, cv::flann::KDTreeIndexParams());
        }
    }
    return _kd.get();
}

MapPoint *Frame::get_point_match(int idx) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    if (idx >= 0 && idx < static_cast<int>(points.size())) {
        return points[idx];
    }
    return nullptr;
}

void Frame::set_point_match(MapPoint *p, int idx) {
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

void Frame::replace_point_match(MapPoint *p, int idx) {
    if (idx >= 0 && idx < static_cast<int>(points.size())) {
        points[idx] = p; // replacing is not critical (it does not create a 'None jump')
    }
}

void Frame::remove_point(MapPoint *p) {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] == p) {
            points[i] = nullptr;
        }
    }
}

void Frame::remove_frame_views(const std::vector<int> &idxs) {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (int idx : idxs) {
        if (idx >= 0 && idx < static_cast<int>(points.size()) && points[idx]) {
            points[idx]->remove_frame_view(this, idx);
        }
    }
}

void Frame::reset_points() {
    std::lock_guard<std::mutex> lock(_lock_features);
    int num_keypoints = kps.size();
    points.assign(num_keypoints, nullptr);
    outliers.assign(num_keypoints, false);
}

std::vector<MapPoint *> Frame::get_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return points;
}

std::vector<MapPoint *> Frame::get_matched_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<MapPoint *> matched_points;
    for (MapPoint *p : points) {
        if (p) {
            matched_points.push_back(p);
        }
    }
    return matched_points;
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

std::vector<MapPoint *> Frame::get_matched_inlier_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<MapPoint *> matched_points;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && !outliers[i]) {
            matched_points.push_back(points[i]);
        }
    }
    return matched_points;
}

std::vector<MapPoint *> Frame::get_matched_good_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<MapPoint *> good_points;
    for (const auto &p : points) {
        if (p && !p->is_bad()) {
            good_points.push_back(p);
        }
    }
    return good_points;
}

std::pair<std::vector<int>, std::vector<MapPoint *>>
Frame::get_matched_good_points_with_idxs() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<int> idxs;
    std::vector<MapPoint *> good_points;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && !points[i]->is_bad()) {
            idxs.push_back(i);
            good_points.push_back(points[i]);
        }
    }
    return std::make_pair(idxs, good_points);
}

int Frame::num_tracked_points(int minObs) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    int count = 0;
    for (const auto &p : points) {
        if (p && !p->is_bad() && p->is_good_with_min_obs(minObs)) {
            count++;
        }
    }
    return count;
}

int Frame::num_matched_inlier_map_points() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    int count = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && !outliers[i] && points[i]->num_observations() > 0) {
            count++;
        }
    }
    return count;
}

int Frame::clean_outlier_map_points() {
    std::lock_guard<std::mutex> lock(_lock_features);
    int cleaned = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && outliers[i]) {
            if (points[i]->num_observations() > 0) {
                // Keep the point but remove the match
                points[i] = nullptr;
                cleaned++;
            }
        }
    }
    return cleaned;
}

void Frame::clean_bad_map_points() {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && points[i]->is_bad()) {
            points[i] = nullptr;
        }
    }
}

void Frame::clean_vo_matches() {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] && points[i]->num_observations() < 1) {
            points[i] = nullptr;
        }
    }
}

void Frame::check_replaced_map_points() {
    std::lock_guard<std::mutex> lock(_lock_features);
    int num_replaced_points = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i]) {
            MapPoint *replacement = points[i]->get_replacement();
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
    constexpr double kMinDepth = 1e-2;
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
            valid_depth_mask[i] = (depth_val > kMinDepth) && std::isfinite(depth_val);
            if (valid_depth_mask[i]) {
                depths[i] = static_cast<float>(depth_val);
                kps_ur[i] = static_cast<float>(kpsu(i, 0) - camera->bf / depth_val);
                valid_depths.push_back(depths[i]);
            }
        }
    }
}

void Frame::compute_stereo_from_rgbd(const cv::Mat &depth) {
    constexpr double kMinDepth = 1e-2;

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
    } else if (depth.type() == CV_64F) {
        extract_depth_values<double>(depth, valid_depth_mask, valid_depths);
    } else if (depth.type() == CV_16F) {
        extract_depth_values<uint16_t>(depth, valid_depth_mask, valid_depths);
    } else {
        MSG_ERROR("Frame::compute_stereo_from_rgbd() - unknown depth type");
        extract_depth_values<float>(depth, valid_depth_mask, valid_depths);
    }

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
            fov_center_c = camera->unproject_point_3d(camera->cx, camera->cy, median_depth);

            // Transform to world coordinates: fov_center_w = Rwc @ fov_center_c + Ow
            fov_center_w = Rwc() * fov_center_c + Ow();
        } else {
            median_depth = 0.0f;
            fov_center_c = Eigen::Vector3d::Zero();
            fov_center_w = Eigen::Vector3d::Zero();
        }
    }
}

void Frame::compute_stereo_matches(const cv::Mat &img, const cv::Mat &img_right) {
    constexpr double kRatioTest = 0.9;
    constexpr bool kRowMatching = true;
    const double min_z = camera->b;
    const double min_disparity = 0;
    const double max_disparity = camera->bf / min_z;

    if (!FeatureSharedInfo::stereo_matching_callback) {
        MSG_ERROR("Frame::compute_stereo_matches() - stereo_matching_callback not set");
        return;
    }

    const auto &[idxs1, idxs2] = FeatureSharedInfo::stereo_matching_callback(
        img, img_right, des, des_r, kps, kps_r, kRatioTest, kRowMatching, max_disparity);
    if (idxs1.size() == 0 || idxs2.size() == 0) {
        MSG_WARN("Frame::compute_stereo_matches() - no stereo matches found");
        return;
    }

    // Filter by good disparities
    std::vector<int> good_matched_idxs1, good_matched_idxs2;
    std::vector<double> good_disparities;
    good_matched_idxs1.reserve(idxs1.size());
    good_matched_idxs2.reserve(idxs2.size());
    good_disparities.reserve(idxs1.size());

    for (size_t i = 0; i < idxs1.size(); ++i) {
        double disparity = kps(idxs1[i], 0) - kps_r(idxs2[i], 0);
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
        constexpr double fmat_err_thld = 3.0;
        constexpr double confidence = 0.999;
#if OPENCV_VERSION_MAJOR >= 4
        constexpr int ransac_method = cv::USAC_MAGSAC;
#else
        constexpr int ransac_method = cv::RANSAC;
#endif
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, ransac_method, fmat_err_thld, confidence,
                                           mask_inliers);

        std::vector<int> filtered_idxs1, filtered_idxs2;
        std::vector<double> filtered_disparities;

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
            stereo_match_subpixel_correlation(good_matched_idxs1, good_matched_idxs2, kps, kps_r,
                                              min_disparity, max_disparity, img_bw, img_right_bw);

        // Update arrays with refined results
        std::vector<int> final_idxs1, final_idxs2;
        std::vector<double> final_disparities;
        final_idxs1.reserve(valid_idxs.size());
        final_idxs2.reserve(valid_idxs.size());
        final_disparities.reserve(valid_idxs.size());

        for (size_t i = 0; i < valid_idxs.size(); ++i) {
            int idx = valid_idxs[i];
            final_idxs1.push_back(good_matched_idxs1[idx]);
            final_idxs2.push_back(good_matched_idxs2[idx]);
            final_disparities.push_back(refined_disparities[i]);

            // Update kps_ur with refined right u-coordinates
            kps_ur[good_matched_idxs1[idx]] = refined_us_right[i];
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
        Eigen::Matrix4d pose_l = Eigen::Matrix4d::Identity();
        Eigen::Vector3d t_rl(-camera->b, 0, 0);
        Eigen::Matrix4d pose_rl = Eigen::Matrix4d::Identity();
        pose_rl.block<3, 1>(0, 3) = t_rl;

        // Convert keypoints to normalized coordinates
        std::vector<Eigen::Vector2d> kpsn_l, kpsn_r;
        kpsn_l.reserve(good_matched_idxs1.size());
        kpsn_r.reserve(good_matched_idxs2.size());

        Eigen::Matrix3d Kinv = camera->Kinv;
        for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
            const Eigen::Vector3d kp_l(kps(good_matched_idxs1[i], 0), kps(good_matched_idxs1[i], 1),
                                       1.0);
            const Eigen::Vector3d kp_r(kps_r(good_matched_idxs2[i], 0),
                                       kps_r(good_matched_idxs2[i], 1), 1.0);
            kpsn_l.emplace_back((Kinv * kp_l).head<2>());
            kpsn_r.emplace_back((Kinv * kp_r).head<2>());
        }

        // Triangulate points:
        // Using the depth from the disparity to back-project to 3D points from left image
        // Then project the 3D points to right image to check the reprojection errors
        std::vector<Eigen::Vector3d> pts3d;
        pts3d.reserve(good_matched_idxs1.size());

        for (size_t i = 0; i < good_matched_idxs1.size(); ++i) {
            const double disparity = good_disparities[i];
            if (disparity > 0) {
                const double depth = camera->bf / disparity;
                Eigen::Vector3d pt3d(kpsn_l[i].x() * depth, kpsn_l[i].y() * depth, depth);
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
                Eigen::Vector3d pt3d_r = pts3d[i] + t_rl;

                // auto [uv_l, depth_l] = camera->project_point(pts3d[i]);
                auto [uv_r, depth_r] = camera->project_point(pt3d_r);

                // if (depth_l > 0 && depth_r > 0) {
                if (depth_r > 0) {
                    // Compute reprojection errors
                    // Eigen::Vector2d err_l = uv_l - Eigen::Vector2d(kps(good_matched_idxs1[i], 0),
                    // kps(good_matched_idxs1[i], 1));
                    Eigen::Vector2d err_r = uv_r - Eigen::Vector2d(kps_r(good_matched_idxs2[i], 0),
                                                                   kps_r(good_matched_idxs2[i], 1));

                    // double err_l_sqr = err_l.squaredNorm();
                    double err_r_sqr = err_r.squaredNorm();

                    // Get sigma from octave levels
                    // double sigma_l = std::pow(2.0, octaves[good_matched_idxs1[i]]);
                    double sigma_r = std::pow(2.0, octaves_r[good_matched_idxs2[i]]);

                    // double chi2_l = err_l_sqr / (sigma_l * sigma_l);
                    double chi2_r = err_r_sqr / (sigma_r * sigma_r);

                    // good_chi2_mask.push_back(chi2_l < kChi2Mono && chi2_r < kChi2Mono);
                    good_chi2_mask.push_back(chi2_r < kChi2Mono);
                } else {
                    good_chi2_mask.push_back(false);
                }
            } else {
                good_chi2_mask.push_back(false);
            }
        }

        // Filter by chi-squared results
        std::vector<int> filtered_idxs1, filtered_idxs2;
        std::vector<double> filtered_disparities;

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
        const int idx = good_matched_idxs1[i];
        depths[idx] = camera->bf / good_disparities[i];
        if (!do_subpixel_stereo_matching) {
            kps_ur[idx] = kps_r(good_matched_idxs2[i], 0);
        }
    }

    std::cout << "Frame::compute_stereo_matches() - found final " << good_matched_idxs1.size()
              << " stereo matches" << std::endl;

    // Compute median depth if enabled
    if (is_compute_median_depth) {
        std::vector<float> valid_depths;
        valid_depths.reserve(depths.size());

        for (float depth : depths) {
            if (depth > kMinDepth) {
                valid_depths.push_back(depth);
            }
        }

        if (!valid_depths.empty()) {
            std::sort(valid_depths.begin(), valid_depths.end());
            median_depth = valid_depths[valid_depths.size() / 2];

            // Compute FOV center
            fov_center_c = camera->unproject_point_3d(camera->cx, camera->cy, median_depth);
            fov_center_w = _pose->Rwc() * fov_center_c + _pose->Ow();

            std::cout << "Frame::compute_stereo_matches() - median depth: " << median_depth
                      << std::endl;
        }
    }
}

std::pair<Eigen::Vector3d, bool> Frame::unproject_point_3d(int idx, bool transform_in_world) const {
    const double &depth = depths[idx];
    if (depth > kMinDepth) {
        Eigen::Vector3d pt3d(depth * kpsn(idx, 0), depth * kpsn(idx, 1), depth);
        if (transform_in_world) {
            pt3d = _pose->Rwc() * pt3d + _pose->Ow();
        }
        return std::make_pair(pt3d, true);
    } else {
        return std::make_pair(Eigen::Vector3d(), false);
    }
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<bool>>
Frame::unproject_points_3d(const std::vector<int> &idxs, bool transform_in_world) const {
    std::vector<Eigen::Vector3d> pts3d;
    pts3d.reserve(idxs.size());
    std::vector<bool> valid_mask(idxs.size(), false);

    for (const int idx : idxs) {
        const double &depth = depths[idx];
        if (depth > kMinDepth) {
            Eigen::Vector3d pt3d(depth * kpsn(idx, 0), depth * kpsn(idx, 1), depth);
            if (transform_in_world) {
                pt3d = _pose->Rwc() * pt3d + _pose->Ow();
            }
            pts3d.emplace_back(pt3d);
            valid_mask[idx] = true;
        }
    }
    return std::make_pair(pts3d, valid_mask);
}

float Frame::compute_points_median_depth(const std::vector<Eigen::Vector3d> &points3d,
                                         float percentile) const {
    Eigen::Vector3d Rcw2;
    double tcw2;
    {
        std::lock_guard<std::mutex> lock(_lock_pose);
        Rcw2 = _pose->Rcw().row(2); // just 2-nd row
        tcw2 = _pose->tcw()[2];     // just 2-nd element (z-axis)
    }
    std::vector<Eigen::Vector3d> points3d_frame;
    if (points3d.empty()) {
        // we use points array (from Frame)
        for (const auto &mp : points) {
            if (mp) {
                points3d_frame.emplace_back(mp->pt());
            }
        }
    }
    const auto &points3d_to_use = points3d.empty() ? points3d_frame : points3d;

    if (!points3d_to_use.empty()) {
        std::vector<double> depths;
        depths.reserve(points3d_to_use.size());
        for (const auto &pt3d : points3d_to_use) {
            const double zc = Rcw2.dot(pt3d) + tcw2;
            depths.push_back(zc);
        }

        std::sort(depths.begin(), depths.end());
        const size_t idx =
            std::min(static_cast<size_t>(depths.size() * percentile), depths.size() - 1);
        return depths[idx];
    } else {
        return -1.0f;
    }
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
void extract_semantic_at_keypoints_impl(const cv::Mat &semantic_img, const Eigen::MatrixXd &kps,
                                        cv::Mat &kps_sem) {
    const int num_kps = static_cast<int>(kps.rows());
    const int channels = semantic_img.channels();
    kps_sem = cv::Mat(num_kps, channels, semantic_img.type());

    for (int i = 0; i < num_kps; ++i) {
        const int u = static_cast<int>(kps(i, 0));
        const int v = static_cast<int>(kps(i, 1));

        if (u >= 0 && u < semantic_img.cols && v >= 0 && v < semantic_img.rows) {
            if constexpr (is_single_channel) {
                // semantic_img is single-channel with depth T
                kps_sem.at<T>(i, 0) = semantic_img.at<T>(v, u);
            } else {
                // semantic_img is interleaved multi-channel with depth T
                const T *pixel_ptr = semantic_img.ptr<T>(v) + u * channels;
                // copy channel values into output row i
                for (int c = 0; c < channels; ++c) {
                    kps_sem.at<T>(i, c) = pixel_ptr[c];
                }
            }
        } else {
            // Default values for out-of-bounds => fill with 0
            for (int c = 0; c < channels; ++c) {
                kps_sem.at<T>(i, c) = 0;
            }
        }
    }
}

void extract_semantic_at_keypoints(const cv::Mat &semantic_img, const Eigen::MatrixXd &kps,
                                   cv::Mat &kps_sem) {
    const int depth = semantic_img.depth();
    const bool single = (semantic_img.channels() == 1);

    switch (depth) {
    case CV_8U:
        single ? extract_semantic_at_keypoints_impl<true, uchar>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, uchar>(semantic_img, kps, kps_sem);
        break;
    case CV_16U:
        single ? extract_semantic_at_keypoints_impl<true, ushort>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, ushort>(semantic_img, kps, kps_sem);
        break;
    case CV_32F:
        single ? extract_semantic_at_keypoints_impl<true, float>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, float>(semantic_img, kps, kps_sem);
        break;
    case CV_64F:
        single ? extract_semantic_at_keypoints_impl<true, double>(semantic_img, kps, kps_sem)
               : extract_semantic_at_keypoints_impl<false, double>(semantic_img, kps, kps_sem);
        break;
    default:
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported image depth for semantics");
    }
}

void Frame::set_semantics(const cv::Mat &semantic_img) {
    if (is_store_imgs) {
        this->semantic_img = semantic_img.clone();
    }
    if (kps.size() > 0) {
        std::lock_guard<std::mutex> lock(_lock_features);

        // Extract semantic information at keypoint locations
        extract_semantic_at_keypoints(semantic_img, kps, kps_sem);

        // Ensure contiguous memory layout (equivalent to np.ascontiguousarray)
        if (!kps_sem.isContinuous()) {
            kps_sem = kps_sem.clone();
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
}

cv::Mat Frame::draw_feature_trails(const cv::Mat &img, const std::vector<int> &kps_idxs,
                                   int trail_max_length) const {
    cv::Mat img_out = img.clone();
    std::lock_guard<std::mutex> lock(_lock_features);

    // use distorted coordinates when drawing on distorted original image
    for (const int idx : kps_idxs) {
        const auto &kp = kps.row(idx);
        const auto uv = Eigen::Vector2i(std::floor(kp[0]), std::floor(kp[1]));

        const int radius = kDrawFeatureRadius[octaves[idx]];
        const auto &mp = points[idx];
        if (mp && !mp->is_bad()) {
            // there is a corresponding 3D map point
            const auto &p_frame_views = mp->frame_views(); // list of (Frame, idx)
            if (!p_frame_views.empty()) {
                const auto &color =
                    (p_frame_views.size() > 2) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
                cv::circle(img_out, cv::Point(uv[0], uv[1]), radius, color, 1);
                // draw the trail (for each keypoint, its trail_max_length corresponding points
                // in previous frames)
                std::vector<cv::Point> pts;
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
                if (pts.size() > 1) {
                    const auto &color = colors_myjet[pts.size()];
                    const auto &cv_color = cv::Scalar(color[0], color[1], color[2]) * 255;
                    cv::polylines(img_out, pts, false, cv_color, 1, cv::LINE_AA);
                }
            }
        } else {
            // no corresponding 3D map point
            cv::circle(img_out, cv::Point(uv[0], uv[1]), 2, cv::Scalar(0, 0, 0));
        }
    }
    return img_out;
}

cv::Mat Frame::draw_all_feature_trails(const cv::Mat &img) const {
    std::vector<int> all_idxs(kps.size());
    std::iota(all_idxs.begin(), all_idxs.end(), 0);
    return draw_feature_trails(img, all_idxs);
}

void Frame::manage_features(const cv::Mat &img, const cv::Mat &img_right) {
    constexpr bool kVerbose = true;
    if (FeatureSharedInfo::feature_detect_and_compute_callback) {
        if constexpr (kVerbose) {
            std::cout << "Frame::manage_features() called" << std::endl;
        }
        // const auto &[keypoints_, des_] =
        // FeatureSharedInfo::feature_detect_and_compute_callback(img);

        // Call Python feature detection for left and right images
        const auto &[result_left, result_right] =
            detect_and_compute_features_parallel(img, img_right);

        const auto &[keypoints_, des_] = result_left;
        des = des_;

        // Convert keypoints to internal format
        kps.resize(keypoints_.size(), 2);
        octaves.resize(keypoints_.size());
        sizes.resize(keypoints_.size());
        angles.resize(keypoints_.size());

        // cv::KeyPoint <-> (pt.x, pt.y, size, angle, response, octave)

        for (size_t i = 0; i < keypoints_.size(); ++i) {
            const auto &kp = keypoints_[i];
            kps(i, 0) = std::get<0>(kp);
            kps(i, 1) = std::get<1>(kp);
            sizes[i] = std::get<2>(kp);
            angles[i] = std::get<3>(kp);
            // response is not used
            octaves[i] = std::get<5>(kp);
        }

        // Continue with undistortion and normalization
        if (camera && kps.size() > 0) {
            // Convert to undistorted coordinates
            kpsu = camera->undistort_points(kps);
            kpsn = camera->unproject_points(kpsu);

            // Initialize point arrays
            int num_kps = kps.size();
            points.assign(num_kps, nullptr);
            outliers.assign(num_kps, false);
        }

        if constexpr (kVerbose) {
            std::cout << "Frame::manage_features() - found " << kps.size() << " keypoints"
                      << std::endl;
        }

        // Handle stereo if needed
        if (!img_right.empty()) {
            // const auto &[keypoints_r_, des_r_] =
            // feature_detect_and_compute_right_callback(img_right);

            const auto &[keypoints_r_, des_r_] = result_right;
            des_r = des_r_;

            kps_r.resize(keypoints_r_.size(), 2);
            octaves_r.resize(keypoints_r_.size());

            for (size_t i = 0; i < keypoints_r_.size(); ++i) {
                const auto &kp = keypoints_r_[i];
                kps_r(i, 0) = std::get<0>(kp);
                kps_r(i, 1) = std::get<1>(kp);
                // sizes_r[i] = std::get<2>(kp);
                // angles_r[i] = std::get<3>(kp);
                // response is not used
                octaves_r[i] = std::get<5>(kp);
            }

            if constexpr (kVerbose) {
                std::cout << "Frame::manage_features() - found " << kps_r.size()
                          << " right keypoints" << std::endl;
            }
        }
    }
}

std::pair<FeatureDetectAndComputeOutput, FeatureDetectAndComputeOutput>
Frame::detect_and_compute_features_parallel(const cv::Mat &img, const cv::Mat &img_right) {
    constexpr bool kVerbose = false; // Fix: reduce logging

    if (FeatureSharedInfo::feature_detect_and_compute_callback &&
        FeatureSharedInfo::feature_detect_and_compute_right_callback) {
        // Parallel detection with GIL management
        if constexpr (kVerbose) {
            std::cout << "Frame::detect_and_compute_features_parallel() - parallel detection with "
                         "GIL management"
                      << std::endl;
        }

        // Release GIL and run parallel detection
        FeatureDetectAndComputeOutput result_left, result_right;

        {
            pybind11::gil_scoped_release release; // Release GIL for parallel execution

            auto left_future =
                std::async(std::launch::async, [this, &img]() -> FeatureDetectAndComputeOutput {
                    if (img.empty()) {
                        return FeatureDetectAndComputeOutput();
                    }
                    pybind11::gil_scoped_acquire acquire; // Acquire GIL just for this callback
                    return FeatureSharedInfo::feature_detect_and_compute_callback(img);
                });

            auto right_future = std::async(
                std::launch::async, [this, &img_right]() -> FeatureDetectAndComputeOutput {
                    if (img_right.empty()) {
                        return FeatureDetectAndComputeOutput();
                    }
                    pybind11::gil_scoped_acquire acquire; // Acquire GIL just for this callback
                    return FeatureSharedInfo::feature_detect_and_compute_right_callback(img_right);
                });

            // Wait for both to complete (GIL is still released here)
            result_left = left_future.get();
            result_right = right_future.get();
        }
        // GIL is automatically reacquired when leaving the scope

        return {result_left, result_right};

    } else if (FeatureSharedInfo::feature_detect_and_compute_callback) {
        // Single left image callback - no need for GIL management
        if constexpr (kVerbose) {
            std::cout << "Frame::detect_and_compute_features_parallel() - just detect on left image"
                      << std::endl;
        }
        if (img.empty()) {
            return {FeatureDetectAndComputeOutput(), FeatureDetectAndComputeOutput()};
        } else {
            return {FeatureSharedInfo::feature_detect_and_compute_callback(img),
                    FeatureDetectAndComputeOutput()};
        }
    } else {
        throw std::runtime_error("Feature detection and compute callbacks are not set");
    }
}

} // namespace pyslam
