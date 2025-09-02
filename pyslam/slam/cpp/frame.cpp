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

#include <algorithm>
#include <cmath>

namespace pyslam {

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

std::vector<Eigen::Vector3d>
FrameBase::transform_points(const std::vector<Eigen::Vector3d> &points) const {
    std::lock_guard<std::mutex> lock(_lock_pose);
    Eigen::Matrix3d Rcw_mat = _pose->get_rotation_matrix();
    Eigen::Vector3d tcw_vec = _pose->position();

    std::vector<Eigen::Vector3d> result;
    result.reserve(points.size());
    for (const auto &point : points) {
        result.push_back(Rcw_mat * point + tcw_vec);
    }
    return result;
}

std::pair<std::vector<cv::Point2f>, std::vector<double>>
FrameBase::project_points(const std::vector<Eigen::Vector3d> &points,
                          bool do_stereo_project) const {
    auto pcs = transform_points(points);
    if (do_stereo_project) {
        // project_stereo returns Point3f, we need to convert to Point2f
        auto stereo_result = camera->project_stereo(pcs);
        std::vector<cv::Point2f> points2f;
        points2f.reserve(stereo_result.first.size());
        for (const auto &p3f : stereo_result.first) {
            points2f.emplace_back(p3f.x, p3f.y);
        }
        return std::make_pair(points2f, stereo_result.second);
    } else {
        return camera->project(pcs);
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<double>>
FrameBase::project_map_points(const std::vector<MapPoint *> &map_points,
                              bool do_stereo_project) const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(map_points.size());
    for (const auto *mp : map_points) {
        if (mp)
            points.push_back(mp->pt());
    }
    return project_points(points, do_stereo_project);
}

std::pair<cv::Point2f, double> FrameBase::project_point(const Eigen::Vector3d &pw) const {
    Eigen::Vector3d pc = transform_point(pw); // p w.r.t. camera
    // Convert single Vector3d to vector for camera->project()
    std::vector<Eigen::Vector3d> pcs = {pc};
    auto result = camera->project(pcs);
    return std::make_pair(result.first[0], result.second[0]);
}

std::pair<cv::Point2f, double> FrameBase::project_map_point(const MapPoint *map_point) const {
    return project_point(map_point->pt());
}

bool FrameBase::is_in_image(const cv::Point2f &uv, float z) const {
    return camera->is_in_image(uv, z);
}

std::vector<bool> FrameBase::are_in_image(const std::vector<cv::Point2f> &uvs,
                                          const std::vector<double> &zs) const {
    return camera->are_in_image(uvs, zs);
}

std::tuple<bool, cv::Point2f, float> FrameBase::is_visible(const MapPoint *map_point) const {
    cv::Point2f uv;
    double z;
    std::tie(uv, z) = project_map_point(map_point);
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

std::tuple<std::vector<bool>, std::vector<cv::Point2f>, std::vector<float>, std::vector<float>>
FrameBase::are_visible(const std::vector<MapPoint *> &map_points, bool do_stereo_project) const {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<float> min_dists;
    std::vector<float> max_dists;

    points.reserve(map_points.size());
    normals.reserve(map_points.size());
    min_dists.reserve(map_points.size());
    max_dists.reserve(map_points.size());

    for (const auto *mp : map_points) {
        if (mp) {
            points.push_back(mp->pt());
            normals.push_back(mp->get_normal());
            float min_dist = mp->min_distance();
            float max_dist = mp->max_distance();
            min_dists.push_back(min_dist);
            max_dists.push_back(max_dist);
        }
    }

    auto projection_result = project_points(points, do_stereo_project);
    const auto &uvs = projection_result.first;
    const auto &zs_double = projection_result.second;

    Eigen::Vector3d Ow_vec = Ow();
    std::vector<Eigen::Vector3d> POs;
    std::vector<float> dists;
    POs.reserve(points.size());
    dists.reserve(points.size());

    for (const auto &point : points) {
        Eigen::Vector3d PO = point - Ow_vec;
        dists.push_back(PO.norm());
        POs.push_back(PO / dists.back());
    }

    std::vector<float> cos_view;
    cos_view.reserve(normals.size());
    for (size_t i = 0; i < normals.size(); ++i) {
        cos_view.push_back(normals[i].dot(POs[i]));
    }

    auto are_in_image_flags = are_in_image(uvs, zs_double);
    std::vector<bool> are_in_good_view_angle;
    are_in_good_view_angle.reserve(cos_view.size());
    for (size_t i = 0; i < cos_view.size(); ++i) {
        are_in_good_view_angle.push_back(cos_view[i] > 0.5 * dists[i]);
    }

    std::vector<bool> are_in_good_distance;
    are_in_good_distance.reserve(dists.size());
    for (size_t i = 0; i < dists.size(); ++i) {
        are_in_good_distance.push_back(dists[i] >= min_dists[i] && dists[i] <= max_dists[i]);
    }

    std::vector<bool> out_flags;
    out_flags.reserve(are_in_image_flags.size());
    for (size_t i = 0; i < are_in_image_flags.size(); ++i) {
        out_flags.push_back(are_in_image_flags[i] && are_in_good_view_angle[i] &&
                            are_in_good_distance[i]);
    }

    // Convert zs_double to float for return
    std::vector<float> zs_float;
    zs_float.reserve(zs_double.size());
    for (double z : zs_double) {
        zs_float.push_back(static_cast<float>(z));
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
    kps.clear();
    kps_r.clear();
    kpsu.clear();
    kpsn.clear();
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
        this->depth_img = depth.clone();
    }
    if (is_store_imgs && !semantic_img.empty()) {
        this->semantic_img = semantic_img.clone();
    }

    // TODO: Implement feature detection and matching logic here
    // This would involve calling the feature tracker to detect keypoints and
    // compute descriptors
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
        if (!kpsu.empty()) {
            // Convert kpsu to cv::Mat for FLANN
            cv::Mat kpsu_mat(kpsu.size(), 2, CV_32F);
            for (size_t i = 0; i < kpsu.size(); ++i) {
                kpsu_mat.at<float>(i, 0) = kpsu[i].x;
                kpsu_mat.at<float>(i, 1) = kpsu[i].y;
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

void Frame::compute_stereo_from_rgbd(const std::vector<cv::Point2f> &kps_data,
                                     const cv::Mat &depth) {
    // TODO: Implement stereo computation from RGB-D
    // This would involve extracting depth values at keypoint locations
    // and computing stereo disparities
}

void Frame::compute_stereo_matches(const cv::Mat &img, const cv::Mat &img_right) {
    // TODO: Implement stereo matching
    // This would involve matching features between left and right images
    // and computing disparities
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<bool>>
Frame::unproject_points_3d(const std::vector<int> &idxs, bool transform_in_world) const {
    // TODO: Implement 3D point unprojection
    // This would involve converting 2D keypoints to 3D points using depth
    // information
    return std::make_pair(std::vector<Eigen::Vector3d>(), std::vector<bool>());
}

float Frame::compute_points_median_depth(const std::vector<Eigen::Vector3d> *points3d,
                                         float percentile) const {
    // TODO: Implement median depth computation
    // This would involve computing depths of 3D points and finding the median
    return 0.0f;
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

void Frame::set_semantics(const cv::Mat &semantic_img) {
    if (is_store_imgs) {
        this->semantic_img = semantic_img.clone();
    }
    if (!kps.empty()) {
        std::lock_guard<std::mutex> lock(_lock_features);
        // TODO: Extract semantic information at keypoint locations
        // This would involve sampling the semantic image at keypoint coordinates
    }
}

void Frame::ensure_contiguous() {
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
    // TODO: Implement feature trail drawing
    // This would involve drawing keypoints and their tracking trails
    return img.clone();
}

cv::Mat Frame::draw_all_feature_trails(const cv::Mat &img) const {
    std::vector<int> all_idxs(kps.size());
    std::iota(all_idxs.begin(), all_idxs.end(), 0);
    return draw_feature_trails(img, all_idxs);
}

std::string Frame::to_json() const {
    // TODO: Implement JSON serialization
    // This would involve converting all frame data to JSON format
    return "{}";
}

Frame Frame::from_json(const std::string &json_str) {
    // TODO: Implement JSON deserialization
    // This would involve parsing JSON and reconstructing the frame
    return Frame(nullptr);
}

void Frame::replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                     const std::vector<Frame *> &frames,
                                     const std::vector<KeyFrame *> &keyframes) {
    // TODO: Implement ID replacement with actual objects
    // This would involve replacing saved IDs with reloaded objects after
    // deserialization
}

} // namespace pyslam
