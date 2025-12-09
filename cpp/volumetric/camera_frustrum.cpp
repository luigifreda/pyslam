/**
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

#include "camera_frustrum.h"

#include <limits>

namespace volumetric {

CameraFrustrum::CameraFrustrum(const Eigen::Matrix3d &K, const int width, const int height,
                               const Eigen::Matrix4d &T_cw, const float depth_max,
                               const float depth_min)
    : width_(width), height_(height), fx_(K(0, 0)), fy_(K(1, 1)), cx_(K(0, 2)), cy_(K(1, 2)),
      R_cw_(T_cw.block<3, 3>(0, 0)), t_cw_(T_cw.block<3, 1>(0, 3)), depth_max_(depth_max),
      depth_min_(depth_min) {
    update_cache_();
}

CameraFrustrum::CameraFrustrum(const float fx, const float fy, const float cx, const float cy,
                               const int width, const int height,
                               const Eigen::Quaterniond &orientation,
                               const Eigen::Vector3d &translation, const float depth_max,
                               const float depth_min)
    : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
      R_cw_(orientation.normalized().toRotationMatrix()), t_cw_(translation), depth_max_(depth_max),
      depth_min_(depth_min) {
    update_cache_();
}

CameraFrustrum::CameraFrustrum(const float fx, const float fy, const float cx, const float cy,
                               const int width, const int height, const Eigen::Matrix4d &T_cw,
                               const float depth_max, const float depth_min)
    : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
      R_cw_(T_cw.block<3, 3>(0, 0)), t_cw_(T_cw.block<3, 1>(0, 3)), depth_max_(depth_max),
      depth_min_(depth_min) {
    update_cache_();
}

void CameraFrustrum::set_width(const int width) {
    width_ = width;
    cache_is_valid_ = false;
}

void CameraFrustrum::set_height(const int height) {
    height_ = height;
    cache_is_valid_ = false;
}

void CameraFrustrum::set_depth_max(const float depth_max) {
    depth_max_ = depth_max;
    cache_is_valid_ = false;
}

void CameraFrustrum::set_depth_min(const float depth_min) {
    depth_min_ = depth_min;
    cache_is_valid_ = false;
}

void CameraFrustrum::set_intrinsics(const Eigen::Matrix3d &K) {
    fx_ = K(0, 0);
    fy_ = K(1, 1);
    cx_ = K(0, 2);
    cy_ = K(1, 2);
    cache_is_valid_ = false;
}

void CameraFrustrum::set_intrinsics(const float fx, const float fy, const float cx,
                                    const float cy) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    cache_is_valid_ = false;
}

void CameraFrustrum::set_T_cw(const Eigen::Matrix4d &T_cw) {
    R_cw_ = T_cw.block<3, 3>(0, 0);
    t_cw_ = T_cw.block<3, 1>(0, 3);
    cache_is_valid_ = false;
}

void CameraFrustrum::set_T_cw(const Eigen::Quaterniond &orientation,
                              const Eigen::Vector3d &translation) {
    R_cw_ = orientation.normalized().toRotationMatrix();
    t_cw_ = translation;
    cache_is_valid_ = false;
}

int CameraFrustrum::get_width() const { return width_; }
int CameraFrustrum::get_height() const { return height_; }
float CameraFrustrum::get_fx() const { return fx_; }
float CameraFrustrum::get_fy() const { return fy_; }
float CameraFrustrum::get_cx() const { return cx_; }
float CameraFrustrum::get_cy() const { return cy_; }

Eigen::Matrix3d CameraFrustrum::get_K() const {
    Eigen::Matrix3d K;
    K << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
    return K;
}

Eigen::Matrix4d CameraFrustrum::get_T_cw() const {
    Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
    T_cw.block<3, 3>(0, 0) = R_cw_;
    T_cw.block<3, 1>(0, 3) = t_cw_;
    return T_cw;
}

const Eigen::Matrix3d &CameraFrustrum::get_R_cw() const { return R_cw_; }

const Eigen::Quaterniond CameraFrustrum::get_orientation_cw() const {
    return Eigen::Quaterniond(R_cw_);
}

const Eigen::Vector3d &CameraFrustrum::get_t_cw() const { return t_cw_; }

const BoundingBox3D &CameraFrustrum::get_bbox() const {
    update_cache_();
    return bbox_;
}

const OrientedBoundingBox3D &CameraFrustrum::get_obb() const {
    update_cache_();
    return obb_;
}

const std::vector<Eigen::Vector3d> &CameraFrustrum::get_corners() const {
    update_cache_();
    return corners_;
}

template <typename T> bool CameraFrustrum::is_in_bbox(const T x_w, const T y_w, const T z_w) const {
    update_cache_();
    return bbox_.contains(x_w, y_w, z_w);
}

template <typename T> bool CameraFrustrum::is_in_bbox(const Eigen::Matrix<T, 3, 1> &point_w) const {
    update_cache_();
    return bbox_.contains(point_w);
}

template <typename T> bool CameraFrustrum::is_in_obb(const T x_w, const T y_w, const T z_w) const {
    update_cache_();
    return obb_.contains(x_w, y_w, z_w);
}

template <typename T> bool CameraFrustrum::is_in_obb(const Eigen::Matrix<T, 3, 1> &point_w) const {
    update_cache_();
    return obb_.contains(point_w);
}

template <typename T>
std::pair<bool, ImagePoint> CameraFrustrum::contains(const T x_w, const T y_w, const T z_w) const {
    update_cache_();
    return contains(Eigen::Matrix<T, 3, 1>(x_w, y_w, z_w));
}

template <typename T>
std::pair<bool, ImagePoint> CameraFrustrum::contains(const Eigen::Matrix<T, 3, 1> &point_w) const {
    // update_cache_(); // no need to update the cache here since we are only checking if the point
    // is in the frustum

    // Transform from world to camera coordinates
    const Eigen::Vector3d point_c = R_cw_ * point_w.template cast<double>() + t_cw_;

    const float depth = static_cast<float>(point_c.z());
    // Check depth range and ensure z > 0 to avoid division by zero
    const bool is_in_depth_range = depth >= depth_min_ && depth <= depth_max_;
    if (!is_in_depth_range) {
        return std::make_pair(false, ImagePoint{-1.0f, -1.0f, -1.0f});
    }

    // Project to pixel coordinates
    const float u = static_cast<float>(fx_ * (point_c.x() / point_c.z()) + cx_);
    const float v = static_cast<float>(fy_ * (point_c.y() / point_c.z()) + cy_);

    // Check if pixel coordinates are within image bounds
    const bool is_in_image = u >= 0.0f && u < width_ && v >= 0.0f && v < height_;
    return std::make_pair(is_in_image, ImagePoint{u, v, depth});
}

bool CameraFrustrum::is_cache_valid() const { return cache_is_valid_; }

void CameraFrustrum::update_cache_() const {
    if (!cache_is_valid_) {
        corners_ = compute_frustum_corners_world_();
        bbox_ = compute_bbox_();
        obb_ = compute_obb_();
        cache_is_valid_ = true;
    }
}

std::vector<Eigen::Vector3d> CameraFrustrum::compute_frustum_corners_world_() const {
    std::vector<Eigen::Vector3d> corners;

    // Define image corners in pixel coordinates
    const std::vector<Eigen::Vector2d> img_corners = {
        Eigen::Vector2d(0.0, 0.0),        // top-left
        Eigen::Vector2d(width_, 0.0),     // top-right
        Eigen::Vector2d(width_, height_), // bottom-right
        Eigen::Vector2d(0.0, height_)     // bottom-left
    };

    // Compute transformations
    const Eigen::Matrix3d R_wc = R_cw_.transpose(); // rotation from camera to world
    const Eigen::Vector3d t_wc = -R_wc * t_cw_;     // translation from camera to world

    // For each image corner, compute 3D points at near and far planes
    for (const auto &img_corner : img_corners) {
        // Convert pixel to normalized camera coordinates
        const double u = img_corner.x();
        const double v = img_corner.y();
        const double x_norm = (u - cx_) / fx_;
        const double y_norm = (v - cy_) / fy_;

        // Compute 3D points at near and far planes in camera frame
        const Eigen::Vector3d pt_near_cam(x_norm * depth_min_, y_norm * depth_min_, depth_min_);
        const Eigen::Vector3d pt_far_cam(x_norm * depth_max_, y_norm * depth_max_, depth_max_);

        // Transform to world frame
        const Eigen::Vector3d pt_near_world = R_wc * pt_near_cam + t_wc;
        const Eigen::Vector3d pt_far_world = R_wc * pt_far_cam + t_wc;

        corners.push_back(pt_near_world);
        corners.push_back(pt_far_world);
    }

    return corners;
}

BoundingBox3D CameraFrustrum::compute_bbox_() const {
    // Note: This method assumes corners_ is already computed (called from update_cache_)
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();
    for (const auto &corner : corners_) {
        min_x = std::min(min_x, corner.x());
        min_y = std::min(min_y, corner.y());
        min_z = std::min(min_z, corner.z());
        max_x = std::max(max_x, corner.x());
        max_y = std::max(max_y, corner.y());
        max_z = std::max(max_z, corner.z());
    }
    return BoundingBox3D(min_x, min_y, min_z, max_x, max_y, max_z);
}

OrientedBoundingBox3D CameraFrustrum::compute_obb_() const {
    // Note: This method assumes corners_ is already computed (called from update_cache_)
    // For OBB, we want the orientation of the camera frame in world coordinates
    const Eigen::Matrix3d R_wc = R_cw_.transpose();
    const Eigen::Quaterniond orientation_wc = Eigen::Quaterniond(R_wc);

    // Transform corners to camera-aligned frame to compute size and center
    double min_x_cam = std::numeric_limits<double>::max();
    double min_y_cam = std::numeric_limits<double>::max();
    double min_z_cam = std::numeric_limits<double>::max();
    double max_x_cam = std::numeric_limits<double>::lowest();
    double max_y_cam = std::numeric_limits<double>::lowest();
    double max_z_cam = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < corners_.size(); i++) {
        // Transform from world to camera coordinates
        Eigen::Vector3d corner_cam = R_cw_ * corners_[i] + t_cw_;
        min_x_cam = std::min(min_x_cam, corner_cam.x());
        min_y_cam = std::min(min_y_cam, corner_cam.y());
        min_z_cam = std::min(min_z_cam, corner_cam.z());
        max_x_cam = std::max(max_x_cam, corner_cam.x());
        max_y_cam = std::max(max_y_cam, corner_cam.y());
        max_z_cam = std::max(max_z_cam, corner_cam.z());
    }

    // Compute center in camera coordinates, then transform to world coordinates
    const Eigen::Vector3d center_cam((min_x_cam + max_x_cam) / 2.0, (min_y_cam + max_y_cam) / 2.0,
                                     (min_z_cam + max_z_cam) / 2.0);
    const Eigen::Vector3d t_wc = -R_wc * t_cw_; // translation from camera to world
    const Eigen::Vector3d center_w = R_wc * center_cam + t_wc;

    const Eigen::Vector3d size_cam(max_x_cam - min_x_cam, max_y_cam - min_y_cam,
                                   max_z_cam - min_z_cam);

    return OrientedBoundingBox3D(center_w, orientation_wc, size_cam);
}

// Explicit template instantiations for is_in_bbox
template bool CameraFrustrum::is_in_bbox<double>(const double x_w, const double y_w,
                                                 const double z_w) const;
template bool CameraFrustrum::is_in_bbox<float>(const float x_w, const float y_w,
                                                const float z_w) const;
template bool CameraFrustrum::is_in_bbox<double>(const Eigen::Matrix<double, 3, 1> &point_w) const;
template bool CameraFrustrum::is_in_bbox<float>(const Eigen::Matrix<float, 3, 1> &point_w) const;

// Explicit template instantiations for is_in_obb
template bool CameraFrustrum::is_in_obb<double>(const double x_w, const double y_w,
                                                const double z_w) const;
template bool CameraFrustrum::is_in_obb<float>(const float x_w, const float y_w,
                                               const float z_w) const;
template bool CameraFrustrum::is_in_obb<double>(const Eigen::Matrix<double, 3, 1> &point_w) const;
template bool CameraFrustrum::is_in_obb<float>(const Eigen::Matrix<float, 3, 1> &point_w) const;

// Explicit template instantiations for contains
template std::pair<bool, ImagePoint>
CameraFrustrum::contains<double>(const double x_w, const double y_w, const double z_w) const;
template std::pair<bool, ImagePoint>
CameraFrustrum::contains<float>(const float x_w, const float y_w, const float z_w) const;
template std::pair<bool, ImagePoint>
CameraFrustrum::contains<double>(const Eigen::Matrix<double, 3, 1> &point_w) const;
template std::pair<bool, ImagePoint>
CameraFrustrum::contains<float>(const Eigen::Matrix<float, 3, 1> &point_w) const;

} // namespace volumetric