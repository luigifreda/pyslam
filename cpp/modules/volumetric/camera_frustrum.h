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
#pragma once

#include "bounding_boxes.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <vector>

namespace volumetric {

struct ImagePoint {
    float u = -1.0f;
    float v = -1.0f;
    float depth = -1.0f;
};

class CameraFrustrum {
  public:
    CameraFrustrum(const Eigen::Matrix3d &K, const int width, const int height,
                   const Eigen::Matrix4d &T_cw, const float depth_max = 10.0f,
                   const float depth_min = 1e-2f);
    CameraFrustrum(const float fx, const float fy, const float cx, const float cy, const int width,
                   const int height, const Eigen::Quaterniond &orientation,
                   const Eigen::Vector3d &translation, const float depth_max = 10.0f,
                   const float depth_min = 1e-2f);
    CameraFrustrum(const float fx, const float fy, const float cx, const float cy, const int width,
                   const int height, const Eigen::Matrix4d &T_cw, const float depth_max = 10.0f,
                   const float depth_min = 1e-2f);
    ~CameraFrustrum() = default;

    void set_width(const int width);
    void set_height(const int height);

    void set_depth_max(const float depth_max);
    void set_depth_min(const float depth_min);

    void set_intrinsics(const Eigen::Matrix3d &K);
    void set_intrinsics(const float fx, const float fy, const float cx, const float cy);

    void set_T_cw(const Eigen::Matrix4d &T_cw);
    void set_T_cw(const Eigen::Quaterniond &orientation, const Eigen::Vector3d &translation);

    int get_width() const;
    int get_height() const;
    float get_fx() const;
    float get_fy() const;
    float get_cx() const;
    float get_cy() const;
    Eigen::Matrix3d get_K() const;
    Eigen::Matrix4d get_T_cw() const;
    const Eigen::Matrix3d &get_R_cw() const;
    const Eigen::Quaterniond get_orientation_cw() const;
    const Eigen::Vector3d &get_t_cw() const;

    const BoundingBox3D &get_bbox() const;
    const OrientedBoundingBox3D &get_obb() const;

    // get the corners of the camera frustrum in world coordinates
    const std::vector<Eigen::Vector3d> &get_corners() const;

    template <typename T> bool is_in_bbox(const T x_w, const T y_w, const T z_w) const;
    template <typename T> bool is_in_bbox(const Eigen::Matrix<T, 3, 1> &point_w) const;
    template <typename T> bool is_in_obb(const T x_w, const T y_w, const T z_w) const;
    template <typename T> bool is_in_obb(const Eigen::Matrix<T, 3, 1> &point_w) const;

    // check if the point is in the camera frustrum
    // returns a pair of bool and the depth of the point
    // the bool indicates if the point is in the camera frustrum
    // the depth is the depth of the point in the camera frame
    template <typename T>
    std::pair<bool, ImagePoint> contains(const T x_w, const T y_w, const T z_w) const;
    template <typename T>
    std::pair<bool, ImagePoint> contains(const Eigen::Matrix<T, 3, 1> &point_w) const;

    bool is_cache_valid() const;

  private:
    // Helper function to compute frustum corners in world space
    std::vector<Eigen::Vector3d> compute_frustum_corners_world_() const;

    // compute the AABB of the camera frustrum considering the current pose and depth range
    BoundingBox3D compute_bbox_() const;

    // compute the OBB of the camera frustrum considering the current pose and depth range
    OrientedBoundingBox3D compute_obb_() const;

  protected:
    int width_;
    int height_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;

    Eigen::Matrix3d R_cw_; // rotation matrix from world to camera coordinates
    Eigen::Vector3d t_cw_; // translation vector from world to camera coordinates
                           // For 3D points: p_c = R_cw_ * p_w + t_cw_
    float depth_max_;      // maximum depth for the camera frustrum
    float depth_min_;      // minimum depth for the camera frustrum

    mutable bool cache_is_valid_ = false;

    mutable std::vector<Eigen::Vector3d>
        corners_; // corners of the camera frustrum in world coordinates
    mutable BoundingBox3D
        bbox_; // axis-aligned bounding box of the camera frustrum in world coordinates
    mutable OrientedBoundingBox3D obb_; // oriented bounding box of the camera frustrum

    void update_cache_() const;
};

} // namespace volumetric
