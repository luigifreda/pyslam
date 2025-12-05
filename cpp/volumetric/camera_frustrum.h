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

class CameraFrustrum {
  public:
    CameraFrustrum(const Eigen::Matrix3d &K, const int width, const int height,
                   const Eigen::Matrix4d &T_cw, const double depth_max = 10.0,
                   const double depth_min = 0.1)
        : width_(width), height_(height), fx_(K(0, 0)), fy_(K(1, 1)), cx_(K(0, 2)), cy_(K(1, 2)),
          R_cw_(T_cw.block<3, 3>(0, 0)), t_cw_(T_cw.block<3, 1>(0, 3)), depth_max_(depth_max),
          depth_min_(depth_min) {
        update_cache_();
    }

    CameraFrustrum(const float fx, const float fy, const float cx, const float cy, const int width,
                   const int height, const Eigen::Quaterniond &orientation,
                   const Eigen::Vector3d &translation, const double depth_max = 10.0,
                   const double depth_min = 0.1)
        : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          R_cw_(orientation.normalized().toRotationMatrix()), t_cw_(translation),
          depth_max_(depth_max), depth_min_(depth_min) {
        update_cache_();
    }
    ~CameraFrustrum() = default;

    void set_width(const int width) {
        width_ = width;
        cache_is_valid_ = false;
    }
    void set_height(const int height) {
        height_ = height;
        cache_is_valid_ = false;
    }

    void set_depth_max(const double depth_max) {
        depth_max_ = depth_max;
        cache_is_valid_ = false;
    }
    void set_depth_min(const double depth_min) {
        depth_min_ = depth_min;
        cache_is_valid_ = false;
    }

    void set_intrinsics(const Eigen::Matrix3d &K) {
        fx_ = K(0, 0);
        fy_ = K(1, 1);
        cx_ = K(0, 2);
        cy_ = K(1, 2);
        cache_is_valid_ = false;
    }
    void set_intrinsics(const float fx, const float fy, const float cx, const float cy) {
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        cache_is_valid_ = false;
    }

    void set_T_cw(const Eigen::Matrix4d &T_cw) {
        R_cw_ = T_cw.block<3, 3>(0, 0);
        t_cw_ = T_cw.block<3, 1>(0, 3);
        cache_is_valid_ = false;
    }
    void set_T_cw(const Eigen::Quaterniond &orientation, const Eigen::Vector3d &translation) {
        R_cw_ = orientation.normalized().toRotationMatrix();
        t_cw_ = translation;
        cache_is_valid_ = false;
    }

    Eigen::Matrix3d get_K() const {
        Eigen::Matrix3d K;
        K << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
        return K;
    }
    Eigen::Matrix4d get_T_cw() const {
        Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
        T_cw.block<3, 3>(0, 0) = R_cw_;
        T_cw.block<3, 1>(0, 3) = t_cw_;
        return T_cw;
    }
    const Eigen::Matrix3d &get_R_cw() const { return R_cw_; }
    const Eigen::Quaterniond get_orientation_cw() const { return Eigen::Quaterniond(R_cw_); }
    const Eigen::Vector3d &get_t_cw() const { return t_cw_; }

    const OrientedBoundingBox3D &get_obb() const {
        update_cache_();
        return obb_;
    }

    // get the corners of the camera frustrum in world coordinates
    const std::vector<Eigen::Vector3d> &get_corners() const {
        update_cache_();
        return corners_;
    }

    bool is_in_obb(const Eigen::Vector3d &point_w) const {
        update_cache_();
        return obb_.contains(point_w);
    }
    bool is_in_frustum(const Eigen::Vector3d &point_w) const {
        // Transform from world to camera coordinates
        const Eigen::Vector3d point_c = R_cw_ * point_w + t_cw_;

        // Check depth range
        const bool is_in_depth_range = point_c.z() >= depth_min_ && point_c.z() <= depth_max_;
        if (!is_in_depth_range) {
            return false;
        }

        // Project to pixel coordinates
        const double u = fx_ * (point_c.x() / point_c.z()) + cx_;
        const double v = fy_ * (point_c.y() / point_c.z()) + cy_;

        // Check if pixel coordinates are within image bounds
        const bool is_in_image = u >= 0.0 && u < width_ && v >= 0.0 && v < height_;
        return is_in_image;
    }

    bool is_cache_valid() const { return cache_is_valid_; }

  private:
    // Helper function to compute frustum corners in world space
    std::vector<Eigen::Vector3d> compute_frustum_corners_world_() const {
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

    // compute the OBB of the camera frustrum considering the current pose and depth range
    OrientedBoundingBox3D compute_obb_() const {

        // Compute center as average of all corners in world coordinates
        Eigen::Vector3d center_w = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < corners_.size(); i++) {
            center_w += corners_[i];
        }
        center_w /= static_cast<double>(corners_.size());

        // For OBB, we want the orientation of the camera frame in world coordinates
        const Eigen::Matrix3d R_wc = R_cw_.transpose();
        const Eigen::Quaterniond orientation_wc = Eigen::Quaterniond(R_wc);

        // Transform corners to camera-aligned frame to compute size
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

        const Eigen::Vector3d size_cam(max_x_cam - min_x_cam, max_y_cam - min_y_cam,
                                       max_z_cam - min_z_cam);

        return OrientedBoundingBox3D(center_w, orientation_wc, size_cam);
    }

  protected:
    int width_;
    int height_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    Eigen::Matrix3d R_cw_; // rotation matrix from world to camera coordinates
    Eigen::Vector3d t_cw_; // translation vector from world to camera coordinates
                           // For 3D points: p_c = R_cw_ * p_w + t_cw_
    double depth_max_;     // maximum depth for the camera frustrum
    double depth_min_;     // minimum depth for the camera frustrum

    mutable bool cache_is_valid_ = false;

    mutable std::vector<Eigen::Vector3d>
        corners_;                       // corners of the camera frustrum in world coordinates
    mutable OrientedBoundingBox3D obb_; // oriented bounding box of the camera frustrum

    void update_cache_() const {
        if (!cache_is_valid_) {
            corners_ = compute_frustum_corners_world_();
            obb_ = compute_obb_();
            cache_is_valid_ = true;
        }
    }
};