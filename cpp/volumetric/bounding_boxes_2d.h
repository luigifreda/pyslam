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

#include "bounding_boxes_3d.h" // For OBBComputationMethod
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <vector>

namespace volumetric {

// ================================================
// Bounding boxes in 2D
// ================================================

// Axis-aligned bounding box in 2D
struct BoundingBox2D {

    using Ptr = std::shared_ptr<BoundingBox2D>;

    double min_x;
    double min_y;
    double max_x;
    double max_y;

    BoundingBox2D() : min_x(0.0), min_y(0.0), max_x(0.0), max_y(0.0) {}
    BoundingBox2D(double min_x, double min_y, double max_x, double max_y)
        : min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {}
    BoundingBox2D(const Eigen::Vector2d &min_point, const Eigen::Vector2d &max_point)
        : min_x(min_point.x()), min_y(min_point.y()), max_x(max_point.x()), max_y(max_point.y()) {}

    Eigen::Vector2d get_center() const;
    Eigen::Vector2d get_min_point() const;
    Eigen::Vector2d get_max_point() const;

    Eigen::Vector2d get_size() const;

    double get_area() const;
    double get_perimeter() const;
    double get_diagonal_length() const;

    template <typename T> bool contains(const T x_w, const T y_w) const;
    template <typename T> bool contains(const Eigen::Matrix<T, 2, 1> &point_w) const;
    template <typename T>
    std::vector<bool> contains(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) const;

    bool intersects(const BoundingBox2D &other) const;

    template <typename T>
    static BoundingBox2D compute_from_points(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w);
};

// Oriented bounding box (OBB) in 2D
struct OrientedBoundingBox2D {

    using Ptr = std::shared_ptr<OrientedBoundingBox2D>;

    Eigen::Vector2d center;
    double angle_rad;     // rotation about z in radians
    Eigen::Vector2d size; // width and height in object-attached axes coordinates

    OrientedBoundingBox2D() : center(0.0, 0.0), angle_rad(0.0), size(0.0, 0.0) {}
    OrientedBoundingBox2D(const Eigen::Vector2d &center, const double angle_rad,
                          const Eigen::Vector2d &size)
        : center(center), angle_rad(angle_rad), size(size) {}

    // Note: get_volume() doesn't make sense for 2D, but kept for API consistency
    // Returns area instead (volume would be 0 in 2D)
    double get_volume() const;
    double get_area() const;
    double get_perimeter() const;
    double get_diagonal_length() const;

    // get the corners of the OBB in world coordinates
    std::vector<Eigen::Vector2d> get_corners() const;

    template <typename T> bool contains(const T x_w, const T y_w) const;
    template <typename T> bool contains(const Eigen::Matrix<T, 2, 1> &point_w) const;
    template <typename T>
    std::vector<bool> contains(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) const;

    bool intersects(const OrientedBoundingBox2D &other) const;

    bool intersects(const BoundingBox2D &other) const;

    template <typename T>
    static OrientedBoundingBox2D
    compute_from_points(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w,
                        const OBBComputationMethod &method = OBBComputationMethod::PCA);
};

// ================================================
// Internal helper structures for 2D OBB
// ================================================

namespace detail_2d {

// OBB frame in 2D. This is a minimal and simple way to represent the OBB in a way that is easy to
// use for the SAT test and drawing.
struct OBBFrame2D {
    Eigen::Vector2d center;
    Eigen::Matrix2d axes;
    Eigen::Vector2d half;
};

inline OBBFrame2D make_frame(const OrientedBoundingBox2D &obb) {
    OBBFrame2D f;
    const Eigen::Rotation2Dd R(obb.angle_rad);
    f.axes = R.toRotationMatrix();
    f.center = obb.center;
    f.half = obb.size / 2.0;
    return f;
}

inline OBBFrame2D make_frame(const BoundingBox2D &aabb) {
    OBBFrame2D f;
    f.axes.setIdentity();
    f.center = Eigen::Vector2d((aabb.min_x + aabb.max_x) * 0.5, (aabb.min_y + aabb.max_y) * 0.5);
    f.half = Eigen::Vector2d((aabb.max_x - aabb.min_x) * 0.5, (aabb.max_y - aabb.min_y) * 0.5);
    return f;
}

} // namespace detail_2d

} // namespace volumetric
