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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>

namespace volumetric {

// ================================================
// Bounding boxes in 3D
// ================================================

// Axis-aligned bounding box in 3D
struct BoundingBox3D {
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;

    BoundingBox3D() : min_x(0.0), min_y(0.0), min_z(0.0), max_x(0.0), max_y(0.0), max_z(0.0) {}
    BoundingBox3D(double min_x, double min_y, double min_z, double max_x, double max_y,
                  double max_z)
        : min_x(min_x), min_y(min_y), min_z(min_z), max_x(max_x), max_y(max_y), max_z(max_z) {}
    BoundingBox3D(const Eigen::Vector3d &min_point, const Eigen::Vector3d &max_point)
        : min_x(min_point.x()), min_y(min_point.y()), min_z(min_point.z()), max_x(max_point.x()),
          max_y(max_point.y()), max_z(max_point.z()) {}

    Eigen::Vector3d get_min_point() const;
    Eigen::Vector3d get_max_point() const;

    Eigen::Vector3d get_center() const;
    Eigen::Vector3d get_size() const;

    double get_volume() const;

    double get_surface_area() const;

    double get_diagonal_length() const;

    template <typename T> bool contains(const T x_w, const T y_w, const T z_w) const;
    template <typename T> bool contains(const Eigen::Matrix<T, 3, 1> &point_w) const;

    template <typename T>
    std::vector<bool> contains(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w) const;

    bool intersects(const BoundingBox3D &other) const;
};

// Oriented bounding box (OBB) in 3D
struct OrientedBoundingBox3D {
    Eigen::Vector3d center; // Center (x, y, z) in world coordinates
    Eigen::Quaterniond
        orientation; // Rotation quaternion (w, x, y, z) from object-attached to world coordinates
    Eigen::Vector3d size; // Size (width, height, depth) in object-attached axes coordinates
                          // Width  = size.x(), Height = size.y(), Depth = size.z()

    OrientedBoundingBox3D()
        : center(0.0, 0.0, 0.0), orientation(1.0, 0.0, 0.0, 0.0), size(0.0, 0.0, 0.0) {}

    OrientedBoundingBox3D(const Eigen::Vector3d &center, const Eigen::Quaterniond &orientation,
                          const Eigen::Vector3d &size)
        : center(center), orientation(orientation), size(size) {}

    double get_volume() const;

    double get_surface_area() const;
    double get_diagonal_length() const;

    // get the corners of the OBB in world coordinates
    std::vector<Eigen::Vector3d> get_corners() const;

    template <typename T> bool contains(const T x, const T y, const T z) const;
    template <typename T> bool contains(const Eigen::Matrix<T, 3, 1> &point_w) const;

    template <typename T>
    std::vector<bool> contains(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w) const;

    bool intersects(const OrientedBoundingBox3D &other) const;

    bool intersects(const BoundingBox3D &other) const;
};

// ================================================
// Bounding boxes in 2D
// ================================================

// Axis-aligned bounding box in 2D
struct BoundingBox2D {
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
};

// Oriented bounding box (OBB) in 2D
struct OrientedBoundingBox2D {
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
};

} // namespace volumetric