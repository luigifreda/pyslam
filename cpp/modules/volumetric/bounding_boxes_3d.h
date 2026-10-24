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
#include <memory>
#include <vector>

namespace volumetric {

enum class OBBComputationMethod {
    PCA,                ///< PCA on all input points
    CONVEX_HULL_MINIMAL ///< Scan convex-hull triangles, pick box with smallest volume
};

// ================================================
// Bounding boxes in 3D
// ================================================

// Axis-aligned bounding box in 3D
struct BoundingBox3D {

    using Ptr = std::shared_ptr<BoundingBox3D>;

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

    template <typename T>
    static BoundingBox3D compute_from_points(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w);
};

// Oriented bounding box (OBB) in 3D
struct OrientedBoundingBox3D {

    using Ptr = std::shared_ptr<OrientedBoundingBox3D>;

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

    Eigen::Matrix4d
    get_matrix() const; // Transformation matrix from object-attached to world coordinates
    Eigen::Matrix4d
    get_inverse_matrix() const; // Transformation matrix from world to object-attached coordinates

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

    template <typename T>
    static OrientedBoundingBox3D
    compute_from_points(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w,
                        const OBBComputationMethod &method = OBBComputationMethod::PCA);

    template <typename T>
    static OrientedBoundingBox3D
    compute_from_points(const std::vector<std::array<T, 3>> &points_w,
                        const OBBComputationMethod &method = OBBComputationMethod::PCA);
};

// ================================================
// Internal helper structures for 3D OBB
// ================================================

namespace detail_3d {

// OBB frame in 3D. This is a minimal and simple way to represent the OBB in a way that is easy to
// use for the SAT test and drawing.
struct OBBFrame3D {
    Eigen::Vector3d center;
    Eigen::Vector3d half;
    Eigen::Matrix3d axes; // columns are unit axes (from object-attached to world coordinates)
};

inline OBBFrame3D make_frame(const OrientedBoundingBox3D &obb) {
    OBBFrame3D f;
    const Eigen::Quaterniond q_norm = obb.orientation.normalized();
    f.axes = q_norm.normalized().toRotationMatrix();
    f.center = obb.center;
    f.half = obb.size / 2.0;
    return f;
}

inline OBBFrame3D make_frame(const BoundingBox3D &aabb) {
    OBBFrame3D f;
    f.axes.setIdentity();
    f.center = Eigen::Vector3d((aabb.min_x + aabb.max_x) * 0.5, (aabb.min_y + aabb.max_y) * 0.5,
                               (aabb.min_z + aabb.max_z) * 0.5);
    f.half = Eigen::Vector3d((aabb.max_x - aabb.min_x) * 0.5, (aabb.max_y - aabb.min_y) * 0.5,
                             (aabb.max_z - aabb.min_z) * 0.5);
    return f;
}

} // namespace detail_3d

} // namespace volumetric
