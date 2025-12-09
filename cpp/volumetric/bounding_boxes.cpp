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

#include "bounding_boxes.h"

#include <cmath>

namespace volumetric {

// ================================================
// BoundingBox3D implementations
// ================================================

Eigen::Vector3d BoundingBox3D::get_min_point() const {
    return Eigen::Vector3d(min_x, min_y, min_z);
}

Eigen::Vector3d BoundingBox3D::get_max_point() const {
    return Eigen::Vector3d(max_x, max_y, max_z);
}

Eigen::Vector3d BoundingBox3D::get_center() const {
    return Eigen::Vector3d((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0);
}

Eigen::Vector3d BoundingBox3D::get_size() const {
    return Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z);
}

double BoundingBox3D::get_volume() const {
    return (max_x - min_x) * (max_y - min_y) * (max_z - min_z);
}

double BoundingBox3D::get_surface_area() const {
    return 2.0 * ((max_x - min_x) * (max_y - min_y) + (max_x - min_x) * (max_z - min_z) +
                  (max_y - min_y) * (max_z - min_z));
}

double BoundingBox3D::get_diagonal_length() const {
    return std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y) +
                     (max_z - min_z) * (max_z - min_z));
}

template <typename T> bool BoundingBox3D::contains(const T x_w, const T y_w, const T z_w) const {
    return x_w >= min_x && x_w <= max_x && y_w >= min_y && y_w <= max_y && z_w >= min_z &&
           z_w <= max_z;
}

template <typename T> bool BoundingBox3D::contains(const Eigen::Matrix<T, 3, 1> &point_w) const {
    return contains(point_w.x(), point_w.y(), point_w.z());
}

template <typename T>
std::vector<bool>
BoundingBox3D::contains(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w) const {
    std::vector<bool> contains_mask(points_w.size(), false);
    for (size_t i = 0; i < points_w.size(); i++) {
        contains_mask[i] = contains(points_w[i]);
    }
    return contains_mask;
}

bool BoundingBox3D::intersects(const BoundingBox3D &other) const {
    return (min_x <= other.max_x && max_x >= other.min_x) &&
           (min_y <= other.max_y && max_y >= other.min_y) &&
           (min_z <= other.max_z && max_z >= other.min_z);
}

// ================================================
// OrientedBoundingBox3D implementations
// ================================================

double OrientedBoundingBox3D::get_volume() const { return size.x() * size.y() * size.z(); }

double OrientedBoundingBox3D::get_surface_area() const {
    return 2.0 * ((size.x() * size.y()) + (size.x() * size.z()) + (size.y() * size.z()));
}

double OrientedBoundingBox3D::get_diagonal_length() const {
    return std::sqrt((size.x() * size.x()) + (size.y() * size.y()) + (size.z() * size.z()));
}

std::vector<Eigen::Vector3d> OrientedBoundingBox3D::get_corners() const {
    std::vector<Eigen::Vector3d> corners;
    corners.reserve(8);
    const Eigen::Quaterniond q_norm = orientation.normalized();
    const Eigen::Matrix3d R_wo =
        q_norm.toRotationMatrix(); // from object-attached axes to world coordinates,
                                   // p_w = R_wo * p_o + center
    const Eigen::Vector3d half_size = size / 2.0;

    // Assuming object-attached axes coordinates are x=right, y=down, z=forward
    // Near plane corners (z = -1) - object-attached axes coordinates
    corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                          Eigen::Vector3d(1.0, 1.0, -1.0)))); // front-left-near
    corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                          Eigen::Vector3d(-1.0, 1.0, -1.0)))); // front-right-near
    corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                          Eigen::Vector3d(-1.0, -1.0, -1.0)))); // back-right-near
    corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                          Eigen::Vector3d(1.0, -1.0, -1.0)))); // back-left-near

    // Far plane corners (z = +1)
    corners.push_back(
        center + R_wo * (half_size.cwiseProduct(Eigen::Vector3d(1.0, 1.0, 1.0)))); // front-left-far
    corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                          Eigen::Vector3d(-1.0, 1.0, 1.0)))); // front-right-far
    corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                          Eigen::Vector3d(-1.0, -1.0, 1.0)))); // back-right-far
    corners.push_back(
        center + R_wo * (half_size.cwiseProduct(Eigen::Vector3d(1.0, -1.0, 1.0)))); // back-left-far

    return corners;
}

template <typename T>
bool OrientedBoundingBox3D::contains(const T x_w, const T y_w, const T z_w) const {
    return contains(Eigen::Matrix<T, 3, 1>(x_w, y_w, z_w));
}

template <typename T>
bool OrientedBoundingBox3D::contains(const Eigen::Matrix<T, 3, 1> &point_w) const {
    const Eigen::Quaterniond q_norm = orientation.normalized();
    const Eigen::Vector3d point_o = q_norm.inverse() * (point_w.template cast<double>() - center);
    const Eigen::Vector3d half_size = size / 2.0;
    return point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
           point_o.y() >= -half_size.y() && point_o.y() <= half_size.y() &&
           point_o.z() >= -half_size.z() && point_o.z() <= half_size.z();
}

template <typename T>
std::vector<bool>
OrientedBoundingBox3D::contains(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w) const {
    std::vector<bool> contains_mask(points_w.size(), false);
    const Eigen::Quaterniond q_norm = orientation.normalized();
    const Eigen::Vector3d half_size = size / 2.0;
    for (size_t i = 0; i < points_w.size(); i++) {
        const Eigen::Vector3d point_o =
            q_norm.inverse() * (points_w[i].template cast<double>() - center);
        contains_mask[i] = point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
                           point_o.y() >= -half_size.y() && point_o.y() <= half_size.y() &&
                           point_o.z() >= -half_size.z() && point_o.z() <= half_size.z();
    }
    return contains_mask;
}

bool OrientedBoundingBox3D::intersects(const OrientedBoundingBox3D &other) const {
    // Check if any corner of this box is contained in the other box
    const std::vector<Eigen::Vector3d> corners = get_corners();
    for (const auto &corner : corners) {
        if (other.contains(corner)) {
            return true;
        }
    }
    // Check if any corner of the other box is contained in this box
    // Not redundant (e.g. other box is completely inside this box)
    const std::vector<Eigen::Vector3d> other_corners = other.get_corners();
    for (const auto &corner : other_corners) {
        if (contains(corner)) {
            return true;
        }
    }
    return false;
}

bool OrientedBoundingBox3D::intersects(const BoundingBox3D &other) const {
    const std::vector<Eigen::Vector3d> corners = get_corners();
    for (const auto &corner : corners) {
        if (other.contains(corner)) {
            return true;
        }
    }
    return false;
}

// ================================================
// BoundingBox2D implementations
// ================================================

Eigen::Vector2d BoundingBox2D::get_center() const {
    return Eigen::Vector2d((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
}

Eigen::Vector2d BoundingBox2D::get_min_point() const { return Eigen::Vector2d(min_x, min_y); }

Eigen::Vector2d BoundingBox2D::get_max_point() const { return Eigen::Vector2d(max_x, max_y); }

Eigen::Vector2d BoundingBox2D::get_size() const {
    return Eigen::Vector2d(max_x - min_x, max_y - min_y);
}

double BoundingBox2D::get_area() const { return (max_x - min_x) * (max_y - min_y); }

double BoundingBox2D::get_perimeter() const { return 2.0 * ((max_x - min_x) + (max_y - min_y)); }

double BoundingBox2D::get_diagonal_length() const {
    return std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y));
}

template <typename T> bool BoundingBox2D::contains(const T x_w, const T y_w) const {
    return x_w >= min_x && x_w <= max_x && y_w >= min_y && y_w <= max_y;
}

template <typename T> bool BoundingBox2D::contains(const Eigen::Matrix<T, 2, 1> &point_w) const {
    return contains(point_w.x(), point_w.y());
}

template <typename T>
std::vector<bool>
BoundingBox2D::contains(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) const {
    std::vector<bool> contains_mask(points_w.size(), false);
    for (size_t i = 0; i < points_w.size(); i++) {
        contains_mask[i] = contains(points_w[i]);
    }
    return contains_mask;
}

bool BoundingBox2D::intersects(const BoundingBox2D &other) const {
    return (min_x <= other.max_x && max_x >= other.min_x) &&
           (min_y <= other.max_y && max_y >= other.min_y);
}

// ================================================
// OrientedBoundingBox2D implementations
// ================================================

double OrientedBoundingBox2D::get_volume() const { return size.x() * size.y(); }

double OrientedBoundingBox2D::get_area() const { return size.x() * size.y(); }

double OrientedBoundingBox2D::get_perimeter() const { return 2.0 * ((size.x() + size.y())); }

double OrientedBoundingBox2D::get_diagonal_length() const {
    return std::sqrt((size.x() * size.x()) + (size.y() * size.y()));
}

std::vector<Eigen::Vector2d> OrientedBoundingBox2D::get_corners() const {
    std::vector<Eigen::Vector2d> corners;
    corners.reserve(4);
    const Eigen::Rotation2Dd R_wo(angle_rad); // from object-attached axes to world coordinates,
                                              // p_w = R_wo * p_o + center
    const Eigen::Vector2d half_size = size / 2.0;
    corners.push_back(center + R_wo * (half_size.cwiseProduct(Eigen::Vector2d(1.0, 1.0))));
    corners.push_back(center + R_wo * (half_size.cwiseProduct(Eigen::Vector2d(-1.0, 1.0))));
    corners.push_back(center + R_wo * (half_size.cwiseProduct(Eigen::Vector2d(-1.0, -1.0))));
    corners.push_back(center + R_wo * (half_size.cwiseProduct(Eigen::Vector2d(1.0, -1.0))));
    return corners;
}

template <typename T> bool OrientedBoundingBox2D::contains(const T x_w, const T y_w) const {
    return contains(Eigen::Matrix<T, 2, 1>(x_w, y_w));
}

template <typename T>
bool OrientedBoundingBox2D::contains(const Eigen::Matrix<T, 2, 1> &point_w) const {
    const Eigen::Rotation2Dd R_wo(angle_rad);
    const Eigen::Vector2d point_o = R_wo.inverse() * (point_w.template cast<double>() - center);
    const Eigen::Vector2d half_size = size / 2.0;
    return point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
           point_o.y() >= -half_size.y() && point_o.y() <= half_size.y();
}

template <typename T>
std::vector<bool>
OrientedBoundingBox2D::contains(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) const {
    std::vector<bool> contains_mask(points_w.size(), false);
    const Eigen::Rotation2Dd R_wo(angle_rad);
    const Eigen::Vector2d half_size = size / 2.0;
    for (size_t i = 0; i < points_w.size(); i++) {
        const Eigen::Vector2d point_o =
            R_wo.inverse() * (points_w[i].template cast<double>() - center);
        contains_mask[i] = point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
                           point_o.y() >= -half_size.y() && point_o.y() <= half_size.y();
    }
    return contains_mask;
}

bool OrientedBoundingBox2D::intersects(const OrientedBoundingBox2D &other) const {
    // Check if any corner of this box is contained in the other box
    const std::vector<Eigen::Vector2d> corners = get_corners();
    for (const auto &corner : corners) {
        if (other.contains(corner)) {
            return true;
        }
    }
    // Check if any corner of the other box is contained in this box
    // Not redundant (e.g. other box is completely inside this box)
    const std::vector<Eigen::Vector2d> other_corners = other.get_corners();
    for (const auto &corner : other_corners) {
        if (contains(corner)) {
            return true;
        }
    }
    return false;
}

bool OrientedBoundingBox2D::intersects(const BoundingBox2D &other) const {
    // Check if any corner of this OBB is contained in the AABB
    const std::vector<Eigen::Vector2d> corners = get_corners();
    for (const auto &corner : corners) {
        if (other.contains(corner)) {
            return true;
        }
    }
    // Check if any corner of the AABB is contained in this OBB
    const Eigen::Vector2d aabb_corners[4] = {
        Eigen::Vector2d(other.min_x, other.min_y), // bottom-left
        Eigen::Vector2d(other.max_x, other.min_y), // bottom-right
        Eigen::Vector2d(other.max_x, other.max_y), // top-right
        Eigen::Vector2d(other.min_x, other.max_y)  // top-left
    };
    for (const auto &corner : aabb_corners) {
        if (contains(corner)) {
            return true;
        }
    }
    return false;
}

// Explicit template instantiations
template bool BoundingBox3D::contains<double>(const double x, const double y, const double z) const;
template bool BoundingBox3D::contains<float>(const float x, const float y, const float z) const;
template bool BoundingBox3D::contains<double>(const Eigen::Matrix<double, 3, 1> &point_w) const;
template bool BoundingBox3D::contains<float>(const Eigen::Matrix<float, 3, 1> &point_w) const;
template std::vector<bool>
BoundingBox3D::contains<double>(const std::vector<Eigen::Matrix<double, 3, 1>> &points_w) const;
template std::vector<bool>
BoundingBox3D::contains<float>(const std::vector<Eigen::Matrix<float, 3, 1>> &points_w) const;

template bool OrientedBoundingBox3D::contains<double>(const double x_w, const double y_w,
                                                      const double z_w) const;
template bool OrientedBoundingBox3D::contains<float>(const float x_w, const float y_w,
                                                     const float z_w) const;
template bool
OrientedBoundingBox3D::contains<double>(const Eigen::Matrix<double, 3, 1> &point_w) const;
template bool
OrientedBoundingBox3D::contains<float>(const Eigen::Matrix<float, 3, 1> &point_w) const;
template std::vector<bool> OrientedBoundingBox3D::contains<double>(
    const std::vector<Eigen::Matrix<double, 3, 1>> &points_w) const;
template std::vector<bool> OrientedBoundingBox3D::contains<float>(
    const std::vector<Eigen::Matrix<float, 3, 1>> &points_w) const;

template bool BoundingBox2D::contains<double>(const Eigen::Matrix<double, 2, 1> &point_w) const;
template bool BoundingBox2D::contains<float>(const Eigen::Matrix<float, 2, 1> &point_w) const;
template std::vector<bool>
BoundingBox2D::contains<double>(const std::vector<Eigen::Matrix<double, 2, 1>> &points_w) const;
template std::vector<bool>
BoundingBox2D::contains<float>(const std::vector<Eigen::Matrix<float, 2, 1>> &points_w) const;

template bool
OrientedBoundingBox2D::contains<double>(const Eigen::Matrix<double, 2, 1> &point_w) const;
template bool
OrientedBoundingBox2D::contains<float>(const Eigen::Matrix<float, 2, 1> &point_w) const;
template std::vector<bool> OrientedBoundingBox2D::contains<double>(
    const std::vector<Eigen::Matrix<double, 2, 1>> &points_w) const;
template std::vector<bool> OrientedBoundingBox2D::contains<float>(
    const std::vector<Eigen::Matrix<float, 2, 1>> &points_w) const;

} // namespace volumetric
