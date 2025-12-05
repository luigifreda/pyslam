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

    Eigen::Vector3d get_min_point() const { return Eigen::Vector3d(min_x, min_y, min_z); }
    Eigen::Vector3d get_max_point() const { return Eigen::Vector3d(max_x, max_y, max_z); }

    Eigen::Vector3d get_center() const {
        return Eigen::Vector3d((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0);
    }
    Eigen::Vector3d get_size() const {
        return Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z);
    }

    double get_volume() const { return (max_x - min_x) * (max_y - min_y) * (max_z - min_z); }

    double get_surface_area() const {
        return 2.0 * ((max_x - min_x) * (max_y - min_y) + (max_x - min_x) * (max_z - min_z) +
                      (max_y - min_y) * (max_z - min_z));
    }

    double get_diagonal_length() const {
        return std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y) +
                         (max_z - min_z) * (max_z - min_z));
    }

    bool contains(const Eigen::Vector3d &point_w) const {
        return point_w.x() >= min_x && point_w.x() <= max_x && point_w.y() >= min_y &&
               point_w.y() <= max_y && point_w.z() >= min_z && point_w.z() <= max_z;
    }

    std::vector<bool> contains(const std::vector<Eigen::Vector3d> &points_w) const {
        std::vector<bool> contains_mask(points_w.size(), false);
        for (size_t i = 0; i < points_w.size(); i++) {
            contains_mask[i] = contains(points_w[i]);
        }
        return contains_mask;
    }

    bool intersects(const BoundingBox3D &other) const {
        return (min_x <= other.max_x && max_x >= other.min_x) &&
               (min_y <= other.max_y && max_y >= other.min_y) &&
               (min_z <= other.max_z && max_z >= other.min_z);
    }
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

    double get_volume() const { return size.x() * size.y() * size.z(); }

    double get_surface_area() const {
        return 2.0 * ((size.x() * size.y()) + (size.x() * size.z()) + (size.y() * size.z()));
    }
    double get_diagonal_length() const {
        return std::sqrt((size.x() * size.x()) + (size.y() * size.y()) + (size.z() * size.z()));
    }

    // get the corners of the OBB in world coordinates
    std::vector<Eigen::Vector3d> get_corners() const {
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
        corners.push_back(center + R_wo * (half_size.cwiseProduct(Eigen::Vector3d(
                                              -1.0, 1.0, -1.0)))); // front-right-near
        corners.push_back(center + R_wo * (half_size.cwiseProduct(Eigen::Vector3d(
                                              -1.0, -1.0, -1.0)))); // back-right-near
        corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                              Eigen::Vector3d(1.0, -1.0, -1.0)))); // back-left-near

        // Far plane corners (z = +1)
        corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                              Eigen::Vector3d(1.0, 1.0, 1.0)))); // front-left-far
        corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                              Eigen::Vector3d(-1.0, 1.0, 1.0)))); // front-right-far
        corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                              Eigen::Vector3d(-1.0, -1.0, 1.0)))); // back-right-far
        corners.push_back(center + R_wo * (half_size.cwiseProduct(
                                              Eigen::Vector3d(1.0, -1.0, 1.0)))); // back-left-far

        return corners;
    }

    bool contains(const Eigen::Vector3d &point_w) const {
        const Eigen::Quaterniond q_norm = orientation.normalized();
        const Eigen::Vector3d point_o = q_norm.inverse() * (point_w - center);
        const Eigen::Vector3d half_size = size / 2.0;
        return point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
               point_o.y() >= -half_size.y() && point_o.y() <= half_size.y() &&
               point_o.z() >= -half_size.z() && point_o.z() <= half_size.z();
    }

    std::vector<bool> contains(const std::vector<Eigen::Vector3d> &points_w) const {
        std::vector<bool> contains_mask(points_w.size(), false);
        const Eigen::Quaterniond q_norm = orientation.normalized();
        const Eigen::Vector3d half_size = size / 2.0;
        for (size_t i = 0; i < points_w.size(); i++) {
            const Eigen::Vector3d point_o = q_norm.inverse() * (points_w[i] - center);
            contains_mask[i] = point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
                               point_o.y() >= -half_size.y() && point_o.y() <= half_size.y() &&
                               point_o.z() >= -half_size.z() && point_o.z() <= half_size.z();
        }
        return contains_mask;
    }

    bool intersects(const OrientedBoundingBox3D &other) const {
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

    bool intersects(const BoundingBox3D &other) const {
        const std::vector<Eigen::Vector3d> corners = get_corners();
        for (const auto &corner : corners) {
            if (other.contains(corner)) {
                return true;
            }
        }
        return false;
    }
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

    Eigen::Vector2d get_center() const {
        return Eigen::Vector2d((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
    }
    Eigen::Vector2d get_min_point() const { return Eigen::Vector2d(min_x, min_y); }
    Eigen::Vector2d get_max_point() const { return Eigen::Vector2d(max_x, max_y); }

    Eigen::Vector2d get_size() const { return Eigen::Vector2d(max_x - min_x, max_y - min_y); }

    double get_area() const { return (max_x - min_x) * (max_y - min_y); }
    double get_perimeter() const { return 2.0 * ((max_x - min_x) + (max_y - min_y)); }
    double get_diagonal_length() const {
        return std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y));
    }

    bool contains(const Eigen::Vector2d &point_w) const {
        return point_w.x() >= min_x && point_w.x() <= max_x && point_w.y() >= min_y &&
               point_w.y() <= max_y;
    }
    std::vector<bool> contains(const std::vector<Eigen::Vector2d> &points_w) const {
        std::vector<bool> contains_mask(points_w.size(), false);
        for (size_t i = 0; i < points_w.size(); i++) {
            contains_mask[i] = contains(points_w[i]);
        }
        return contains_mask;
    }
    bool intersects(const BoundingBox2D &other) const {
        return (min_x <= other.max_x && max_x >= other.min_x) &&
               (min_y <= other.max_y && max_y >= other.min_y);
    }
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
    double get_volume() const { return size.x() * size.y(); }
    double get_area() const { return size.x() * size.y(); }
    double get_perimeter() const { return 2.0 * ((size.x() + size.y())); }
    double get_diagonal_length() const {
        return std::sqrt((size.x() * size.x()) + (size.y() * size.y()));
    }

    // get the corners of the OBB in world coordinates
    std::vector<Eigen::Vector2d> get_corners() const {
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

    bool contains(const Eigen::Vector2d &point_w) const {
        const Eigen::Rotation2Dd R_wo(angle_rad);
        const Eigen::Vector2d point_o = R_wo.inverse() * (point_w - center);
        const Eigen::Vector2d half_size = size / 2.0;
        return point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
               point_o.y() >= -half_size.y() && point_o.y() <= half_size.y();
    }

    std::vector<bool> contains(const std::vector<Eigen::Vector2d> &points_w) const {
        std::vector<bool> contains_mask(points_w.size(), false);
        const Eigen::Rotation2Dd R_wo(angle_rad);
        const Eigen::Vector2d half_size = size / 2.0;
        for (size_t i = 0; i < points_w.size(); i++) {
            const Eigen::Vector2d point_o = R_wo.inverse() * (points_w[i] - center);
            contains_mask[i] = point_o.x() >= -half_size.x() && point_o.x() <= half_size.x() &&
                               point_o.y() >= -half_size.y() && point_o.y() <= half_size.y();
        }
        return contains_mask;
    }

    bool intersects(const OrientedBoundingBox2D &other) const {
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

    bool intersects(const BoundingBox2D &other) const {
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
};
