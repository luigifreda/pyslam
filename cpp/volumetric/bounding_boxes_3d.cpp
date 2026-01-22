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

#include "bounding_boxes_3d.h"

#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

#if QHULL_FOUND
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullError.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullLinkedList.h>
#include <libqhullcpp/QhullVertex.h>
#include <libqhullcpp/QhullVertexSet.h>
#endif

namespace volumetric {

namespace detail {
// Helper trait to extract scalar type from VecT (Eigen::Matrix or std::array)
template <typename VecT, typename = void> struct vector_scalar_type {
    // For std::array<T, 3>, extract T
    using type = typename std::remove_reference<decltype(std::declval<VecT>()[0])>::type;
};

// Specialization for Eigen types that have Scalar member
template <typename VecT> struct vector_scalar_type<VecT, std::void_t<typename VecT::Scalar>> {
    using type = typename VecT::Scalar;
};

template <typename VecT> using vector_scalar_t = typename vector_scalar_type<VecT>::type;

// Helper trait to check if VecT is an Eigen type (has Scalar member)
template <typename VecT, typename = void> struct is_eigen_type : std::false_type {};

template <typename VecT>
struct is_eigen_type<VecT, std::void_t<typename VecT::Scalar>> : std::true_type {};

template <typename VecT> constexpr bool is_eigen_type_v = is_eigen_type<VecT>::value;
} // namespace detail

namespace {

inline bool sat_intersects(const detail_3d::OBBFrame3D &a, const detail_3d::OBBFrame3D &b) {
    const double eps = 1e-9;

    Eigen::Matrix3d R;
    Eigen::Matrix3d AbsR;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = a.axes.col(i).dot(b.axes.col(j));
            AbsR(i, j) = std::abs(R(i, j)) + eps;
        }
    }

    // translation in A frame
    Eigen::Vector3d t_world = b.center - a.center;
    Eigen::Vector3d t(t_world.dot(a.axes.col(0)), t_world.dot(a.axes.col(1)),
                      t_world.dot(a.axes.col(2)));

    // Test A's axes
    for (int i = 0; i < 3; ++i) {
        const double ra = a.half[i];
        const double rb = b.half[0] * AbsR(i, 0) + b.half[1] * AbsR(i, 1) + b.half[2] * AbsR(i, 2);
        if (std::abs(t[i]) > ra + rb) {
            return false;
        }
    }

    // Test B's axes
    for (int j = 0; j < 3; ++j) {
        const double ra = a.half[0] * AbsR(0, j) + a.half[1] * AbsR(1, j) + a.half[2] * AbsR(2, j);
        const double rb = b.half[j];
        const double t_proj = std::abs(t_world.dot(b.axes.col(j)));
        if (t_proj > ra + rb) {
            return false;
        }
    }

    // Test cross products of axes
    // From Gottschalk et al. 1996 (OBBTree)
    double ra, rb, t_val;

    // L = A0 x B0
    ra = a.half[1] * AbsR(2, 0) + a.half[2] * AbsR(1, 0);
    rb = b.half[1] * AbsR(0, 2) + b.half[2] * AbsR(0, 1);
    t_val = std::abs(t[2] * R(1, 0) - t[1] * R(2, 0));
    if (t_val > ra + rb)
        return false;

    // L = A0 x B1
    ra = a.half[1] * AbsR(2, 1) + a.half[2] * AbsR(1, 1);
    rb = b.half[0] * AbsR(0, 2) + b.half[2] * AbsR(0, 0);
    t_val = std::abs(t[2] * R(1, 1) - t[1] * R(2, 1));
    if (t_val > ra + rb)
        return false;

    // L = A0 x B2
    ra = a.half[1] * AbsR(2, 2) + a.half[2] * AbsR(1, 2);
    rb = b.half[0] * AbsR(0, 1) + b.half[1] * AbsR(0, 0);
    t_val = std::abs(t[2] * R(1, 2) - t[1] * R(2, 2));
    if (t_val > ra + rb)
        return false;

    // L = A1 x B0
    ra = a.half[0] * AbsR(2, 0) + a.half[2] * AbsR(0, 0);
    rb = b.half[1] * AbsR(1, 2) + b.half[2] * AbsR(1, 1);
    t_val = std::abs(t[0] * R(2, 0) - t[2] * R(0, 0));
    if (t_val > ra + rb)
        return false;

    // L = A1 x B1
    ra = a.half[0] * AbsR(2, 1) + a.half[2] * AbsR(0, 1);
    rb = b.half[0] * AbsR(1, 2) + b.half[2] * AbsR(1, 0);
    t_val = std::abs(t[0] * R(2, 1) - t[2] * R(0, 1));
    if (t_val > ra + rb)
        return false;

    // L = A1 x B2
    ra = a.half[0] * AbsR(2, 2) + a.half[2] * AbsR(0, 2);
    rb = b.half[0] * AbsR(1, 1) + b.half[1] * AbsR(1, 0);
    t_val = std::abs(t[0] * R(2, 2) - t[2] * R(0, 2));
    if (t_val > ra + rb)
        return false;

    // L = A2 x B0
    ra = a.half[0] * AbsR(1, 0) + a.half[1] * AbsR(0, 0);
    rb = b.half[1] * AbsR(2, 2) + b.half[2] * AbsR(2, 1);
    t_val = std::abs(t[1] * R(0, 0) - t[0] * R(1, 0));
    if (t_val > ra + rb)
        return false;

    // L = A2 x B1
    ra = a.half[0] * AbsR(1, 1) + a.half[1] * AbsR(0, 1);
    rb = b.half[0] * AbsR(2, 2) + b.half[2] * AbsR(2, 0);
    t_val = std::abs(t[1] * R(0, 1) - t[0] * R(1, 1));
    if (t_val > ra + rb)
        return false;

    // L = A2 x B2
    ra = a.half[0] * AbsR(1, 2) + a.half[1] * AbsR(0, 2);
    rb = b.half[0] * AbsR(2, 1) + b.half[1] * AbsR(2, 0);
    t_val = std::abs(t[1] * R(0, 2) - t[0] * R(1, 2));
    if (t_val > ra + rb)
        return false;

    return true;
}

} // namespace

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

template <typename T>
BoundingBox3D
BoundingBox3D::compute_from_points(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w) {
    if (points_w.empty()) {
        return BoundingBox3D();
    }
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();
    for (const auto &point_w : points_w) {
        min_x = std::min(min_x, static_cast<double>(point_w.x()));
        min_y = std::min(min_y, static_cast<double>(point_w.y()));
        min_z = std::min(min_z, static_cast<double>(point_w.z()));
        max_x = std::max(max_x, static_cast<double>(point_w.x()));
        max_y = std::max(max_y, static_cast<double>(point_w.y()));
        max_z = std::max(max_z, static_cast<double>(point_w.z()));
    }
    return BoundingBox3D(min_x, min_y, min_z, max_x, max_y, max_z);
}

// ================================================
// OrientedBoundingBox3D implementations
// ================================================

// Transformation matrix from object-attached to world coordinates
Eigen::Matrix4d OrientedBoundingBox3D::get_matrix() const {
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    matrix.block<3, 3>(0, 0) = orientation.normalized().toRotationMatrix();
    matrix.block<3, 1>(0, 3) = center;
    return matrix;
}

// Transformation matrix from world to object-attached coordinates
Eigen::Matrix4d OrientedBoundingBox3D::get_inverse_matrix() const {
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d R_ow = orientation.normalized().toRotationMatrix().transpose();
    const Eigen::Vector3d t_ow = -R_ow * center;
    matrix.block<3, 3>(0, 0) = R_ow;
    matrix.block<3, 1>(0, 3) = t_ow;
    return matrix;
}

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
        q_norm.normalized().toRotationMatrix(); // from object-attached axes to world coordinates,
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
    // Use a small tolerance to account for floating-point errors
    const double tol = 1e-10;
    return point_o.x() >= -half_size.x() - tol && point_o.x() <= half_size.x() + tol &&
           point_o.y() >= -half_size.y() - tol && point_o.y() <= half_size.y() + tol &&
           point_o.z() >= -half_size.z() - tol && point_o.z() <= half_size.z() + tol;
}

template <typename T>
std::vector<bool>
OrientedBoundingBox3D::contains(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w) const {
    std::vector<bool> contains_mask(points_w.size(), false);
    const Eigen::Quaterniond q_norm = orientation.normalized();
    const Eigen::Vector3d half_size = size / 2.0;
    // Use a small tolerance to account for floating-point errors
    const double tol = 1e-10;
    for (size_t i = 0; i < points_w.size(); i++) {
        const Eigen::Vector3d point_o =
            q_norm.inverse() * (points_w[i].template cast<double>() - center);
        contains_mask[i] =
            point_o.x() >= -half_size.x() - tol && point_o.x() <= half_size.x() + tol &&
            point_o.y() >= -half_size.y() - tol && point_o.y() <= half_size.y() + tol &&
            point_o.z() >= -half_size.z() - tol && point_o.z() <= half_size.z() + tol;
    }
    return contains_mask;
}

bool OrientedBoundingBox3D::intersects(const OrientedBoundingBox3D &other) const {
    return sat_intersects(detail_3d::make_frame(*this), detail_3d::make_frame(other));
}

bool OrientedBoundingBox3D::intersects(const BoundingBox3D &other) const {
    return sat_intersects(detail_3d::make_frame(*this), detail_3d::make_frame(other));
}

namespace detail {

// PCA-based OBB computation
// 1. Compute centroid
// 2. Compute covariance
// 3. Eigen-decomposition → principal axes
// 4. Use these axes as the box frame, project points into it
// 5. Take min/max in that frame → extents and center
template <typename VecT> // accepts Eigen::Matrix<T, 3, 1> or std::array<T, 3>
inline OrientedBoundingBox3D compute_obb_pca_3d(const std::vector<VecT> &points_w) {
    OrientedBoundingBox3D obb;

    const std::size_t n = points_w.size();
    if (n == 0) {
        // default empty box
        return obb;
    }

    // Extract scalar type using helper trait
    using Scalar = detail::vector_scalar_t<VecT>;

    if (n == 1) {
        const auto &point = points_w[0];
        if constexpr (detail::is_eigen_type_v<VecT>) {
            obb.center = point.template cast<double>();
        } else {
            obb.center = Eigen::Vector3d(point[0], point[1], point[2]);
        }

        obb.size.setZero();
        obb.orientation.setIdentity();
        return obb;
    }
    if (n == 2) {
        const auto &point1 = points_w[0];
        const auto &point2 = points_w[1];
        Eigen::Vector3d c;
        Eigen::Vector3d diff;
        if constexpr (detail::is_eigen_type_v<VecT>) {
            const auto p1 = point1.template cast<double>();
            const auto p2 = point2.template cast<double>();
            c = 0.5 * (p1 + p2);
            diff = p2 - p1;
        } else {
            const auto p1 = Eigen::Vector3d(point1[0], point1[1], point1[2]);
            const auto p2 = Eigen::Vector3d(point2[0], point2[1], point2[2]);
            c = 0.5 * (p1 + p2);
            diff = p2 - p1;
        }
        const double diff_norm = diff.norm();

        if (diff_norm < 1e-10) {
            // Degenerate case: two identical points
            obb.center = c;
            obb.size.setZero();
            obb.orientation.setIdentity();
            return obb;
        }

        // Compute orientation: align the segment direction with the x-axis
        // The segment direction becomes the first principal axis
        Eigen::Vector3d segment_dir = diff / diff_norm;

        // Build an orthonormal basis with segment_dir as the first axis
        // Choose a vector not parallel to segment_dir for the second axis
        Eigen::Vector3d axis1 = segment_dir;
        Eigen::Vector3d axis2;

        // Find a vector perpendicular to segment_dir
        if (std::abs(axis1.x()) < 0.9) {
            axis2 = Eigen::Vector3d(1.0, 0.0, 0.0).cross(axis1).normalized();
        } else {
            axis2 = Eigen::Vector3d(0.0, 1.0, 0.0).cross(axis1).normalized();
        }

        Eigen::Vector3d axis3 = axis1.cross(axis2).normalized();

        // Build rotation matrix: columns are the basis vectors
        Eigen::Matrix3d R;
        R.col(0) = axis1;
        R.col(1) = axis2;
        R.col(2) = axis3;

        // Ensure right-handed
        if (R.determinant() < 0.0) {
            R.col(2) = -R.col(2);
        }

        // Project points into this frame to get extents
        Eigen::Vector3d min_pt(std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity());
        Eigen::Vector3d max_pt(-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity());

        for (const auto &p : points_w) {
            Eigen::Vector3d local;
            if constexpr (std::is_same_v<VecT, Eigen::Matrix<Scalar, 3, 1>> ||
                          std::is_base_of_v<Eigen::MatrixBase<VecT>, VecT>) {
                local = R.transpose() * (p.template cast<double>() - c);
            } else {
                local = R.transpose() * (Eigen::Vector3d(p[0], p[1], p[2]) - c);
            }
            min_pt = min_pt.cwiseMin(local);
            max_pt = max_pt.cwiseMax(local);
        }

        Eigen::Vector3d extents = 0.5 * (max_pt - min_pt);
        Eigen::Vector3d center_local = 0.5 * (max_pt + min_pt);
        Eigen::Vector3d center_world = c + R * center_local;

        obb.center = center_world;
        obb.size = 2.0 * extents;
        obb.orientation = Eigen::Quaterniond(R);
        return obb;
    }

    // 1-2. Centroid and covariance in one pass (Welford)
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    std::size_t k = 0;
    for (const auto &p : points_w) {
        ++k;
        Eigen::Vector3d delta;
        Eigen::Vector3d delta2;
        if constexpr (detail::is_eigen_type_v<VecT>) {
            const auto p_d = p.template cast<double>();
            delta = p_d - centroid;
            centroid += delta / static_cast<double>(k);
            delta2 = p_d - centroid;
        } else {
            const auto p_d = Eigen::Vector3d(p[0], p[1], p[2]);
            delta = p_d - centroid;
            centroid += delta / static_cast<double>(k);
            delta2 = p_d - centroid;
        }
        cov += delta * delta2.transpose();
    }
    cov /= static_cast<double>(n);

    // 3. Eigen-decomposition of symmetric covariance
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    Eigen::Matrix3d eigenvectors = solver.eigenvectors(); // columns

    // sort eigenvalues descending (and eigenvectors accordingly)
    std::array<int, 3> order = {0, 1, 2};
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (eigenvalues[order[j]] > eigenvalues[order[i]]) {
                std::swap(order[i], order[j]);
            }
        }
    }

    Eigen::Matrix3d R;
    R.col(0) = eigenvectors.col(order[0]);
    R.col(1) = eigenvectors.col(order[1]);
    R.col(2) = eigenvectors.col(order[2]);

    // Ensure right-handed basis
    if (R.col(0).cross(R.col(1)).dot(R.col(2)) < 0.0) {
        R.col(2) = -R.col(2);
    }

    // 4. Project points into PCA frame
    Eigen::Vector3d min_pt(std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity());
    Eigen::Vector3d max_pt(-std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity());

    for (const auto &p : points_w) {
        Eigen::Vector3d local;
        if constexpr (detail::is_eigen_type_v<VecT>) {
            local = R.transpose() * (p.template cast<double>() - centroid);
        } else {
            local = R.transpose() * (Eigen::Vector3d(p[0], p[1], p[2]) - centroid);
        }
        min_pt = min_pt.cwiseMin(local);
        max_pt = max_pt.cwiseMax(local);
    }

    Eigen::Vector3d extents = 0.5 * (max_pt - min_pt); // half-lengths
    Eigen::Vector3d center_local = 0.5 * (max_pt + min_pt);
    Eigen::Vector3d center_world = centroid + R * center_local;

    obb.center = center_world;
    obb.orientation = Eigen::Quaterniond(R);
    obb.size = 2.0 * extents;

    return obb;
}

// Fill hull_vertices and triangle indices of the 3D convex hull of input points.
// You can implement this via your own QuickHull, CGAL, or wrap Qhull, etc.
bool compute_convex_hull_3d(const std::vector<Eigen::Vector3d> &points,
                            std::vector<Eigen::Vector3d> &hull_vertices,
                            std::vector<Eigen::Vector3i> &hull_triangles) {
#if QHULL_FOUND

    using namespace orgQhull;

    hull_vertices.clear();
    hull_triangles.clear();

    const int num_points = static_cast<int>(points.size());
    if (num_points < 4) {
        // Degenerate cases: just return input as hull, no triangles
        hull_vertices = points;
        return false;
    }

    // --- 1. Copy Eigen points into a flat array for Qhull -------------------
    // (this is safer than relying on Eigen's internal layout / alignment)
    std::vector<double> coords;
    coords.reserve(static_cast<size_t>(num_points) * 3);
    for (const auto &p : points) {
        coords.push_back(p.x());
        coords.push_back(p.y());
        coords.push_back(p.z());
    }

    try {
        // --- 2. Run qhull in 3D with triangulation ("Qt") -------------------
        Qhull qhull;
        const char *qhull_cmd = "Qt"; // triangulate non-simplicial facets

        qhull.runQhull(
            /*inputComment*/ "",
            /*dimension   */ 3,
            /*pointCount  */ num_points,
            /*coords      */ coords.data(),
            /*qhullCommand*/ qhull_cmd);

        if (qhull.qhullStatus() != 0) {
            if (qhull.hasQhullMessage()) {
                throw QhullError(qhull.qhullStatus(), qhull.qhullMessage().c_str());
            } else {
                throw QhullError(qhull.qhullStatus(), "Qhull failed without message");
            }
        }

        // --- 3. Map original point ids -> hull vertex indices ---------------
        // qhull gives us a subset of input points as the convex hull vertices.
        const std::size_t nvertex = static_cast<std::size_t>(qhull.vertexCount());
        hull_vertices.resize(nvertex);

        std::vector<int> inputId_to_vertexIdx(static_cast<std::size_t>(num_points), -1);

        QhullVertexList vlist(qhull.vertexList());
        std::size_t i_vertex = 0;
        for (auto v = vlist.begin(); v != vlist.end(); ++v) {
            QhullPoint pt = (*v).point(); // point on the hull, refers to an input point
            const int input_id = pt.id(); // index in the original input array

            // Map input index -> hull vertex index
            inputId_to_vertexIdx[static_cast<std::size_t>(input_id)] = static_cast<int>(i_vertex);

            // Copy coordinates to Eigen
            hull_vertices[i_vertex] = Eigen::Vector3d(pt[0], pt[1], pt[2]);

            ++i_vertex;
        }

        // --- 4. Extract triangular facets as index triplets -----------------
        // Thanks to "Qt", facets should be simplicial in 3D (triangles).
        for (QhullFacet facet = qhull.beginFacet(); facet != qhull.endFacet();
             facet = facet.next()) {
            if (!facet.isGood()) {
                // skip non-"good" facets (depends on your needs; you can also keep them)
                continue;
            }

            if (facet.isSimplicial()) {
                QhullVertexSet fv = facet.vertices();
                const std::size_t n = static_cast<std::size_t>(fv.count());
                if (n != 3) {
                    // Should not happen with Qt in pure 3D, but be robust.
                    continue;
                }

                const int i0 = inputId_to_vertexIdx[static_cast<std::size_t>(fv[0].point().id())];
                const int i1 = inputId_to_vertexIdx[static_cast<std::size_t>(fv[1].point().id())];
                const int i2 = inputId_to_vertexIdx[static_cast<std::size_t>(fv[2].point().id())];

                if (i0 < 0 || i1 < 0 || i2 < 0) {
                    // Something went wrong with mapping; skip this facet
                    continue;
                }

                hull_triangles.emplace_back(i0, i1, i2);
            } else {
                // With "Qt" you shouldn't hit this, but you could triangulate
                // non-simplicial facets here if you drop "Qt".
                // For OBB you usually only need the hull points, so you can also ignore.
            }
        }
    } catch (const QhullError &e) {
        // Handle or propagate — here we just throw up to the caller
        throw;
    }

    return true;
#else
    throw std::runtime_error("Qhull not found. Please install Qhull and rebuild the project.");
    return false;
#endif
}

// Convex hull-based OBB computation
// 1. Compute convex hull → hull vertices + triangle indices.
// 2. For each triangle in the hull
//  a. Build an orthonormal basis:
//      u (normalized first edge),
//      v (normalized component of second edge orthogonal to u)
//      w (cross product of u and v)
//  b. Form rotation matrix  R=[u v w]
//  c. Project all original (or hull) points into this frame: q= R^T * p
//  d. Get min / max along each axis, compute extents, center, volume.
//  e. Keep the box with the smallest volume.
inline OrientedBoundingBox3D
compute_obb_convex_hull_minimal_3d(const std::vector<Eigen::Vector3d> &points_w) {
    OrientedBoundingBox3D obb;

    if (points_w.size() < 3) {
        // For degenerate cases, just fall back to PCA or an AABB
        return compute_obb_pca_3d(points_w);
    }

    // 1. Compute convex hull
    std::vector<Eigen::Vector3d> hull_vertices;
    std::vector<Eigen::Vector3i> hull_triangles; // (i,j,k) indices into hull_vertices

    if (!compute_convex_hull_3d(points_w, hull_vertices, hull_triangles) || hull_vertices.empty() ||
        hull_triangles.empty()) {
        // Fallback if hull computation fails
        return compute_obb_pca_3d(points_w);
    }

    const double INF = std::numeric_limits<double>::infinity();
    double best_volume = INF;
    Eigen::Vector3d best_center = Eigen::Vector3d::Zero();
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d best_extents = Eigen::Vector3d::Zero();

    const double eps = 1e-9;

    // 2. For each triangle in the hull, define a frame and compute AABB in that frame
    for (const Eigen::Vector3i &tri : hull_triangles) {
        // Bounds check
        if (tri[0] < 0 || static_cast<std::size_t>(tri[0]) >= hull_vertices.size() || tri[1] < 0 ||
            static_cast<std::size_t>(tri[1]) >= hull_vertices.size() || tri[2] < 0 ||
            static_cast<std::size_t>(tri[2]) >= hull_vertices.size()) {
            continue;
        }
        const Eigen::Vector3d &p0 = hull_vertices[tri[0]];
        const Eigen::Vector3d &p1 = hull_vertices[tri[1]];
        const Eigen::Vector3d &p2 = hull_vertices[tri[2]];

        Eigen::Vector3d e0 = p1 - p0;
        Eigen::Vector3d e1 = p2 - p0;

        if (e0.squaredNorm() < eps || e1.squaredNorm() < eps) {
            continue; // degenerate
        }

        Eigen::Vector3d u = e0.normalized(); // axis 0

        // Remove component of e1 along u to get an independent axis in the triangle plane
        Eigen::Vector3d e1_ortho = e1 - e1.dot(u) * u;
        if (e1_ortho.squaredNorm() < eps) {
            continue; // nearly collinear
        }

        Eigen::Vector3d v = e1_ortho.normalized(); // axis 1
        Eigen::Vector3d w = u.cross(v);            // axis 2 (normal to triangle plane)
        double w_norm = w.norm();
        if (w_norm < eps) {
            continue; // degenerate
        }
        w /= w_norm;

        Eigen::Matrix3d R;
        R.col(0) = u;
        R.col(1) = v;
        R.col(2) = w;

        // Ensure right-handed (should already be, but just in case)
        if (R.col(0).cross(R.col(1)).dot(R.col(2)) < 0.0) {
            R.col(2) = -R.col(2);
        }

        // 3. Project all original (or hull) points into this frame:
        Eigen::Vector3d min_pt(INF, INF, INF);
        Eigen::Vector3d max_pt(-INF, -INF, -INF);

        for (const auto &p : hull_vertices) {
            Eigen::Vector3d q = R.transpose() * p; // local coordinates (no translation)
            min_pt = min_pt.cwiseMin(q);
            max_pt = max_pt.cwiseMax(q);
        }

        Eigen::Vector3d extents = 0.5 * (max_pt - min_pt);
        Eigen::Vector3d center_local = 0.5 * (max_pt + min_pt);
        Eigen::Vector3d center_world = R * center_local;

        double volume = 8.0 * extents.x() * extents.y() * extents.z();
        if (!std::isfinite(volume)) {
            continue;
        }

        if (volume < best_volume) {
            best_volume = volume;
            best_center = center_world;
            best_R = R;
            best_extents = extents;
        }
    }

    if (best_volume == INF) {
        // all triangles degenerate? fallback
        return compute_obb_pca_3d(points_w);
    }

    obb.center = best_center;
    obb.orientation = Eigen::Quaterniond(best_R);
    obb.size = 2.0 * best_extents;

    return obb;
}

template <typename T>
std::vector<Eigen::Vector3d>
convert_to_eigen_vector3d(const std::vector<Eigen::Matrix<T, 3, 1>> &points) {
    std::vector<Eigen::Vector3d> pts_d;
    pts_d.reserve(points.size());
    for (const auto &p : points) {
        pts_d.emplace_back(p.template cast<double>());
    }
    return pts_d;
};

template <typename T>
std::vector<Eigen::Vector3d>
convert_to_eigen_vector3d(const std::vector<std::array<T, 3>> &points) {
    std::vector<Eigen::Vector3d> pts_d;
    pts_d.reserve(points.size());
    for (const auto &p : points) {
        pts_d.emplace_back(Eigen::Vector3d(p[0], p[1], p[2]));
    }
    return pts_d;
};

} // namespace detail

template <typename T>
OrientedBoundingBox3D
OrientedBoundingBox3D::compute_from_points(const std::vector<Eigen::Matrix<T, 3, 1>> &points_w,
                                           const OBBComputationMethod &method) {
    switch (method) {
    case OBBComputationMethod::PCA:
        return detail::compute_obb_pca_3d(points_w);
    case OBBComputationMethod::CONVEX_HULL_MINIMAL:
        if constexpr (std::is_same_v<T, double>) {
            return detail::compute_obb_convex_hull_minimal_3d(points_w);
        } else {
            return detail::compute_obb_convex_hull_minimal_3d(
                detail::convert_to_eigen_vector3d(points_w));
        }
    default:
        return detail::compute_obb_pca_3d(points_w);
    }
}

template <typename T>
OrientedBoundingBox3D
OrientedBoundingBox3D::compute_from_points(const std::vector<std::array<T, 3>> &points_w,
                                           const OBBComputationMethod &method) {
    switch (method) {
    case OBBComputationMethod::PCA:
        return detail::compute_obb_pca_3d(points_w);
    case OBBComputationMethod::CONVEX_HULL_MINIMAL:
        return detail::compute_obb_convex_hull_minimal_3d(
            detail::convert_to_eigen_vector3d(points_w));
    default:
        return detail::compute_obb_pca_3d(points_w);
    }
}

// ================================================
// Explicit template instantiations
// ================================================

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

// Explicit template instantiations for compute_from_points
template BoundingBox3D BoundingBox3D::compute_from_points<double>(
    const std::vector<Eigen::Matrix<double, 3, 1>> &points_w);
template BoundingBox3D
BoundingBox3D::compute_from_points<float>(const std::vector<Eigen::Matrix<float, 3, 1>> &points_w);

template OrientedBoundingBox3D OrientedBoundingBox3D::compute_from_points<double>(
    const std::vector<Eigen::Matrix<double, 3, 1>> &points_w, const OBBComputationMethod &method);
template OrientedBoundingBox3D OrientedBoundingBox3D::compute_from_points<float>(
    const std::vector<Eigen::Matrix<float, 3, 1>> &points_w, const OBBComputationMethod &method);

template OrientedBoundingBox3D OrientedBoundingBox3D::compute_from_points<double>(
    const std::vector<std::array<double, 3>> &points_w, const OBBComputationMethod &method);
template OrientedBoundingBox3D
OrientedBoundingBox3D::compute_from_points<float>(const std::vector<std::array<float, 3>> &points_w,
                                                  const OBBComputationMethod &method);

} // namespace volumetric
