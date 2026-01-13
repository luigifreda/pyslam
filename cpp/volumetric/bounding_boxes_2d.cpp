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

#include "bounding_boxes_2d.h"

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

namespace {
inline bool sat_intersects(const detail_2d::OBBFrame2D &a, const detail_2d::OBBFrame2D &b) {
    const double eps = 1e-9;
    // Normalize axes to ensure correct SAT test (axes from rotation matrices should already be
    // unit, but be safe)
    Eigen::Vector2d a0 = a.axes.col(0).normalized();
    Eigen::Vector2d a1 = a.axes.col(1).normalized();
    Eigen::Vector2d b0 = b.axes.col(0).normalized();
    Eigen::Vector2d b1 = b.axes.col(1).normalized();
    const Eigen::Vector2d axes_to_test[4] = {a0, a1, b0, b1};

    const Eigen::Vector2d t_world = b.center - a.center;
    for (const auto &L : axes_to_test) {
        const double ra = a.half[0] * std::abs(a0.dot(L)) + a.half[1] * std::abs(a1.dot(L));
        const double rb = b.half[0] * std::abs(b0.dot(L)) + b.half[1] * std::abs(b1.dot(L));
        const double dist = std::abs(t_world.dot(L));
        if (dist > ra + rb + eps) {
            return false;
        }
    }
    return true;
}

} // namespace

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

template <typename T>
BoundingBox2D
BoundingBox2D::compute_from_points(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) {
    if (points_w.empty()) {
        return BoundingBox2D();
    }
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    for (const auto &p : points_w) {
        min_x = std::min(min_x, static_cast<double>(p.x()));
        min_y = std::min(min_y, static_cast<double>(p.y()));
        max_x = std::max(max_x, static_cast<double>(p.x()));
        max_y = std::max(max_y, static_cast<double>(p.y()));
    }
    return BoundingBox2D(min_x, min_y, max_x, max_y);
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
    // Use a small tolerance to account for floating-point errors
    const double tol = 1e-10;
    return point_o.x() >= -half_size.x() - tol && point_o.x() <= half_size.x() + tol &&
           point_o.y() >= -half_size.y() - tol && point_o.y() <= half_size.y() + tol;
}

template <typename T>
std::vector<bool>
OrientedBoundingBox2D::contains(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) const {
    std::vector<bool> contains_mask(points_w.size(), false);
    const Eigen::Rotation2Dd R_wo(angle_rad);
    const Eigen::Vector2d half_size = size / 2.0;
    // Use a small tolerance to account for floating-point errors
    const double tol = 1e-10;
    for (size_t i = 0; i < points_w.size(); i++) {
        const Eigen::Vector2d point_o =
            R_wo.inverse() * (points_w[i].template cast<double>() - center);
        contains_mask[i] =
            point_o.x() >= -half_size.x() - tol && point_o.x() <= half_size.x() + tol &&
            point_o.y() >= -half_size.y() - tol && point_o.y() <= half_size.y() + tol;
    }
    return contains_mask;
}

bool OrientedBoundingBox2D::intersects(const OrientedBoundingBox2D &other) const {
    return sat_intersects(detail_2d::make_frame(*this), detail_2d::make_frame(other));
}

bool OrientedBoundingBox2D::intersects(const BoundingBox2D &other) const {
    return sat_intersects(detail_2d::make_frame(*this), detail_2d::make_frame(other));
}

namespace detail {
template <typename T>
OrientedBoundingBox2D compute_obb_pca_2d(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) {
    // PCA-based OBB computation
    // 1. Compute centroid
    // 2. Compute covariance
    // 3. Eigen-decomposition → principal axes
    // 4. Use these axes as the box frame, project points into it
    // 5. Take min/max in that frame → extents and center

    const std::size_t n = points_w.size();
    if (n == 0) {
        // default empty box
        return OrientedBoundingBox2D();
    }
    if (n == 1) {
        OrientedBoundingBox2D obb;
        obb.center = points_w[0].template cast<double>();
        obb.size.setZero();
        obb.angle_rad = 0.0;
        return obb;
    }

    // 1. Compute centroid
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const auto &p : points_w) {
        centroid += p.template cast<double>();
    }
    centroid /= static_cast<double>(n);

    // 2. Compute covariance
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto &p : points_w) {
        Eigen::Vector2d d = p.template cast<double>() - centroid;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(n);

    // 3. Eigen-decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors(); // columns

    // Sort eigenvalues descending (and eigenvectors accordingly)
    std::array<int, 2> order = {0, 1};
    if (eigenvalues[1] > eigenvalues[0]) {
        std::swap(order[0], order[1]);
    }

    Eigen::Vector2d principal_axis = eigenvectors.col(order[0]);

    // Compute angle from principal axis (atan2 of the y/x components)
    double angle = std::atan2(principal_axis.y(), principal_axis.x());
    Eigen::Rotation2Dd R(angle);

    // 4. Project points into the frame defined by the principal axis
    Eigen::Vector2d min_pt(std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity());
    Eigen::Vector2d max_pt(-std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity());

    for (const auto &p : points_w) {
        Eigen::Vector2d local = R.inverse() * (p.template cast<double>() - centroid);
        min_pt = min_pt.cwiseMin(local);
        max_pt = max_pt.cwiseMax(local);
    }

    // 5. Compute extents and center
    Eigen::Vector2d extents = 0.5 * (max_pt - min_pt);
    Eigen::Vector2d center_local = 0.5 * (max_pt + min_pt);
    Eigen::Vector2d center_world = centroid + R * center_local;

    return OrientedBoundingBox2D(center_world, angle, 2.0 * extents);
}

// Compute 2D convex hull using Qhull
// Returns hull vertices and edges (as pairs of vertex indices)
bool compute_convex_hull_2d(const std::vector<Eigen::Vector2d> &points,
                            std::vector<Eigen::Vector2d> &hull_vertices,
                            std::vector<Eigen::Vector2i> &hull_edges) {
#if QHULL_FOUND

    using namespace orgQhull;

    hull_vertices.clear();
    hull_edges.clear();

    const int num_points = static_cast<int>(points.size());
    if (num_points < 3) {
        // Degenerate cases: just return input as hull, no edges
        hull_vertices = points;
        return false;
    }

    // --- 1. Copy Eigen points into a flat array for Qhull -------------------
    std::vector<double> coords;
    coords.reserve(static_cast<size_t>(num_points) * 2);
    for (const auto &p : points) {
        coords.push_back(p.x());
        coords.push_back(p.y());
    }

    try {
        // --- 2. Run qhull in 2D ---------------------------------------------
        Qhull qhull;
        const char *qhull_cmd = ""; // Default command for 2D (facets are edges)

        qhull.runQhull(
            /*inputComment*/ "",
            /*dimension   */ 2,
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
        const std::size_t nvertex = static_cast<std::size_t>(qhull.vertexCount());
        hull_vertices.resize(nvertex);

        std::vector<int> inputId_to_vertexIdx(static_cast<std::size_t>(num_points), -1);

        QhullVertexList vlist(qhull.vertexList());
        std::size_t i_vertex = 0;
        for (auto v = vlist.begin(); v != vlist.end(); ++v) {
            QhullPoint pt = (*v).point();
            const int input_id = pt.id();

            inputId_to_vertexIdx[static_cast<std::size_t>(input_id)] = static_cast<int>(i_vertex);

            // Copy coordinates to Eigen (only x, y)
            hull_vertices[i_vertex] = Eigen::Vector2d(pt[0], pt[1]);

            ++i_vertex;
        }

        // --- 4. Extract edges from facets (each facet is an edge in 2D) -----
        std::set<std::pair<int, int>> unique_edges;

        for (QhullFacet facet = qhull.beginFacet(); facet != qhull.endFacet();
             facet = facet.next()) {
            if (!facet.isGood()) {
                continue;
            }

            QhullVertexSet fv = facet.vertices();
            const std::size_t n = static_cast<std::size_t>(fv.count());
            if (n != 2) {
                continue;
            }

            const int i0 = inputId_to_vertexIdx[static_cast<std::size_t>(fv[0].point().id())];
            const int i1 = inputId_to_vertexIdx[static_cast<std::size_t>(fv[1].point().id())];
            if (i0 < 0 || i1 < 0) {
                continue;
            }
            std::pair<int, int> edge = (i0 < i1) ? std::make_pair(i0, i1) : std::make_pair(i1, i0);
            unique_edges.insert(edge);
        }

        hull_edges.reserve(unique_edges.size());
        for (const auto &e : unique_edges) {
            hull_edges.emplace_back(e.first, e.second);
        }
    } catch (const QhullError &e) {
        throw;
    }

    return true;
#else
    throw std::runtime_error("Qhull not found. Please install Qhull and rebuild the project.");
    return false;
#endif
}

template <typename T>
OrientedBoundingBox2D
compute_obb_convex_hull_minimal_2d(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w) {
    // Convex hull-based OBB computation for 2D
    // 1. Compute convex hull → hull vertices + edge indices
    // 2. For each edge in the hull
    //  a. Use edge direction as first axis (u)
    //  b. Use perpendicular direction as second axis (v)
    //  c. Form rotation matrix R=[u v]
    //  d. Project all original points into this frame: q = R^T * p
    //  e. Get min/max along each axis, compute extents, center, area
    //  f. Keep the box with the smallest area

    // Cast to double for computation
    std::vector<Eigen::Vector2d> pts_d;
    pts_d.reserve(points_w.size());
    for (const auto &p : points_w) {
        pts_d.emplace_back(static_cast<double>(p.x()), static_cast<double>(p.y()));
    }

    OrientedBoundingBox2D obb;

    if (pts_d.size() < 2) {
        // For degenerate cases, fall back to PCA
        return detail::compute_obb_pca_2d(points_w);
    }

    // 1. Compute convex hull
    std::vector<Eigen::Vector2d> hull_vertices;
    std::vector<Eigen::Vector2i> hull_edges; // (i,j) indices into hull_vertices

    if (!compute_convex_hull_2d(pts_d, hull_vertices, hull_edges) || hull_vertices.empty() ||
        hull_edges.empty()) {
        // Fallback if hull computation fails
        return detail::compute_obb_pca_2d(points_w);
    }

    const double INF = std::numeric_limits<double>::infinity();
    double best_area = INF;
    Eigen::Vector2d best_center = Eigen::Vector2d::Zero();
    double best_angle = 0.0;
    Eigen::Vector2d best_extents = Eigen::Vector2d::Zero();

    const double eps = 1e-9;

    // 2. For each edge in the hull, define a frame and compute AABB in that frame
    for (const Eigen::Vector2i &edge : hull_edges) {
        // Bounds check
        if (edge[0] < 0 || static_cast<std::size_t>(edge[0]) >= hull_vertices.size() ||
            edge[1] < 0 || static_cast<std::size_t>(edge[1]) >= hull_vertices.size()) {
            continue;
        }
        const Eigen::Vector2d &p0 = hull_vertices[edge[0]];
        const Eigen::Vector2d &p1 = hull_vertices[edge[1]];

        Eigen::Vector2d e0 = p1 - p0;

        if (e0.squaredNorm() < eps) {
            continue; // degenerate edge
        }

        Eigen::Vector2d u = e0.normalized(); // axis 0 (along edge)

        // Perpendicular axis (rotate u by 90 degrees)
        Eigen::Vector2d v(-u.y(), u.x()); // axis 1 (perpendicular to edge)

        // Form rotation matrix R = [u v]
        Eigen::Matrix2d R;
        R.col(0) = u;
        R.col(1) = v;

        // Ensure right-handed (should already be, but just in case)
        if (R.determinant() < 0.0) {
            R.col(1) = -R.col(1);
        }

        // 3. Project all original points into this frame
        Eigen::Vector2d min_pt(INF, INF);
        Eigen::Vector2d max_pt(-INF, -INF);

        for (const auto &p : hull_vertices) {
            Eigen::Vector2d q = R.transpose() * p; // local coordinates
            min_pt = min_pt.cwiseMin(q);
            max_pt = max_pt.cwiseMax(q);
        }

        Eigen::Vector2d extents = 0.5 * (max_pt - min_pt);
        Eigen::Vector2d center_local = 0.5 * (max_pt + min_pt);
        Eigen::Vector2d center_world = R * center_local;

        double area = 4.0 * extents.x() * extents.y();
        if (!std::isfinite(area)) {
            continue;
        }

        // Compute angle from rotation matrix
        double angle = std::atan2(R(1, 0), R(0, 0));

        if (area < best_area) {
            best_area = area;
            best_center = center_world;
            best_angle = angle;
            best_extents = extents;
        }
    }

    if (best_area == INF) {
        // all edges degenerate? fallback
        return detail::compute_obb_pca_2d(points_w);
    }

    obb.center = best_center;
    obb.angle_rad = best_angle;
    obb.size = 2.0 * best_extents;

    return obb;
}

} // namespace detail

template <typename T>
OrientedBoundingBox2D
OrientedBoundingBox2D::compute_from_points(const std::vector<Eigen::Matrix<T, 2, 1>> &points_w,
                                           const OBBComputationMethod &method) {
    switch (method) {
    case OBBComputationMethod::PCA:
        return detail::compute_obb_pca_2d(points_w);
    case OBBComputationMethod::CONVEX_HULL_MINIMAL:
        return detail::compute_obb_convex_hull_minimal_2d(points_w);
    default:
        return detail::compute_obb_pca_2d(points_w);
    }
}

// ================================================
// Explicit template instantiations
// ================================================

template bool BoundingBox2D::contains<double>(const Eigen::Matrix<double, 2, 1> &point_w) const;
template bool BoundingBox2D::contains<float>(const Eigen::Matrix<float, 2, 1> &point_w) const;
template std::vector<bool>
BoundingBox2D::contains<double>(const std::vector<Eigen::Matrix<double, 2, 1>> &points_w) const;
template std::vector<bool>
BoundingBox2D::contains<float>(const std::vector<Eigen::Matrix<float, 2, 1>> &points_w) const;

template bool BoundingBox2D::contains<double>(const double x_w, const double y_w) const;
template bool BoundingBox2D::contains<float>(const float x_w, const float y_w) const;
template bool
OrientedBoundingBox2D::contains<double>(const Eigen::Matrix<double, 2, 1> &point_w) const;
template bool
OrientedBoundingBox2D::contains<float>(const Eigen::Matrix<float, 2, 1> &point_w) const;
template bool OrientedBoundingBox2D::contains<double>(const double x_w, const double y_w) const;
template bool OrientedBoundingBox2D::contains<float>(const float x_w, const float y_w) const;
template std::vector<bool> OrientedBoundingBox2D::contains<double>(
    const std::vector<Eigen::Matrix<double, 2, 1>> &points_w) const;
template std::vector<bool> OrientedBoundingBox2D::contains<float>(
    const std::vector<Eigen::Matrix<float, 2, 1>> &points_w) const;

// Explicit template instantiations for compute_from_points
template BoundingBox2D BoundingBox2D::compute_from_points<double>(
    const std::vector<Eigen::Matrix<double, 2, 1>> &points_w);
template BoundingBox2D
BoundingBox2D::compute_from_points<float>(const std::vector<Eigen::Matrix<float, 2, 1>> &points_w);

template OrientedBoundingBox2D OrientedBoundingBox2D::compute_from_points<double>(
    const std::vector<Eigen::Matrix<double, 2, 1>> &points_w, const OBBComputationMethod &method);
template OrientedBoundingBox2D OrientedBoundingBox2D::compute_from_points<float>(
    const std::vector<Eigen::Matrix<float, 2, 1>> &points_w, const OBBComputationMethod &method);

} // namespace volumetric
