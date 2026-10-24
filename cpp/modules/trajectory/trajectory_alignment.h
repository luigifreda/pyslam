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
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace trajectory_tools {

// ================================================================
// Association functions
// ================================================================

struct AssocResult {
    Eigen::VectorXd timestamps;
    Eigen::Matrix<double, 3, Eigen::Dynamic> filter_points;
    Eigen::Matrix<double, 3, Eigen::Dynamic> gt_points;
    double max_dt;
};

std::tuple<std::vector<double>, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
find_trajectories_associations(const std::vector<double> &filter_timestamps,
                               const std::vector<Eigen::Vector3d> &filter_t_wi,
                               const std::vector<double> &gt_timestamps,
                               const std::vector<Eigen::Vector3d> &gt_t_wi,
                               double max_align_dt = 1e-1, bool verbose = true) {

    const size_t num_filter_timestamps = filter_timestamps.size();

    std::vector<double> timestamps_associations;
    std::vector<Eigen::Vector3d> filter_associations;
    std::vector<Eigen::Vector3d> gt_associations;

    timestamps_associations.reserve(num_filter_timestamps);
    filter_associations.reserve(num_filter_timestamps);
    gt_associations.reserve(num_filter_timestamps);

    double max_dt = 0.0;

    for (size_t i = 0; i < filter_timestamps.size(); ++i) {
        double timestamp = filter_timestamps[i];

        // Find the index of the GT timestamp that is greater than the filter timestamp
        auto upper = std::upper_bound(gt_timestamps.begin(), gt_timestamps.end(), timestamp);
        // Get the previous index so that gt_timestamps[j] <= timestamp < gt_timestamps[j+1]
        int j = std::distance(gt_timestamps.begin(), upper) - 1;

        if (j < 0 || j >= static_cast<int>(gt_timestamps.size()) - 1)
            continue;

        double dt = timestamp - gt_timestamps[j];
        double dt_gt = gt_timestamps[j + 1] - gt_timestamps[j];
        double abs_dt = std::abs(dt);

        if (dt < 0 || dt_gt <= 0 || abs_dt > max_align_dt)
            continue;

        max_dt = std::max(max_dt, abs_dt);
        double ratio = dt / dt_gt;

        Eigen::Vector3d interpolated = (1.0 - ratio) * gt_t_wi[j] + ratio * gt_t_wi[j + 1];

        timestamps_associations.push_back(timestamp);
        filter_associations.push_back(filter_t_wi[i]);
        gt_associations.push_back(interpolated);
    }

    if (verbose) {
        std::cout << "find_trajectories_associations: max trajectory align dt: " << max_dt
                  << std::endl;
    }

    return std::make_tuple(timestamps_associations, filter_associations, gt_associations);
}

template <typename DerivedFilter, typename DerivedGT>
inline AssocResult
find_trajectories_associations_eigen(const Eigen::Ref<const Eigen::VectorXd> &filter_timestamps,
                                     const Eigen::DenseBase<DerivedFilter> &filter_t_wi,
                                     const Eigen::Ref<const Eigen::VectorXd> &gt_timestamps,
                                     const Eigen::DenseBase<DerivedGT> &gt_t_wi,
                                     double max_align_dt = 1e-1, bool verbose = true) {

    const size_t num_filter_timestamps = static_cast<size_t>(filter_timestamps.size());

    std::vector<int> kept_indices;
    kept_indices.reserve(num_filter_timestamps);

    double max_dt = 0.0;

    const double *gt_ts_ptr = gt_timestamps.data();
    const size_t gt_size = static_cast<size_t>(gt_timestamps.size());

    for (size_t i = 0; i < num_filter_timestamps; ++i) {
        double timestamp = filter_timestamps(static_cast<Eigen::Index>(i));

        // Find the index of the GT timestamp that is greater than the filter timestamp
        const double *upper = std::upper_bound(gt_ts_ptr, gt_ts_ptr + gt_size, timestamp);
        // Get the previous index so that gt_timestamps[j] <= timestamp < gt_timestamps[j+1]
        int j = static_cast<int>(upper - gt_ts_ptr) - 1;

        if (j < 0 || j >= static_cast<int>(gt_size) - 1)
            continue;

        double dt = timestamp - gt_ts_ptr[j];
        double dt_gt = gt_ts_ptr[j + 1] - gt_ts_ptr[j];
        double abs_dt = std::abs(dt);

        if (dt < 0 || dt_gt <= 0 || abs_dt > max_align_dt)
            continue;

        max_dt = std::max(max_dt, abs_dt);
        kept_indices.push_back(static_cast<int>(i));
    }

    const size_t M = kept_indices.size();
    AssocResult out{Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M)),
                    Eigen::Matrix<double, 3, Eigen::Dynamic>(3, static_cast<Eigen::Index>(M)),
                    Eigen::Matrix<double, 3, Eigen::Dynamic>(3, static_cast<Eigen::Index>(M)),
                    max_dt};

    size_t k = 0;
    for (int idx : kept_indices) {
        double timestamp = filter_timestamps(static_cast<Eigen::Index>(idx));
        const double *upper = std::upper_bound(gt_ts_ptr, gt_ts_ptr + gt_size, timestamp);
        int j = static_cast<int>(upper - gt_ts_ptr) - 1;

        double dt = timestamp - gt_ts_ptr[j];
        double dt_gt = gt_ts_ptr[j + 1] - gt_ts_ptr[j];
        double ratio = dt / dt_gt;

        out.timestamps(static_cast<Eigen::Index>(k)) = timestamp;
        out.filter_points.col(static_cast<Eigen::Index>(k)) =
            filter_t_wi.col(static_cast<Eigen::Index>(idx));
        out.gt_points.col(static_cast<Eigen::Index>(k)) =
            (1.0 - ratio) * gt_t_wi.col(j) + ratio * gt_t_wi.col(j + 1);
        ++k;
    }

    if (verbose) {
        std::cout << "find_trajectories_associations: max trajectory align dt: " << max_dt
                  << std::endl;
    }

    return out;
}

// ================================================================
// Alignment functions
// ================================================================

std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, bool>
align_3d_points_with_svd(const std::vector<Eigen::Vector3d> &gt_points,
                         const std::vector<Eigen::Vector3d> &est_points, bool find_scale = true) {
    if (gt_points.size() != est_points.size() || gt_points.empty())
        return {Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), false};

    size_t N = gt_points.size();
    Eigen::MatrixXd gt(3, N), est(3, N);
    for (size_t i = 0; i < N; ++i) {
        gt.col(i) = gt_points[i];
        est.col(i) = est_points[i];
    }

    Eigen::Vector3d mean_gt = gt.rowwise().mean();
    Eigen::Vector3d mean_est = est.rowwise().mean();
    gt.colwise() -= mean_gt;
    est.colwise() -= mean_est;

    Eigen::Matrix3d cov = gt * est.transpose();
    if (find_scale)
        cov /= static_cast<double>(N);

    double scale = 1.0;
    double variance_gt = 0;
    if (find_scale)
        variance_gt = gt.squaredNorm() / static_cast<double>(N);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0)
        S(2, 2) = -1;

    Eigen::Matrix3d R = U * S * V.transpose();
    if (find_scale)
        scale = variance_gt / (svd.singularValues().asDiagonal() * S).trace();

    Eigen::Vector3d t = mean_gt - scale * R * mean_est;

    Eigen::Matrix4d T_gt_est = Eigen::Matrix4d::Identity();
    T_gt_est.topLeftCorner<3, 3>() = scale * R;
    T_gt_est.topRightCorner<3, 1>() = t;

    Eigen::Matrix4d T_est_gt = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d sR_inv = (1.0 / scale) * R.transpose();
    T_est_gt.topLeftCorner<3, 3>() = sR_inv;
    T_est_gt.topRightCorner<3, 1>() = -sR_inv * t;

    return {T_gt_est, T_est_gt, true};
}

template <typename DerivedGT, typename DerivedEST>
inline std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, bool>
align_3d_points_with_svd_eigen(const Eigen::DenseBase<DerivedGT> &gt_points,
                               const Eigen::DenseBase<DerivedEST> &est_points, bool find_scale) {

    if (gt_points.cols() != est_points.cols() || gt_points.cols() < 3)
        return {Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), false};

    const Eigen::Index N = gt_points.cols();

    const Eigen::Vector3d mean_gt = gt_points.rowwise().mean();
    const Eigen::Vector3d mean_est = est_points.rowwise().mean();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    double variance_gt = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Vector3d centered_gt = gt_points.col(i) - mean_gt;
        const Eigen::Vector3d centered_est = est_points.col(i) - mean_est;
        cov.noalias() += centered_gt * centered_est.transpose();
        if (find_scale)
            variance_gt += centered_gt.squaredNorm();
    }
    if (find_scale) {
        cov /= static_cast<double>(N);
        variance_gt /= static_cast<double>(N);
    }

    double scale = 1.0;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0)
        S(2, 2) = -1;

    Eigen::Matrix3d R = U * S * V.transpose();
    const double eps = 1e-12;
    if (find_scale) {
        const double denom = (svd.singularValues().asDiagonal() * S).trace();
        if (std::isfinite(variance_gt) && std::isfinite(denom) && denom > eps &&
            variance_gt > eps) {
            scale = variance_gt / denom;
        } else {
            return {Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), false};
        }
    }

    Eigen::Vector3d t = mean_gt - scale * R * mean_est;

    Eigen::Matrix4d T_gt_est = Eigen::Matrix4d::Identity();
    T_gt_est.topLeftCorner<3, 3>() = scale * R;
    T_gt_est.topRightCorner<3, 1>() = t;

    Eigen::Matrix4d T_est_gt = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d sR_inv = (1.0 / scale) * R.transpose();
    T_est_gt.topLeftCorner<3, 3>() = sR_inv;
    T_est_gt.topRightCorner<3, 1>() = -sR_inv * t;

    const bool finite_ok = T_gt_est.allFinite() && T_est_gt.allFinite();
    return {T_gt_est, T_est_gt, finite_ok};
}

} // namespace trajectory_tools