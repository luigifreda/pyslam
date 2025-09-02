#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace py = pybind11;

std::tuple<std::vector<double>,
           std::vector<Eigen::Vector3d>,
           std::vector<Eigen::Vector3d>>
find_trajectories_associations(const std::vector<double>& filter_timestamps,
                                const std::vector<Eigen::Vector3d>& filter_t_wi,
                                const std::vector<double>& gt_timestamps,
                                const std::vector<Eigen::Vector3d>& gt_t_wi,
                                double max_align_dt = 1e-1,
                                bool verbose = true) {

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

        auto upper = std::upper_bound(gt_timestamps.begin(), gt_timestamps.end(), timestamp);
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
        std::cout << "find_trajectories_associations: max trajectory align dt: " << max_dt << std::endl;
    }

    return std::make_tuple(timestamps_associations, filter_associations, gt_associations);
}

std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, bool>
align_3d_points_with_svd(const std::vector<Eigen::Vector3d>& gt_points,
                         const std::vector<Eigen::Vector3d>& est_points,
                         bool find_scale = true) {
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
    T_gt_est.topLeftCorner<3,3>() = scale * R;
    T_gt_est.topRightCorner<3,1>() = t;

    Eigen::Matrix4d T_est_gt = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d sR_inv = (1.0 / scale) * R.transpose();
    T_est_gt.topLeftCorner<3,3>() = sR_inv;
    T_est_gt.topRightCorner<3,1>() = -sR_inv * t;

    return {T_gt_est, T_est_gt, true};
}

PYBIND11_MODULE(trajectory_tools, m) {
    m.def("find_trajectories_associations", &find_trajectories_associations,
          py::arg("filter_timestamps"),
          py::arg("filter_t_wi"),
          py::arg("gt_timestamps"),
          py::arg("gt_t_wi"),
          py::arg("max_align_dt") = 1e-1,
          py::arg("verbose") = true,
          R"pbdoc(
              Find associations between 3D trajectories timestamps in filter and gt.

              Args:
                  filter_timestamps: List of timestamps in seconds
                  filter_t_wi: List of 3D filter poses
                  gt_timestamps: List of timestamps in seconds
                  gt_t_wi: List of 3D ground truth poses
                  max_align_dt: Maximum allowed time difference
                  verbose: If true, prints alignment info

              Returns:
                  Tuple of (timestamps_associations, filter_associations, gt_associations)
          )pbdoc");

    m.def("align_3d_points_with_svd", &align_3d_points_with_svd,
        py::arg("gt_points"),
        py::arg("est_points"),
        py::arg("find_scale") = true,
        R"pbdoc(
            Align corresponding 3D points using Kabsch–Umeyama SVD.

            Args:
                gt_points: List of 3D ground truth points
                est_points: List of 3D estimated points
                find_scale: Estimate similarity scale (Sim(3))

            Returns:
                Tuple (T_gt_est [4x4], T_est_gt [4x4], success)
        )pbdoc");
}