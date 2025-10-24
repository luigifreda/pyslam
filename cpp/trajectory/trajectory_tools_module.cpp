#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;

// 1: Use zero-memory copy for mapping numpy arrays to Eigen matrices and GIL release
// 0: Use regular management with copy of numpy arrays to Eigen matrices and GIL release
#define USE_ZERO_MEMORY_COPY 1

#if USE_ZERO_MEMORY_COPY

namespace {

struct AssocResult {
    Eigen::VectorXd timestamps;                             // size M
    Eigen::Matrix<double, 3, Eigen::Dynamic> filter_points; // 3xM
    Eigen::Matrix<double, 3, Eigen::Dynamic> gt_points;     // 3xM
    double max_dt;
};

template <typename DerivedFilter, typename DerivedGT>
inline AssocResult
find_trajectories_associations_eigen(const Eigen::Ref<const Eigen::VectorXd> &filter_timestamps,
                                     const Eigen::DenseBase<DerivedFilter> &filter_t_wi,
                                     const Eigen::Ref<const Eigen::VectorXd> &gt_timestamps,
                                     const Eigen::DenseBase<DerivedGT> &gt_t_wi,
                                     double max_align_dt, bool verbose) {

    const size_t num_filter_timestamps = static_cast<size_t>(filter_timestamps.size());

    std::vector<int> kept_indices;
    kept_indices.reserve(num_filter_timestamps);

    double max_dt = 0.0;

    const double *gt_ts_ptr = gt_timestamps.data();
    const size_t gt_size = static_cast<size_t>(gt_timestamps.size());

    for (size_t i = 0; i < num_filter_timestamps; ++i) {
        double timestamp = filter_timestamps(static_cast<Eigen::Index>(i));

        const double *upper = std::upper_bound(gt_ts_ptr, gt_ts_ptr + gt_size, timestamp);
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

// Mapping helpers requiring contiguous buffers to ensure zero-copy
inline Eigen::Map<const Eigen::VectorXd> map_vector_1d(const py::array &arr) {
    if (!py::isinstance<py::array_t<double>>(arr))
        throw std::invalid_argument("Expected numpy.ndarray of dtype float64 for 1D vector");
    py::buffer_info info = arr.request();
    if (info.ndim != 1)
        throw std::invalid_argument("Expected 1D array");
    if (!(arr.flags() & py::array::c_style))
        throw std::invalid_argument("Vector must be C-contiguous for zero-copy mapping");
    return Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(info.ptr),
                                             static_cast<Eigen::Index>(info.shape[0]));
}

using StridesDyn = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using MapConst3xDyn = Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 0, StridesDyn>;

inline Eigen::Index stride_in_scalars(py::ssize_t stride_bytes) {
    constexpr py::ssize_t scalar_size = static_cast<py::ssize_t>(sizeof(double));
    if (stride_bytes % scalar_size != 0)
        throw std::invalid_argument("Array strides must be multiples of element size (float64)");
    return static_cast<Eigen::Index>(stride_bytes / scalar_size);
}

inline MapConst3xDyn map_points_array(const py::array &arr, const char *name) {
    if (!py::isinstance<py::array_t<double>>(arr))
        throw std::invalid_argument(std::string(name) +
                                    " must be a numpy.ndarray of dtype float64");
    py::buffer_info info = arr.request();
    if (info.ndim != 2)
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    const bool first_dim_three = info.shape[0] == 3;
    const bool second_dim_three = info.shape[1] == 3;
    if (!first_dim_three && !second_dim_three)
        throw std::invalid_argument(std::string(name) +
                                    " must have shape (3, N) or (N, 3) with float64 entries");

    const bool c_contig = static_cast<bool>(arr.flags() & py::array::c_style);
    const bool f_contig = static_cast<bool>(arr.flags() & py::array::f_style);
    if (!c_contig && !f_contig)
        throw std::invalid_argument(std::string(name) + " must be contiguous (C or Fortran)");

    const double *ptr = static_cast<const double *>(info.ptr);
    if (first_dim_three) {
        const Eigen::Index cols = static_cast<Eigen::Index>(info.shape[1]);
        const StridesDyn stride(stride_in_scalars(info.strides[1]),
                                stride_in_scalars(info.strides[0]));
        return MapConst3xDyn(ptr, 3, cols, stride);
    }

    const Eigen::Index cols = static_cast<Eigen::Index>(info.shape[0]);
    const StridesDyn stride(stride_in_scalars(info.strides[0]), stride_in_scalars(info.strides[1]));
    return MapConst3xDyn(ptr, 3, cols, stride);
}

} // namespace

PYBIND11_MODULE(trajectory_tools, m) {
    m.def(
        "find_trajectories_associations",
        [](const py::array &filter_timestamps, const py::array &filter_t_wi_arr,
           const py::array &gt_timestamps, const py::array &gt_t_wi_arr, double max_align_dt,
           bool verbose) {
            // Map numpy buffers without copy (timestamp vectors must be C-contiguous)
            auto ts_filter = map_vector_1d(filter_timestamps);
            auto ts_gt = map_vector_1d(gt_timestamps);
            auto filter_points = map_points_array(filter_t_wi_arr, "filter_t_wi");
            auto gt_points = map_points_array(gt_t_wi_arr, "gt_t_wi");
            AssocResult res;
            {
                py::gil_scoped_release release;
                res = find_trajectories_associations_eigen(ts_filter, filter_points, ts_gt,
                                                           gt_points, max_align_dt, verbose);
            }

            return py::make_tuple(res.timestamps, res.filter_points.transpose(),
                                  res.gt_points.transpose());
        },
        py::arg("filter_timestamps"), py::arg("filter_t_wi"), py::arg("gt_timestamps"),
        py::arg("gt_t_wi"), py::arg("max_align_dt") = 1e-1, py::arg("verbose") = true,
        R"pbdoc(
            Find associations between 3D trajectories timestamps in filter and gt.

            Args:
                filter_timestamps: numpy float64 array shaped (Nf,), C-contiguous
                filter_t_wi: numpy float64 array shaped (Nf, 3) C-contiguous or (3, Nf) Fortran-contiguous
                gt_timestamps: numpy float64 array shaped (Ng,), C-contiguous
                gt_t_wi: numpy float64 array shaped (Ng, 3) C-contiguous or (3, Ng) Fortran-contiguous
                max_align_dt: Maximum allowed time difference
                verbose: If true, prints alignment info

            Returns:
                Tuple (timestamps_associations [M], filter_associations [3xM], gt_associations [3xM])
        )pbdoc");

    m.def(
        "align_3d_points_with_svd",
        [](const py::array &gt_points_arr, const py::array &est_points_arr, bool find_scale) {
            Eigen::Matrix4d T_gt_est, T_est_gt;
            bool ok;
            auto gt_points = map_points_array(gt_points_arr, "gt_points");
            auto est_points = map_points_array(est_points_arr, "est_points");
            {
                py::gil_scoped_release release;
                std::tie(T_gt_est, T_est_gt, ok) =
                    align_3d_points_with_svd_eigen(gt_points, est_points, find_scale);
            }

            return py::make_tuple(T_gt_est, T_est_gt, ok);
        },
        py::arg("gt_points"), py::arg("est_points"), py::arg("find_scale") = true,
        R"pbdoc(
            Align corresponding 3D points using Kabsch-Umeyama SVD.

            Args:
                gt_points: numpy float64 array shaped (N, 3) C-contiguous or (3, N) Fortran-contiguous
                est_points: numpy float64 array shaped (N, 3) C-contiguous or (3, N) Fortran-contiguous
                find_scale: Estimate similarity scale (Sim(3))

            Returns:
                Tuple (T_gt_est [4x4], T_est_gt [4x4], success)
        )pbdoc");
}

#else

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
        std::cout << "find_trajectories_associations: max trajectory align dt: " << max_dt
                  << std::endl;
    }

    return std::make_tuple(timestamps_associations, filter_associations, gt_associations);
}

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

PYBIND11_MODULE(trajectory_tools, m) {
    m.def("find_trajectories_associations", &find_trajectories_associations,
          py::arg("filter_timestamps"), py::arg("filter_t_wi"), py::arg("gt_timestamps"),
          py::arg("gt_t_wi"), py::arg("max_align_dt") = 1e-1, py::arg("verbose") = true,
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

    m.def("align_3d_points_with_svd", &align_3d_points_with_svd, py::arg("gt_points"),
          py::arg("est_points"), py::arg("find_scale") = true,
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

#endif